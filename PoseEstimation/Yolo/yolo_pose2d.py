import cv2
import argparse
import numpy as np
import time
import sys
import math
import os
import shutil
import torch
from ultralytics import YOLO

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QSizePolicy, 
                             QDoubleSpinBox, QLineEdit, QSpinBox)
from PyQt5.QtCore import QThread, pyqtSignal, pyqtSlot, Qt
from PyQt5.QtGui import QImage, QPixmap

import motion_sender

# ==============================================================================
# 1. Settings & Math Filtering
# ==============================================================================

# Default OSC Settings
DEFAULT_OSC_IP = "127.0.0.1"
DEFAULT_OSC_PORT = 9007

class OneEuroFilter:
    def __init__(self, t0, x0, dx0=0.0, min_cutoff=1.0, beta=0.007, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev = x0
        self.dx_prev = dx0
        self.t_prev = t0

    def smoothing_factor(self, t_e, cutoff):
        r = 2 * math.pi * cutoff * t_e
        return r / (r + 1)

    def exponential_smoothing(self, a, x, x_prev):
        return a * x + (1 - a) * x_prev

    def __call__(self, t, x):
        t_e = t - self.t_prev
        if t_e <= 0: return x
        a_d = self.smoothing_factor(t_e, self.d_cutoff)
        dx = (x - self.x_prev) / t_e
        dx_hat = self.exponential_smoothing(a_d, dx, self.dx_prev)
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self.smoothing_factor(t_e, cutoff)
        x_hat = self.exponential_smoothing(a, x, self.x_prev)
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t
        return x_hat

class PoseFilter2D:
    def __init__(self, num_nodes, min_cutoff=0.1, beta=5.0):
        self.num_nodes = num_nodes
        self.filters = None
        self.min_cutoff = min_cutoff
        self.beta = beta

    def update_params(self, min_cutoff, beta):
        self.min_cutoff = min_cutoff
        self.beta = beta
        if self.filters is not None:
            for i in range(self.num_nodes):
                for j in range(2): # 2D for YOLO (x, y)
                    self.filters[i][j].min_cutoff = min_cutoff
                    self.filters[i][j].beta = beta

    def __call__(self, t, pos_2d):
        if self.filters is None:
            self.filters = [[OneEuroFilter(t, pos_2d[i, j], min_cutoff=self.min_cutoff, beta=self.beta) 
                            for j in range(2)] for i in range(self.num_nodes)]
            return pos_2d
        filtered_pos = np.zeros_like(pos_2d)
        for i in range(self.num_nodes):
            for j in range(2):
                filtered_pos[i, j] = self.filters[i][j](t, pos_2d[i, j])
        return filtered_pos

# ==============================================================================
# 2. GUI & THREADING
# ==============================================================================

class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(QImage)
    update_fps_signal = pyqtSignal(float)

    def __init__(self, args):
        super().__init__()
        self.args = args
        self._run_flag = True
        self._new_osc_settings = None
        
        # Select device automatically
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
        
        print(f"Using {self.device.upper()} device")
        
        # Init model with auto-download to models/ directory
        model_name = f"yolov8{self.args.model}-pose.pt"
        model_dir = "models"
        model_path = os.path.join(model_dir, model_name)
        
        os.makedirs(model_dir, exist_ok=True)
        
        if self.args.local_model:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"--local_model flag used, but {model_path} not found.")
            print(f"Loading {model_path}...")
            self.model = YOLO(model_path).to(self.device)
        else:
            if not os.path.exists(model_path):
                print(f"Model not found in {model_dir}/. Downloading {model_name}...")
                self.model = YOLO(model_name).to(self.device)
                if os.path.exists(model_name):
                    shutil.move(model_name, model_path)
                    print(f"Successfully moved {model_name} into {model_dir}/ directory.")
            else:
                print(f"Loading existing {model_path}...")
                self.model = YOLO(model_path).to(self.device)

        # OSC Setup
        motion_sender.config["ip"] = DEFAULT_OSC_IP
        motion_sender.config["port"] = DEFAULT_OSC_PORT
        self.osc_sender = motion_sender.OscSender(motion_sender.config)

        # Filtering variables
        self.filters = {} # Dictionary mapping skel_index to PoseFilter2D
        self.min_cutoff = 0.1
        self.beta = 5.0

    def update_osc_settings(self, ip, port):
        """Thread-safe way to flag that OSC settings need updating."""
        self._new_osc_settings = (ip, port)

    def set_filter_params(self, min_cutoff, beta):
        self.min_cutoff = min_cutoff
        self.beta = beta
        for skel_filter in self.filters.values():
            skel_filter.update_params(min_cutoff, beta)

    def run(self):
        source = int(self.args.source) if self.args.source.isdigit() else self.args.source
        cap = cv2.VideoCapture(source)

        prev_time = time.time()
        fps_filter = 0.0

        while self._run_flag and cap.isOpened():
            success, frame = cap.read()
            if not success: 
                break 

            # Check for GUI updates to OSC Target
            if self._new_osc_settings is not None:
                new_ip, new_port = self._new_osc_settings
                print(f"Updating OSC Sender to {new_ip}:{new_port}")
                motion_sender.config["ip"] = new_ip
                motion_sender.config["port"] = new_port
                self.osc_sender = motion_sender.OscSender(motion_sender.config)
                self._new_osc_settings = None

            curr_time = time.time()
            fps = 1.0 / (curr_time - prev_time + 1e-6)
            prev_time = curr_time
            fps_filter = 0.9 * fps_filter + 0.1 * fps
            self.update_fps_signal.emit(fps_filter)

            # YOLO inference (we disable printing to keep terminal clean)
            results = self.model(frame, verbose=False)[0]
            
            timestamp_sec = time.time()

            # Process keypoints if humans are detected
            if len(results.keypoints) > 0 and results.keypoints.xyn is not None:
                keypoints_np = results.keypoints.xyn.cpu().numpy()
                keypoints_conf_np = results.keypoints.conf.cpu().numpy() if results.keypoints.conf is not None else None

                num_skeletons = keypoints_np.shape[0]
                if num_skeletons > 0:
                    num_nodes = keypoints_np.shape[1]

                    for skel_index in range(num_skeletons):
                        # Instantiate filter for this person if it doesn't exist
                        if skel_index not in self.filters:
                            self.filters[skel_index] = PoseFilter2D(num_nodes, min_cutoff=self.min_cutoff, beta=self.beta)

                        raw_pos = keypoints_np[skel_index]
                        filtered_pos = self.filters[skel_index](timestamp_sec, raw_pos)

                        # Send via OSC
                        self.osc_sender.send(f"/mocap/{skel_index}/joint/pos_world", filtered_pos)
                        
                        if keypoints_conf_np is not None:
                            self.osc_sender.send(f"/mocap/{skel_index}/joint/visibility", keypoints_conf_np[skel_index])

            # Render YOLO's built-in visualizer plot
            annotated_frame = results.plot()
            annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            
            # Emit to GUI
            h, w, ch = annotated_frame_rgb.shape
            bytes_per_line = ch * w
            q_img = QImage(annotated_frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.change_pixmap_signal.emit(q_img)

        cap.release()

    def stop(self):
        self._run_flag = False
        self.wait()


class MainWindow(QMainWindow):
    def __init__(self, args):
        super().__init__()
        self.setWindowTitle("YOLO Pose Estimation")
        self.resize(850, 650)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.layout = QVBoxLayout(self.central_widget)
        self.layout.setContentsMargins(0, 0, 0, 0) 

        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.image_label.setStyleSheet("background-color: black;") 
        self.layout.addWidget(self.image_label, stretch=1) 

        # --- Control Row 1: Filter Setup ---
        self.filter_layout = QHBoxLayout()
        self.filter_layout.setContentsMargins(10, 10, 10, 0)

        self.filter_layout.addWidget(QLabel("One-Euro Filter - Min Cutoff:"))
        self.cutoff_spin = QDoubleSpinBox()
        self.cutoff_spin.setRange(0.001, 5.0)
        self.cutoff_spin.setSingleStep(0.05)
        self.cutoff_spin.setDecimals(3)
        self.cutoff_spin.setValue(0.1)
        self.cutoff_spin.valueChanged.connect(self.on_filter_changed)
        self.filter_layout.addWidget(self.cutoff_spin)

        self.filter_layout.addWidget(QLabel("Beta (Speed):"))
        self.beta_spin = QDoubleSpinBox()
        self.beta_spin.setRange(0.0, 50.0)
        self.beta_spin.setSingleStep(0.5)
        self.beta_spin.setDecimals(2)
        self.beta_spin.setValue(5.0)
        self.beta_spin.valueChanged.connect(self.on_filter_changed)
        self.filter_layout.addWidget(self.beta_spin)

        self.filter_layout.addStretch(1)
        self.layout.addLayout(self.filter_layout)

        # --- Control Row 2: OSC Settings ---
        self.osc_layout = QHBoxLayout()
        self.osc_layout.setContentsMargins(10, 5, 10, 0)

        self.osc_layout.addWidget(QLabel("OSC IP:"))
        self.ip_input = QLineEdit(DEFAULT_OSC_IP)
        self.ip_input.setFixedWidth(120)
        self.osc_layout.addWidget(self.ip_input)

        self.osc_layout.addWidget(QLabel("Port:"))
        self.port_spin = QSpinBox()
        self.port_spin.setRange(1024, 65535)
        self.port_spin.setValue(DEFAULT_OSC_PORT)
        self.port_spin.setFixedWidth(80)
        self.osc_layout.addWidget(self.port_spin)

        self.btn_apply_osc = QPushButton("Apply OSC")
        self.btn_apply_osc.clicked.connect(self.on_apply_osc)
        self.osc_layout.addWidget(self.btn_apply_osc)

        self.osc_layout.addStretch(1)
        self.layout.addLayout(self.osc_layout)

        # --- Control Row 3: Standard Controls Layout ---
        self.controls_layout = QHBoxLayout()
        self.controls_layout.setContentsMargins(10, 5, 10, 10) 

        self.btn_exit = QPushButton("Exit")
        self.btn_exit.clicked.connect(self.close)
        self.btn_exit.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.controls_layout.addWidget(self.btn_exit)

        self.controls_layout.addStretch(1)

        self.fps_label = QLabel("FPS: 0.0")
        self.fps_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.controls_layout.addWidget(self.fps_label)

        self.layout.addLayout(self.controls_layout)

        # Initialize and start thread
        self.thread = VideoThread(args)
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.update_fps_signal.connect(self.update_fps)
        self.thread.start()

    def closeEvent(self, event):
        self.thread.stop()
        event.accept()

    @pyqtSlot(QImage)
    def update_image(self, qt_img):
        self.image_label.setPixmap(QPixmap.fromImage(qt_img).scaled(
            self.image_label.width(), self.image_label.height(), Qt.KeepAspectRatio))

    @pyqtSlot(float)
    def update_fps(self, fps):
        self.fps_label.setText(f"FPS: {fps:.1f}")

    def on_filter_changed(self):
        cutoff = self.cutoff_spin.value()
        beta = self.beta_spin.value()
        self.thread.set_filter_params(cutoff, beta)

    def on_apply_osc(self):
        ip = self.ip_input.text().strip()
        port = self.port_spin.value()
        self.thread.update_osc_settings(ip, port)


def main():
    parser = argparse.ArgumentParser(description="YOLO Pose OSC Sender with 1-Euro Filter")
    parser.add_argument("--source", type=str, default="0", help="Webcam index (e.g. 0) or path to video file")
    parser.add_argument("--model", type=str, choices=["n", "s", "m", "l", "x"], default="s", 
                        help="YOLO model size (n=nano, s=small, m=medium, l=large, x=extra-large)")
    parser.add_argument("--local_model", action="store_true", help="Look for model in local 'models/' directory instead of auto-downloading")
    args = parser.parse_args()

    app = QApplication(sys.argv)
    window = MainWindow(args)
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()