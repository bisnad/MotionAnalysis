import cv2
import argparse
import numpy as np
import time
import sys
import os
import shutil
import torch
from ultralytics import YOLO, YOLOWorld

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QSizePolicy, 
                             QLineEdit, QDoubleSpinBox, QSpinBox)
from PyQt5.QtCore import QThread, pyqtSignal, pyqtSlot, Qt
from PyQt5.QtGui import QImage, QPixmap

import motion_sender

# ==============================================================================
# 1. Settings
# ==============================================================================

# Default OSC Settings
DEFAULT_OSC_IP = "127.0.0.1"
DEFAULT_OSC_PORT = 9007

# Object Class Settings
classes_list = None

# ==============================================================================
# 2. THREADING & INFERENCE
# ==============================================================================

class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(QImage)
    update_fps_signal = pyqtSignal(float)

    def __init__(self, args):
        super().__init__()
        self.args = args
        self._run_flag = True
        
        # Thread-safe update flags
        self._new_classes = None
        self._new_osc_settings = None
        self.conf_threshold = args.conf
        
        # Select device automatically
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
        
        print(f"Using {self.device.upper()} device")
        
        # Init YOLO-World model
        model_name = f"yolov8{self.args.model}-worldv2.pt"
        model_dir = "models"
        model_path = os.path.join(model_dir, model_name)
        
        os.makedirs(model_dir, exist_ok=True)
        
        if self.args.local_model:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"--local_model flag used, but {model_path} not found.")
            print(f"Loading {model_path}...")
            self.model = YOLOWorld(model_path).to(self.device)
        else:
            if not os.path.exists(model_path):
                print(f"Model not found in {model_dir}/. Downloading {model_name}...")
                self.model = YOLOWorld(model_name).to(self.device)
                if os.path.exists(model_name):
                    shutil.move(model_name, model_path)
                    print(f"Successfully moved {model_name} into {model_dir}/ directory.")
            else:
                print(f"Loading existing {model_path}...")
                self.model = YOLOWorld(model_path).to(self.device)

        # Initial OSC Setup
        motion_sender.config["ip"] = DEFAULT_OSC_IP
        motion_sender.config["port"] = DEFAULT_OSC_PORT
        self.osc_sender = motion_sender.OscSender(motion_sender.config)

        # OSC Class Setup
        if classes_list is not None:
            self.update_classes(classes_list)

    def update_classes(self, new_classes_list):
        self._new_classes = new_classes_list

    def update_osc_settings(self, ip, port):
        self._new_osc_settings = (ip, port)

    def set_conf_threshold(self, conf):
        self.conf_threshold = conf

    def run(self):
        source = int(self.args.source) if self.args.source.isdigit() else self.args.source
        cap = cv2.VideoCapture(source)

        prev_time = time.time()
        fps_filter = 0.0

        while self._run_flag and cap.isOpened():
            success, frame = cap.read()
            if not success: 
                break 

            # Check for GUI updates
            if self._new_classes is not None:
                print(f"Updating YOLO-World classes to: {self._new_classes}")
                self.model.set_classes(self._new_classes)
                self._new_classes = None

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

            # YOLO inference using dynamic confidence threshold
            results = self.model(frame, verbose=False, conf=self.conf_threshold)[0]
            
            # Process and send bounding boxes over OSC
            boxes = results.boxes
            if boxes is not None and len(boxes) > 0:
                class_names = results.names
                boxes_cls = boxes.cls.cpu().numpy() 
                boxes_conf = boxes.conf.cpu().numpy() 
                boxes_xyxyn = boxes.xyxyn.cpu().numpy() 
                
                for box_index in range(boxes_xyxyn.shape[0]):
                    osc_parameters = np.concatenate((boxes_cls[box_index:box_index+1], boxes_conf[box_index:box_index+1], boxes_xyxyn[box_index]), axis=0)
                    self.osc_sender.send(f"/object/box/{box_index}/bbox", osc_parameters)

                    class_id = int(boxes_cls[box_index])
                    osc_class_name = class_names[class_id]
                    self.osc_sender.send(f"/object/box/{box_index}/class", osc_class_name)

            # Render YOLO visualizer
            annotated_frame = results.plot()
            annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            
            h, w, ch = annotated_frame_rgb.shape
            bytes_per_line = ch * w
            q_img = QImage(annotated_frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.change_pixmap_signal.emit(q_img)

        cap.release()

    def stop(self):
        self._run_flag = False
        self.wait()

# ==============================================================================
# 3. GUI SETUP
# ==============================================================================

class MainWindow(QMainWindow):
    def __init__(self, args):
        super().__init__()
        self.setWindowTitle("YOLO-World Custom Object Detection")
        self.resize(850, 650)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.layout = QVBoxLayout(self.central_widget)
        self.layout.setContentsMargins(0, 0, 0, 0) 

        # Video Display
        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.image_label.setStyleSheet("background-color: black;") 
        self.layout.addWidget(self.image_label, stretch=1) 

        # --- Control Row 1: Custom Classes ---
        self.class_layout = QHBoxLayout()
        self.class_layout.setContentsMargins(10, 10, 10, 0)

        self.class_layout.addWidget(QLabel("Custom Classes (comma-separated):"))
        self.class_input = QLineEdit()
        self.class_input.setPlaceholderText("e.g. dancer, microphone, water bottle")
        self.class_layout.addWidget(self.class_input)

        self.btn_update_classes = QPushButton("Set Classes")
        self.btn_update_classes.clicked.connect(self.on_update_classes)
        self.class_layout.addWidget(self.btn_update_classes)

        self.layout.addLayout(self.class_layout)

        # --- Control Row 2: Confidence and OSC Settings ---
        self.settings_layout = QHBoxLayout()
        self.settings_layout.setContentsMargins(10, 5, 10, 0)

        # Confidence Threshold
        self.settings_layout.addWidget(QLabel("Confidence:"))
        self.conf_spin = QDoubleSpinBox()
        self.conf_spin.setRange(0.01, 1.0)
        self.conf_spin.setSingleStep(0.05)
        self.conf_spin.setDecimals(2)
        self.conf_spin.setValue(args.conf)
        self.conf_spin.valueChanged.connect(self.on_conf_changed)
        self.settings_layout.addWidget(self.conf_spin)
        
        self.settings_layout.addSpacing(20)

        # OSC IP
        self.settings_layout.addWidget(QLabel("OSC IP:"))
        self.ip_input = QLineEdit(DEFAULT_OSC_IP)
        self.ip_input.setFixedWidth(120)
        self.settings_layout.addWidget(self.ip_input)

        # OSC Port
        self.settings_layout.addWidget(QLabel("Port:"))
        self.port_spin = QSpinBox()
        self.port_spin.setRange(1024, 65535)
        self.port_spin.setValue(DEFAULT_OSC_PORT)
        self.port_spin.setFixedWidth(80)
        self.settings_layout.addWidget(self.port_spin)

        # Apply OSC Button
        self.btn_apply_osc = QPushButton("Apply OSC")
        self.btn_apply_osc.clicked.connect(self.on_apply_osc)
        self.settings_layout.addWidget(self.btn_apply_osc)

        self.settings_layout.addStretch(1)
        self.layout.addLayout(self.settings_layout)

        # --- Control Row 3: Bottom Controls ---
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

    def on_update_classes(self):
        text = self.class_input.text()
        if text.strip():
            new_classes = [c.strip() for c in text.split(",") if c.strip()]
            self.thread.update_classes(new_classes)
            
    def on_conf_changed(self, value):
        self.thread.set_conf_threshold(value)
        
    def on_apply_osc(self):
        ip = self.ip_input.text().strip()
        port = self.port_spin.value()
        self.thread.update_osc_settings(ip, port)


def main():
    parser = argparse.ArgumentParser(description="YOLO-World Custom Classes OSC Sender")
    parser.add_argument("--source", type=str, default="0", help="Webcam index (e.g. 0) or path to video file")
    parser.add_argument("--model", type=str, choices=["s", "m", "l", "x"], default="s", 
                        help="YOLO-World model size (s=small, m=medium, l=large, x=extra-large)")
    parser.add_argument("--conf", type=float, default=0.3, help="Confidence threshold for detections")
    parser.add_argument("--local_model", action="store_true", help="Look for model in local 'models/' directory instead of auto-downloading")
    args = parser.parse_args()

    app = QApplication(sys.argv)
    window = MainWindow(args)
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()