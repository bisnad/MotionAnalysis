import cv2
import argparse
import numpy as np
import time
import sys
import math
import os
import shutil
import threading
import torch
from ultralytics import YOLO

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout,
    QHBoxLayout, QPushButton, QLabel, QSizePolicy,
    QDoubleSpinBox, QLineEdit, QSpinBox
)
from PyQt5.QtCore import QThread, pyqtSlot, Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap

import motion_sender


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
        if t_e <= 0:
            return x

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
                for j in range(2):
                    self.filters[i][j].min_cutoff = min_cutoff
                    self.filters[i][j].beta = beta

    def __call__(self, t, pos_2d):
        if self.filters is None:
            self.filters = [
                [OneEuroFilter(t, pos_2d[i, j], min_cutoff=self.min_cutoff, beta=self.beta) for j in range(2)]
                for i in range(self.num_nodes)
            ]
            return pos_2d

        filtered_pos = np.zeros_like(pos_2d)
        for i in range(self.num_nodes):
            for j in range(2):
                filtered_pos[i, j] = self.filters[i][j](t, pos_2d[i, j])
        return filtered_pos


class LatestFrameReader:
    def __init__(self, cap):
        self.cap = cap
        self.lock = threading.Lock()
        self.cond = threading.Condition(self.lock)
        self.running = True
        self.latest_frame = None
        self.latest_id = 0
        self.read_fail = False
        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()

    def _reader(self):
        while self.running:
            success, frame = self.cap.read()
            if not success:
                with self.lock:
                    self.read_fail = True
                    self.cond.notify_all()
                break

            with self.lock:
                self.latest_frame = frame
                self.latest_id += 1
                self.cond.notify_all()

    def get_latest(self, last_id=None, block=True, timeout=1.0):
        with self.lock:
            if block:
                end_time = time.perf_counter() + timeout
                while self.running and not self.read_fail:
                    if self.latest_frame is not None and (last_id is None or self.latest_id != last_id):
                        break
                    remaining = end_time - time.perf_counter()
                    if remaining <= 0:
                        return False, None, last_id
                    self.cond.wait(timeout=remaining)

            if self.latest_frame is None:
                return False, None, last_id

            if last_id is not None and self.latest_id == last_id:
                return False, None, last_id

            return True, self.latest_frame.copy(), self.latest_id

    def stop(self):
        self.running = False
        with self.lock:
            self.cond.notify_all()
        self.thread.join(timeout=1.0)


class VideoThread(QThread):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self._run_flag = True
        self._new_osc_settings = None

        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        print(f"Using {self.device.upper()} device")

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

        motion_sender.config["ip"] = DEFAULT_OSC_IP
        motion_sender.config["port"] = DEFAULT_OSC_PORT
        self.osc_sender = motion_sender.OscSender(motion_sender.config)

        self.filters = {}
        self.min_cutoff = 0.1
        self.beta = 5.0

        self.max_visual_fps = max(1.0, float(args.display_fps))
        self.frame_mode = args.frame_mode

        self._frame_lock = threading.Lock()
        self._stats_lock = threading.Lock()

        self._latest_frame_rgb = None
        self._latest_frame_seq = 0
        self._processing_fps = 0.0
        self._source_fps = 0.0
        self._skipped_frames = 0

    def update_osc_settings(self, ip, port):
        self._new_osc_settings = (ip, port)

    def set_filter_params(self, min_cutoff, beta):
        self.min_cutoff = min_cutoff
        self.beta = beta
        for skel_filter in self.filters.values():
            skel_filter.update_params(min_cutoff, beta)

    def get_latest_frame(self, last_seq=-1):
        with self._frame_lock:
            if self._latest_frame_rgb is None or self._latest_frame_seq == last_seq:
                return None, last_seq
            return self._latest_frame_rgb.copy(), self._latest_frame_seq

    def get_stats(self):
        with self._stats_lock:
            return self._processing_fps, self._source_fps, self._skipped_frames

    def _store_latest_frame(self, frame_rgb):
        with self._frame_lock:
            self._latest_frame_rgb = frame_rgb.copy()
            self._latest_frame_seq += 1

    def _set_stats(self, processing_fps=None, source_fps=None, skipped_inc=0):
        with self._stats_lock:
            if processing_fps is not None:
                self._processing_fps = processing_fps
            if source_fps is not None:
                self._source_fps = source_fps
            self._skipped_frames += skipped_inc

    def _process_frame(self, frame):
        with torch.no_grad():
            results = self.model(frame, verbose=False)[0]

        timestamp_sec = time.time()
        keypoints = results.keypoints

        if keypoints is not None and keypoints.xyn is not None and len(keypoints) > 0:
            keypoints_np = keypoints.xyn.cpu().numpy()
            keypoints_conf_np = keypoints.conf.cpu().numpy() if keypoints.conf is not None else None

            num_skeletons = keypoints_np.shape[0]
            if num_skeletons > 0:
                num_nodes = keypoints_np.shape[1]

                for skel_index in range(num_skeletons):
                    if skel_index not in self.filters:
                        self.filters[skel_index] = PoseFilter2D(
                            num_nodes,
                            min_cutoff=self.min_cutoff,
                            beta=self.beta
                        )

                    raw_pos = keypoints_np[skel_index]
                    filtered_pos = self.filters[skel_index](timestamp_sec, raw_pos)

                    self.osc_sender.send(f"/mocap/{skel_index}/joint/pos_world", filtered_pos)

                    if keypoints_conf_np is not None:
                        self.osc_sender.send(
                            f"/mocap/{skel_index}/joint/visibility",
                            keypoints_conf_np[skel_index]
                        )

        return results

    def run(self):
        source = int(self.args.source) if self.args.source.isdigit() else self.args.source
        cap = cv2.VideoCapture(source)

        if not cap.isOpened():
            print(f"Failed to open source: {self.args.source}")
            return

        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        is_file = not str(self.args.source).isdigit()
        source_fps = 0.0

        if is_file:
            source_fps = cap.get(cv2.CAP_PROP_FPS)
            if not source_fps or source_fps <= 1e-6:
                source_fps = 30.0
        else:
            source_fps = cap.get(cv2.CAP_PROP_FPS)
            if not source_fps or source_fps <= 1e-6:
                source_fps = 0.0

        self._set_stats(source_fps=source_fps)

        frame_period = 1.0 / source_fps if source_fps > 1e-6 else None
        next_frame_time = time.perf_counter() if (is_file and frame_period is not None) else None

        prev_loop_time = time.perf_counter()
        fps_filter = 0.0
        last_visual_time = 0.0
        visual_period = 1.0 / self.max_visual_fps

        latest_reader = None
        last_camera_frame_id = None

        if (not is_file) and self.frame_mode == "realtime":
            latest_reader = LatestFrameReader(cap)

        try:
            while self._run_flag:
                if self._new_osc_settings is not None:
                    new_ip, new_port = self._new_osc_settings
                    print(f"Updating OSC Sender to {new_ip}:{new_port}")
                    motion_sender.config["ip"] = new_ip
                    motion_sender.config["port"] = new_port
                    self.osc_sender = motion_sender.OscSender(motion_sender.config)
                    self._new_osc_settings = None

                if is_file:
                    success, frame = cap.read()
                    if not success:
                        break
                else:
                    if self.frame_mode == "all":
                        success, frame = cap.read()
                        if not success:
                            break
                    else:
                        success, frame, new_id = latest_reader.get_latest(last_id=last_camera_frame_id, block=True, timeout=1.0)
                        if not success:
                            if latest_reader.read_fail:
                                break
                            else:
                                continue

                        if last_camera_frame_id is not None and new_id > last_camera_frame_id + 1:
                            self._set_stats(skipped_inc=(new_id - last_camera_frame_id - 1))
                        last_camera_frame_id = new_id

                results = self._process_frame(frame)

                now = time.perf_counter()
                if now - last_visual_time >= visual_period:
                    annotated_frame = results.plot()
                    annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                    self._store_latest_frame(annotated_frame_rgb)
                    last_visual_time = now

                loop_end = time.perf_counter()
                inst_fps = 1.0 / max(loop_end - prev_loop_time, 1e-6)
                prev_loop_time = loop_end
                fps_filter = inst_fps if fps_filter == 0.0 else (0.9 * fps_filter + 0.1 * inst_fps)
                self._set_stats(processing_fps=fps_filter)

                if is_file and frame_period is not None:
                    if self.frame_mode == "all":
                        pass
                    else:
                        next_frame_time += frame_period
                        delay = next_frame_time - time.perf_counter()

                        if delay > 0:
                            time.sleep(delay)
                        else:
                            frames_behind = int((-delay) / frame_period)
                            skipped_now = 0

                            for _ in range(frames_behind):
                                if cap.grab():
                                    skipped_now += 1
                                    next_frame_time += frame_period
                                else:
                                    break

                            if skipped_now > 0:
                                self._set_stats(skipped_inc=skipped_now)

                            if next_frame_time < time.perf_counter() - frame_period:
                                next_frame_time = time.perf_counter()
        finally:
            if latest_reader is not None:
                latest_reader.stop()
            cap.release()

    def stop(self):
        self._run_flag = False
        self.wait()


class MainWindow(QMainWindow):
    def __init__(self, args):
        super().__init__()
        self.setWindowTitle("YOLO Pose Estimation")
        self.resize(900, 680)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.layout = QVBoxLayout(self.central_widget)
        self.layout.setContentsMargins(0, 0, 0, 0)

        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.image_label.setStyleSheet("background-color: black;")
        self.layout.addWidget(self.image_label, stretch=1)

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

        self.controls_layout = QHBoxLayout()
        self.controls_layout.setContentsMargins(10, 5, 10, 10)

        self.btn_exit = QPushButton("Exit")
        self.btn_exit.clicked.connect(self.close)
        self.btn_exit.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.controls_layout.addWidget(self.btn_exit)

        self.controls_layout.addStretch(1)

        self.fps_label = QLabel("Proc FPS: 0.0 | View FPS: 0.0 | Src FPS: 0.0 | Skipped: 0")
        self.fps_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.controls_layout.addWidget(self.fps_label)

        self.layout.addLayout(self.controls_layout)

        self.thread = VideoThread(args)
        self.thread.start()

        self.last_frame_seq = -1
        self.display_fps = 0.0
        self.prev_display_time = None

        self.ui_timer = QTimer(self)
        self.ui_timer.setInterval(max(1, int(1000 / max(1.0, float(args.ui_fps)))))
        self.ui_timer.timeout.connect(self.refresh_ui)
        self.ui_timer.start()

    def closeEvent(self, event):
        self.ui_timer.stop()
        self.thread.stop()
        event.accept()

    @pyqtSlot()
    def refresh_ui(self):
        frame_rgb, seq = self.thread.get_latest_frame(self.last_frame_seq)

        if frame_rgb is not None:
            self.last_frame_seq = seq

            h, w, ch = frame_rgb.shape
            bytes_per_line = ch * w
            q_img = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()

            pixmap = QPixmap.fromImage(q_img).scaled(
                self.image_label.size(),
                Qt.KeepAspectRatio,
                Qt.FastTransformation
            )
            self.image_label.setPixmap(pixmap)

            now = time.perf_counter()
            if self.prev_display_time is not None:
                inst_display_fps = 1.0 / max(now - self.prev_display_time, 1e-6)
                self.display_fps = (
                    inst_display_fps if self.display_fps == 0.0
                    else 0.9 * self.display_fps + 0.1 * inst_display_fps
                )
            self.prev_display_time = now

        proc_fps, src_fps, skipped = self.thread.get_stats()
        self.fps_label.setText(
            f"Proc FPS: {proc_fps:.1f} | View FPS: {self.display_fps:.1f} | "
            f"Src FPS: {src_fps:.1f} | Skipped: {skipped}"
        )

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
    parser.add_argument("--source", type=str, default="0",
                        help="Webcam index (e.g. 0) or path to video file")
    parser.add_argument("--model", type=str, choices=["n", "s", "m", "l", "x"], default="s",
                        help="YOLO pose model size")
    parser.add_argument("--display_fps", type=float, default=20.0,
                        help="Maximum rate for generating annotated frames for display")
    parser.add_argument("--ui_fps", type=float, default=30.0,
                        help="GUI refresh rate used to pull and display the latest frame")
    parser.add_argument("--frame_mode", type=str, choices=["all", "realtime"], default="realtime",
                        help="all = process every frame in order; realtime = drop stale frames to stay current")
    parser.add_argument("--local_model", action="store_true",
                        help="Look for model in local 'models/' directory instead of auto-downloading")
    args = parser.parse_args()

    app = QApplication(sys.argv)
    window = MainWindow(args)
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()