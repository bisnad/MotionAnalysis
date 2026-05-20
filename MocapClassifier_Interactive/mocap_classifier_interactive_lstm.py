"""
Motion Capture Classification using LSTM Networks - Inference Script
"""

"""
imports
"""

# General imports

import os
import sys
import time
import numpy as np
from collections import deque

# Pytorch imports

import torch
import torch.nn as nn

# OSC imports

from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import BlockingOSCUDPServer
from pythonosc.udp_client import SimpleUDPClient

# GUI Imports

from PyQt5 import QtWidgets, QtCore, QtGui
from vispy import scene
from vispy.app import use_app
from vispy.scene import SceneCanvas

"""
Configurations
"""

# Device Settings

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Mocap settings

mocap_joint_indices = [3, 4, 5, 6, 7] # right arm only
num_features = 21
mocap_data_window_length = 90
mocap_pos_scale = 1.0

# Classification Settings

classes = ["Fluidity", "Levitation", "Particles", "Staccato", "Thrusting"]
class_count = len(classes)

# Training Settigs

save_stats_path = "data/results/stats"
model_weights_file = "data/results/weights/classifier_weights_epoch_200.pth"

# OSC Settings
osc_receive_ip = "0.0.0.0"
osc_receive_port = 9007
osc_send_ip = "127.0.0.1"
osc_send_port = 9008

"""
Load Normalization Stats
"""

# Load normalization stats
mean_np = np.load(os.path.join(save_stats_path, "mean.npy"))
std_np = np.load(os.path.join(save_stats_path, "std.npy"))
std_np[std_np == 0] = 1e-8
data_mean = torch.tensor(mean_np, dtype=torch.float32).to(device)
data_std = torch.tensor(std_np, dtype=torch.float32).to(device)

"""
Classifier
"""

class Classifier(nn.Module):
    def __init__(self, num_features=21, hidden_dim=128, num_classes=5, num_layers=2):
        super().__init__()
        self.joint_embedding = nn.Linear(num_features, hidden_dim)
        self.lstm = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.joint_embedding(x)
        x, _ = torch.max(x, dim=2) 
        out, _ = self.lstm(x)
        return self.classifier(out[:, -1, :])

"""
OSC Worker
"""

# --- Background Worker Thread for OSC and Inference ---
class OSCWorker(QtCore.QThread):
    probs_signal = QtCore.pyqtSignal(np.ndarray)
    fps_signal = QtCore.pyqtSignal(float)

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.buffer_pos = deque(maxlen=mocap_data_window_length + 2)
        self.buffer_rot = deque(maxlen=mocap_data_window_length + 2)
        
        self.current_pos_data = None
        self.frame_count = 0
        self.fps_start_time = time.time()
        self.server = None
        self.is_running = False
        
        # OSC Out Client
        self.osc_client = SimpleUDPClient(osc_send_ip, osc_send_port)

    def set_running(self, state):
        self.is_running = state
        if not state:
            # Clear buffers on stop so old data doesn't corrupt new sessions
            self.buffer_pos.clear()
            self.buffer_rot.clear()
            self.current_pos_data = None

    def osc_pos_handler(self, address, *args):
        if not self.is_running: return
        raw_data = np.array(args)
        pos_data = []
        for j in mocap_joint_indices:
            idx = j * 3
            pos_data.append(raw_data[idx : idx+3] * mocap_pos_scale)
        self.current_pos_data = pos_data

    def osc_rot_handler(self, address, *args):
        if not self.is_running or self.current_pos_data is None: return
            
        raw_data = np.array(args)
        rot_data = []
        for j in mocap_joint_indices:
            idx = j * 4
            rot_data.append(raw_data[idx : idx+4])
            
        self.buffer_pos.append(self.current_pos_data)
        self.buffer_rot.append(rot_data)
        self.current_pos_data = None

        if len(self.buffer_pos) == self.buffer_pos.maxlen:
            pos = np.array(self.buffer_pos)
            rot = np.array(self.buffer_rot)
            
            vel_pos = np.diff(pos, axis=0)
            vel_rot = np.diff(rot, axis=0)
            acc_pos = np.diff(vel_pos, axis=0)
            acc_rot = np.diff(vel_rot, axis=0)
            
            mocap_data = np.concatenate([
                pos[2:], rot[2:], vel_pos[1:], vel_rot[1:], acc_pos, acc_rot
            ], axis=-1)
            
            with torch.no_grad():
                batch_x = torch.FloatTensor(mocap_data).unsqueeze(0).to(device)
                batch_x_norm = (batch_x - data_mean) / data_std
                batch_x_norm = torch.nan_to_num(batch_x_norm)
                
                output = self.model(batch_x_norm)
                probs = torch.softmax(output, dim=1).squeeze().cpu().numpy()
            
            # Broadcast the class probabilities
            self.probs_signal.emit(probs)
            self.osc_client.send_message("/classifier/probs", probs.tolist())
            
            self.frame_count += 1
            current_time = time.time()
            if current_time - self.fps_start_time >= 1.0:
                fps = self.frame_count / (current_time - self.fps_start_time)
                self.fps_signal.emit(fps)
                self.fps_start_time = current_time
                self.frame_count = 0

    def run(self):
        dispatcher = Dispatcher()
        dispatcher.map("/mocap/0/joint/pos_local", self.osc_pos_handler)
        dispatcher.map("/mocap/0/joint/rot_local", self.osc_rot_handler)
        
        self.server = BlockingOSCUDPServer((osc_receive_ip, osc_receive_port), dispatcher)
        print("OSC Receiver listening on port {} ...".format(osc_receive_port))
        self.server.serve_forever()

    def stop(self):
        if self.server:
            self.server.shutdown()
        self.wait()

"""
Main GUI Window
"""

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Real-Time Mocap Classifier")
        self.resize(800, 600)
        self.show_visualization = True

        main_widget = QtWidgets.QWidget()
        self.setCentralWidget(main_widget)
        
        # Main vertical layout
        layout = QtWidgets.QVBoxLayout()
        main_widget.setLayout(layout)

        # 1. VisPy Canvas Setup (White Background)
        use_app('PyQt5')
        self.canvas = SceneCanvas(keys='interactive', show=True, bgcolor='white')
        layout.addWidget(self.canvas.native, stretch=1) # Give the canvas the majority of space
        
        self.view = self.canvas.central_widget.add_view()
        self.view.camera = scene.PanZoomCamera(rect=(-1, -0.2, class_count * 2.5, 1.4))
        
        self.bars = []
        # Adjusted colors for white background
        colors = [(0.1, 0.4, 0.7, 1), (0.8, 0.2, 0.1, 1), (0.1, 0.6, 0.3, 1)]

        for i, cls in enumerate(classes):
            x_pos = i * 2
            # Text label changed to black for white background
            scene.visuals.Text(cls, pos=(x_pos, -0.1), color='black', font_size=14, parent=self.view.scene)
            bar = scene.visuals.Rectangle(center=(x_pos, 0.001), width=1.2, height=0.002, color=colors[i % len(colors)], parent=self.view.scene)
            self.bars.append(bar)

        # 2. Bottom Control Panel
        control_layout = QtWidgets.QHBoxLayout()
        
        self.btn_start = QtWidgets.QPushButton("Start")
        self.btn_stop = QtWidgets.QPushButton("Stop")
        self.btn_toggle_vis = QtWidgets.QPushButton("Disable Visualisation")
        self.btn_exit = QtWidgets.QPushButton("Exit")
        self.fps_label = QtWidgets.QLabel("FPS: 0.0")
        
        self.fps_label.setFont(QtGui.QFont("Arial", 12, QtGui.QFont.Bold))
        self.fps_label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        
        self.btn_start.clicked.connect(self.start_processing)
        self.btn_stop.clicked.connect(self.stop_processing)
        self.btn_toggle_vis.clicked.connect(self.toggle_vis)
        self.btn_exit.clicked.connect(self.close)

        control_layout.addWidget(self.btn_start)
        control_layout.addWidget(self.btn_stop)
        control_layout.addWidget(self.btn_toggle_vis)
        control_layout.addWidget(self.btn_exit)
        control_layout.addStretch()
        control_layout.addWidget(self.fps_label)
        
        layout.addLayout(control_layout)

        # Initialize and start PyTorch Model + OSC Worker
        self.init_model_and_worker()

    def init_model_and_worker(self):
        self.classifier = Classifier(num_features=num_features, hidden_dim=128, num_classes=class_count, num_layers=2).to(device)
        self.classifier.load_state_dict(torch.load(model_weights_file, map_location=device))
        self.classifier.eval()

        self.osc_worker = OSCWorker(self.classifier)
        self.osc_worker.probs_signal.connect(self.update_graph)
        self.osc_worker.fps_signal.connect(self.update_fps)
        self.osc_worker.start()

    def start_processing(self):
        self.osc_worker.set_running(True)
        self.fps_label.setText("FPS: Calculating...")

    def stop_processing(self):
        self.osc_worker.set_running(False)
        self.fps_label.setText("FPS: Stopped")
        # Reset bars to zero visually
        self.update_graph(np.zeros(class_count))

    def toggle_vis(self):
        self.show_visualization = not self.show_visualization
        if self.show_visualization:
            self.btn_toggle_vis.setText("Disable Visualisation")
        else:
            self.btn_toggle_vis.setText("Enable Visualisation")
            
        self.canvas.native.setVisible(self.show_visualization)

    @QtCore.pyqtSlot(np.ndarray)
    def update_graph(self, probs):
        if not self.show_visualization:
            return
            
        for i, prob in enumerate(probs):
            h = max(prob, 0.002) 
            self.bars[i].height = h
            self.bars[i].center = (i * 2, h / 2.0)
            
        self.canvas.update()

    @QtCore.pyqtSlot(float)
    def update_fps(self, fps):
        self.fps_label.setText(f"FPS: {fps:.1f}")

    def closeEvent(self, event):
        self.osc_worker.stop()
        event.accept()

"""
Run Application
"""

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())