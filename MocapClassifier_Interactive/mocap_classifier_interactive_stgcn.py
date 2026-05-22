"""
Motion Capture Classification using ST-GCN - Inference Script
"""

"""
Imports
"""

# General imports
import os
import sys
import time
import numpy as np
from collections import deque
import colorsys

# Pytorch imports
import torch
import torch.nn as nn
import torch.nn.functional as F

# OSC imports
from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import BlockingOSCUDPServer
from pythonosc.udp_client import SimpleUDPClient

# GUI Imports
from PyQt5 import QtWidgets, QtCore, QtGui
from vispy import scene
from vispy.app import use_app
from vispy.scene import SceneCanvas

# Mocap imports
from common import bvh_tools as bvh
from common import fbx_tools as fbx
from common import mocap_tools as mocap

"""
Configurations
"""

# Device Settings
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Mocap settings
mocap_data_file_path = "E:/Data/mocap/Daniel/Zed/fbx/classes/"
mocap_data_file_extensions = [".fbx"] 
mocap_joint_indices = [3, 4, 5, 6, 7]
mocap_data_window_length = 90
mocap_pos_scale = 1.0

# Number of features per joint (pos(3) + rot(4) + vel_pos(3) + vel_rot(4) + acc_pos(3) + acc_rot(4))
num_features = 21

# Architecture parameters (Matching Training script)
stgcn_channels = [64, 128] 
stgcn_temporal_kernel = 9
stgcn_temporal_padding = 4     # (kernel - 1) // 2
stgcn_dropout_rate = 0.2
stgcn_dropedge_rate = 0.1

# Training Settings Paths
save_stats_path = "data/results_stgcn/stats"
model_weights_file = "data/results_stgcn/weights/classifier_weights_epoch_200.pth"

# OSC Settings
osc_receive_ip = "0.0.0.0"
osc_receive_port = 9007
osc_send_ip = "127.0.0.1"
osc_send_port = 9008

"""
Load Normalization Stats
"""

if os.path.exists(save_stats_path):
    mean_np = np.load(os.path.join(save_stats_path, "mean.npy"))
    std_np = np.load(os.path.join(save_stats_path, "std.npy"))
else:
    print(f"Warning: Normalization stats not found at {save_stats_path}. Using zero mean and unit std.")
    mean_np = np.zeros(num_features)
    std_np = np.ones(num_features)

data_mean = torch.tensor(mean_np, dtype=torch.float32).to(device)
data_std = torch.tensor(std_np, dtype=torch.float32).to(device)

"""
Helper Functions to extract Skeleton Topology
"""

def find_classes(directory):
    if not os.path.exists(directory):
        print(f"Warning: Directory {directory} not found. Creating dummy classes for visualization.")
        return ["Class_A", "Class_B"], {"Class_A": 0, "Class_B": 1}
    classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
    class_to_idx = {class_name: i for i, class_name in enumerate(classes)}
    return classes, class_to_idx

def load_class_filenames(directory, extensions):
    _, class_to_idx = find_classes(directory)
    instances = []
    for target_class in sorted(class_to_idx.keys()):
        class_index = class_to_idx[target_class]
        target_dir = os.path.join(directory, target_class)
        if not os.path.isdir(target_dir): continue
        for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
            for fname in sorted(fnames):
                if fname.lower().endswith(tuple(extensions)):
                    instances.append((os.path.join(root, fname), class_index))
    return instances

# Automatically detect classes from folders
classes, _ = find_classes(mocap_data_file_path)
class_count = len(classes)

# Load a single file to extract the skeleton parents array
try:
    fbx_tools = fbx.FBX_Tools()
    mocap_tools = mocap.Mocap_Tools()
    class_files = load_class_filenames(mocap_data_file_path, mocap_data_file_extensions)
    sample_file = class_files[0][0]
    fbx_data = fbx_tools.load(sample_file)
    sample_mocap_data = mocap_tools.fbx_to_mocap(fbx_data)[0]
    skeleton_parents = sample_mocap_data["skeleton"]["parents"]
except Exception as e:
    print(f"Warning: Failed to load skeleton topology from FBX: {e}")
    # Dummy array to allow app startup if files aren't found locally
    skeleton_parents = [-1] + [0] * max(mocap_joint_indices) 

"""
Classifier Model
"""

class Classifier(nn.Module):
    def __init__(self, num_features=21, parents=None, joint_indices=None, num_classes=5, 
                 channels=[32, 64], t_kernel=5, t_pad=2, dropout_rate=0.5, dropedge_rate=0.2):
        super().__init__()
        self.num_joints = len(joint_indices)
        self.parents = parents
        self.joint_indices = joint_indices
        self.dropout_rate = dropout_rate
        self.dropedge_rate = dropedge_rate

        # Build graph structure dynamically from mocap skeleton
        A = self._build_adjacency()
        self.register_buffer('A', A)
        
        # Spatial Temporal Blocks (Dynamically sized)
        self.gcn1 = nn.Conv2d(num_features, channels[0], kernel_size=1)
        self.tcn1 = nn.Conv2d(channels[0], channels[0], kernel_size=(t_kernel, 1), padding=(t_pad, 0))
        
        self.gcn2 = nn.Conv2d(channels[0], channels[1], kernel_size=1)
        self.tcn2 = nn.Conv2d(channels[1], channels[1], kernel_size=(t_kernel, 1), padding=(t_pad, 0))
        
        self.classifier = nn.Linear(channels[-1], num_classes)

    def _build_adjacency(self):
        A = np.zeros((self.num_joints, self.num_joints))
        global_to_local = {global_idx: local_idx for local_idx, global_idx in enumerate(self.joint_indices)}
        
        for local_idx, global_idx in enumerate(self.joint_indices):
            parent_global_idx = self.parents[global_idx]
            if parent_global_idx in global_to_local:
                parent_local_idx = global_to_local[parent_global_idx]
                A[local_idx, parent_local_idx] = 1
                A[parent_local_idx, local_idx] = 1
                
        A += np.eye(self.num_joints)
        D = np.diag(np.sum(A, axis=1))
        D_inv_sqrt = np.linalg.inv(np.sqrt(D))
        A_norm = D_inv_sqrt @ A @ D_inv_sqrt
        
        return torch.tensor(A_norm, dtype=torch.float32)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)

        # Apply DropEdge during training
        if self.training:
            mask = (torch.rand_like(self.A) > self.dropedge_rate).float()
            current_A = self.A * mask
            # Ensure self-loops are kept
            current_A = current_A + torch.eye(self.num_joints, device=self.A.device) * 1e-4
        else:
            current_A = self.A
        
        x = torch.einsum('nctv,vw->nctw', x, current_A)
        x = F.relu(self.gcn1(x))
        x = F.relu(self.tcn1(x))
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        
        x = torch.einsum('nctv,vw->nctw', x, current_A)
        x = F.relu(self.gcn2(x))
        x = F.relu(self.tcn2(x))
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        
        x = F.adaptive_avg_pool2d(x, (1, 1)).view(x.size(0), -1)
        return self.classifier(x)

"""
OSC Worker
"""

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
            
            # Shape matches STGCN: (Time, Joints, Features)
            mocap_data = np.concatenate([
                pos[2:], rot[2:], vel_pos[1:], vel_rot[1:], acc_pos, acc_rot
            ], axis=-1)
            
            with torch.no_grad():
                batch_x = torch.FloatTensor(mocap_data).unsqueeze(0).to(device)
                
                # Apply normalization (data_mean/data_std broadcast smoothly over the last dimension)
                batch_x_norm = (batch_x - data_mean) / (data_std + 1e-8)
                batch_x_norm = torch.nan_to_num(batch_x_norm)
                
                output = self.model(batch_x_norm)
                probs = torch.softmax(output, dim=1).squeeze().cpu().numpy()
            
            # Broadcast the class probabilities
            self.probs_signal.emit(probs)
            self.osc_client.send_message("/motion/class", probs.tolist())
            
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
        self.setWindowTitle("Real-Time ST-GCN Mocap Classifier")
        self.resize(800, 600)
        self.show_visualization = True

        main_widget = QtWidgets.QWidget()
        self.setCentralWidget(main_widget)
        
        layout = QtWidgets.QVBoxLayout()
        main_widget.setLayout(layout)

        # VisPy Canvas
        use_app('PyQt5')
        self.canvas = SceneCanvas(keys='interactive', show=True, bgcolor='white')
        layout.addWidget(self.canvas.native, stretch=1)
        
        self.view = self.canvas.central_widget.add_view()
        self.view.camera = scene.PanZoomCamera(rect=(-1, -0.2, class_count * 2, 1.4))
        
        self.bars = []
        
        bar_colors = MainWindow.generate_distinct_colors(class_count)

        for i, cls in enumerate(classes):
            x_pos = i * 2
            scene.visuals.Text(
                cls,
                pos=(x_pos, -0.1),
                color='black',
                font_size=14,
                parent=self.view.scene
            )
            bar = scene.visuals.Rectangle(
                center=(x_pos, 0.001),
                width=1.2,
                height=0.002,
                color=bar_colors[i],
                parent=self.view.scene
            )
            self.bars.append(bar)


        # Bottom Control Panel
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

        self.init_model_and_worker()

    @staticmethod
    def generate_distinct_colors(n, saturation=0.8, value=0.9, alpha=1.0):
        if n <= 0:
            return []
        
        colors = []
        for i in range(n):
            hue = i / n
            r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
            colors.append((r, g, b, alpha))
        return colors

    def init_model_and_worker(self):
        # Pass the extracted skeleton parents and matching architectural parameters to the classifier
        self.classifier = Classifier(
            num_features=num_features, 
            parents=skeleton_parents, 
            joint_indices=mocap_joint_indices, 
            num_classes=class_count,
            channels=stgcn_channels,
            t_kernel=stgcn_temporal_kernel,
            t_pad=stgcn_temporal_padding,
            dropout_rate=stgcn_dropout_rate,
            dropedge_rate=stgcn_dropedge_rate
        ).to(device)
        
        if os.path.exists(model_weights_file):
            self.classifier.load_state_dict(torch.load(model_weights_file, map_location=device))
        else:
            print(f"Warning: Model weights not found at {model_weights_file}")
            
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
        self.update_graph(np.zeros(class_count))

    def toggle_vis(self):
        self.show_visualization = not self.show_visualization
        
        if self.show_visualization:
            self.btn_toggle_vis.setText("Disable Visualisation")
            # Show canvas and restore standard size
            self.canvas.native.show()
            self.adjustSize()
        else:
            self.btn_toggle_vis.setText("Enable Visualisation")
            # Hide canvas and collapse the vertical space
            self.canvas.native.hide()
            self.fps_label.setText("FPS: --")
            self.centralWidget().adjustSize()
            self.resize(self.width(), 1)

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