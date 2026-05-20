"""
Real-time Motion Capture Clustering
"""

"""
imports
"""

import os, sys, time, subprocess
import numpy as np
import math
import json

from common import utils
from common import bvh_tools as bvh
from common import fbx_tools as fbx
from common import mocap_tools as mocap

from sklearn.cluster import KMeans

import motion_analysis as ma

#gui
from PyQt5 import QtWidgets, QtCore, QtGui
from vispy import scene
from vispy.app import use_app
from vispy.scene import SceneCanvas, visuals

"""
Mocap Settings
"""

mocap_file_path = "data/mocap"
mocap_files = ["Muriel_Take1.fbx"]
mocap_pos_scale = 1.0
mocap_fps = 50
mocap_joint_weight_file = "configs/joint_weights_xsens_fbx.json"
mocap_body_weight = 60

"""
Cluster Settings
"""

cluster_count = 20
cluster_random_state = 170
cluster_excerpt_length = 48 # number of mocap frames per mocap excerpt used for clustering
cluster_excerpt_count = 1000 # number of mocap excerpts used for clustering
motion_feature_names = ["bsphere", "space_effort"]
motion_feature_average = True
cluster_distance_beta = 1.0 # for computing cluster propbabilities

"""
OSC Settings
"""

osc_receive_ip = "0.0.0.0"
osc_receive_port = 9007

osc_send_ip = "127.0.0.1"
osc_send_port = 9008


"""
Load Mocap Data
"""

bvh_tools = bvh.BVH_Tools()
fbx_tools = fbx.FBX_Tools()
mocap_tools = mocap.Mocap_Tools()

all_mocap_data = []

for mocap_file in mocap_files:
    
    print("process file ", mocap_file)
    
    if mocap_file.endswith(".bvh") or mocap_file.endswith(".BVH"):
        bvh_data = bvh_tools.load(mocap_file_path + "/" + mocap_file)
        mocap_data = mocap_tools.bvh_to_mocap(bvh_data)
    elif mocap_file.endswith(".fbx") or mocap_file.endswith(".FBX"):
        fbx_data = fbx_tools.load(mocap_file_path + "/" + mocap_file)
        mocap_data = mocap_tools.fbx_to_mocap(fbx_data)[0] # first skeleton only
    
    mocap_data["skeleton"]["offsets"] *= mocap_pos_scale
    mocap_data["motion"]["pos_local"] *= mocap_pos_scale
    
    # set x and z offset of root joint to zero
    mocap_data["skeleton"]["offsets"][0, 0] = 0.0 
    mocap_data["skeleton"]["offsets"][0, 2] = 0.0 
    
    if mocap_file.endswith(".bvh") or mocap_file.endswith(".BVH"):
        mocap_data["motion"]["rot_local"] = mocap_tools.euler_to_quat_bvh(mocap_data["motion"]["rot_local_euler"], mocap_data["rot_sequence"])
    elif mocap_file.endswith(".fbx") or mocap_file.endswith(".FBX"):
        mocap_data["motion"]["rot_local"] = mocap_tools.euler_to_quat(mocap_data["motion"]["rot_local_euler"], mocap_data["rot_sequence"])
        
    mocap_data["motion"]["pos_world"], mocap_data["motion"]["rot_world"] = mocap_tools.local_to_world(mocap_data["motion"]["rot_local"], mocap_data["motion"]["pos_local"], mocap_data["skeleton"])

    all_mocap_data.append(mocap_data)

# retrieve mocap properties

mocap_data = all_mocap_data[0]
joint_count = mocap_data["motion"]["rot_local"].shape[1]
joint_dim = mocap_data["motion"]["rot_local"].shape[2]
pose_dim = joint_count * joint_dim

offsets = mocap_data["skeleton"]["offsets"].astype(np.float32)
parents = mocap_data["skeleton"]["parents"]
children = mocap_data["skeleton"]["children"]

# retrieve joint weight percentages

with open(mocap_joint_weight_file) as fh:
    mocap_joint_weight_percentages = json.load(fh)
mocap_joint_weight_percentages = mocap_joint_weight_percentages["jointWeights"]

# calc joint weights

mocap_joint_weight_percentages = np.array(mocap_joint_weight_percentages)
mocap_joint_weight_percentages_total = np.sum(mocap_joint_weight_percentages)
joint_weights = mocap_joint_weight_percentages * mocap_body_weight / 100.0

"""
Mocap Analysis
"""

mocap_data["motion"]["pos_world_m"] = mocap_data["motion"]["pos_world"] / 100.0

mocap_data["motion"]["pos_world_smooth"] = ma.smooth(mocap_data["motion"]["pos_world_m"], 25)
mocap_data["motion"]["pos_scalar"] = ma.scalar(mocap_data["motion"]["pos_world_smooth"], "norm")
mocap_data["motion"]["vel_world"] = ma.derivative(mocap_data["motion"]["pos_world_smooth"], 1.0 / 50.0)
mocap_data["motion"]["vel_world_smooth"] = ma.smooth(mocap_data["motion"]["vel_world"], 25)
mocap_data["motion"]["vel_world_scalar"] = ma.scalar(mocap_data["motion"]["vel_world_smooth"], "norm")
mocap_data["motion"]["accel_world"] = ma.derivative(mocap_data["motion"]["vel_world_smooth"], 1.0 / 50.0)
mocap_data["motion"]["accel_world_smooth"] = ma.smooth(mocap_data["motion"]["accel_world"], 25)
mocap_data["motion"]["accel_world_scalar"] = ma.scalar(mocap_data["motion"]["accel_world_smooth"], "norm")
mocap_data["motion"]["jerk_world"] = ma.derivative(mocap_data["motion"]["accel_world_smooth"], 1.0 / 50.0)
mocap_data["motion"]["jerk_world_smooth"] = ma.smooth(mocap_data["motion"]["jerk_world"], 25)
mocap_data["motion"]["jerk_world_scalar"] = ma.scalar(mocap_data["motion"]["jerk_world_smooth"], "norm")
mocap_data["motion"]["qom"] = ma.quantity_of_motion(mocap_data["motion"]["vel_world_scalar"], joint_weights)
mocap_data["motion"]["bbox"] = ma.bounding_box(mocap_data["motion"]["pos_world_m"])
mocap_data["motion"]["bsphere"] = ma.bounding_sphere(mocap_data["motion"]["pos_world_m"])
mocap_data["motion"]["weight_effort"] = ma.weight_effort(mocap_data["motion"]["vel_world_scalar"], joint_weights, 25)
mocap_data["motion"]["space_effort"] = ma.space_effort_v2(mocap_data["motion"]["pos_world_m"], joint_weights, 25)
mocap_data["motion"]["time_effort"] = ma.time_effort(mocap_data["motion"]["accel_world_scalar"], joint_weights, 25)
mocap_data["motion"]["flow_effort"] = ma.flow_effort(mocap_data["motion"]["jerk_world_scalar"], joint_weights, 25)


"""
Process Requested Mocap Features
"""

# gather mocap features

mocap_features = []

for motion_feature_name in motion_feature_names:
    
    mocap_feature = mocap_data["motion"][motion_feature_name]
    mocap_feature = np.reshape(mocap_feature, (mocap_feature.shape[0], -1))
    
    mocap_features.append(mocap_feature)
    
mocap_features = np.concatenate(mocap_features, axis=1)

# normalise mocap features

mocap_features_mean = np.mean(mocap_features, axis=0, keepdims=True)
mocap_features_std = np.std(mocap_features, axis=0, keepdims=True)
mocap_features_norm = (mocap_features - mocap_features_mean) / (mocap_features_std + 1e-8)

# create feature excerpts

feature_excerpts = []

mocap_frame_count = mocap_features_norm.shape[0] 
cluster_excerpt_count = min(cluster_excerpt_count, mocap_frame_count - cluster_excerpt_length)
cluster_excerpt_offset = (mocap_frame_count - cluster_excerpt_length) / cluster_excerpt_count
cluster_excerpt_index = 0.0

while True:
    
    if int(cluster_excerpt_index) > mocap_frame_count - cluster_excerpt_length:
        break
    
    excert_start_frame = int(cluster_excerpt_index)
    excerpt_end_frame = excert_start_frame + cluster_excerpt_length
    
    #print("excert_start_frame ", excert_start_frame, " excerpt_end_frame ", excerpt_end_frame)
    
    excerpt = mocap_features_norm[excert_start_frame:excerpt_end_frame, ...]
    
    #print("excerpt s ", excerpt.shape)
    
    if motion_feature_average:
        
        excerpt = np.mean(excerpt, axis=0, keepdims=True)

    #print("excerpt 2 s ", excerpt.shape)
    
    excerpt = excerpt.flatten()
    
    #print("excerpt 3 s ", excerpt.shape)
    
    feature_excerpts.append(excerpt)
    
    cluster_excerpt_index += cluster_excerpt_offset

feature_excerpts = np.stack(feature_excerpts)

print("feature_excerpts s ", feature_excerpts.shape)

"""
Cluster Mocap Features
"""


km = KMeans(n_clusters=cluster_count, n_init="auto", random_state=cluster_random_state)
cluster_labels = km.fit_predict(feature_excerpts)

# =============================================================================
# GUI Components
# =============================================================================

class ClusteringBarView:
    """Visualization component for clustering results with labels."""

    def __init__(self, cluster_count: int, cluster_labels: list, 
                 colors: list, parent_view):
        self.cluster_count = cluster_count
        self.cluster_labels = cluster_labels
        self.parent_view = parent_view

        # Calculate bar positions and dimensions
        self.bar_width = 0.8 / cluster_count
        self.bar_spacing = 1.0 / cluster_count
        bar_centers_x = np.linspace(self.bar_spacing / 2, 1.0 - self.bar_spacing / 2, cluster_count)

        # Create bars
        self.bars = []
        self.labels = []

        for i in range(cluster_count):
            # Create bar rectangle
            bar = visuals.Rectangle(
                center=(bar_centers_x[i], 0.0),
                width=self.bar_width,
                height=0.01,
                color=colors[i]
            )
            self.bars.append(bar)

            # Create text label
            label = visuals.Text(
                text=str(cluster_labels[i]),
                pos=(bar_centers_x[i], -0.1),
                color='black',
                font_size=12,
                anchor_x='center',
                anchor_y='top'
            )
            self.labels.append(label)

        # Create compound visual
        self.compound = visuals.Compound(self.bars + self.labels, parent=self.parent_view)

    def update(self, values: np.ndarray):
        for bar, value in zip(self.bars, values):
            bar.center = (bar.center[0], value / 2)
            bar.height = max(abs(value), 0.001)  # Prevent zero height

class VisualizationCanvas:
    """Main visualization canvas containing the bar chart."""
    
    def __init__(self, cluster_labels: list, colors: list, size: tuple):
        self.size = size
        self.canvas = SceneCanvas(size=size, keys="interactive")
        self.grid = self.canvas.central_widget.add_grid()
        
        # Create main view for bars
        self.bar_view = self.grid.add_view(0, 0, bgcolor="white")
        
        # Initialize bar visualization
        self.bars = ClusteringBarView(
            cluster_count=len(colors),
            cluster_labels=cluster_labels,
            colors=colors,
            parent_view=self.bar_view.scene
        )
        
        # Configure camera
        self.bar_view.camera = "panzoom"
        self.bar_view.camera.set_range(x=(0.0, 1.0), y=(-0.2, 1.0))

        # --- NEW THROTLLING CODE ---
        self.latest_data = None
        self.gui_timer = QtCore.QTimer()
        self.gui_timer.timeout.connect(self._render_frame)
        self.gui_timer.start(33)  # Updates every 33ms (approx 30 FPS)
        
    def set_visualization_enabled(self, enabled: bool):
        """Turn the GUI updates on or off."""
        if enabled:
            self.gui_timer.start(33)
        else:
            self.gui_timer.stop()
            # Flatten the bars to indicate visualization is suspended
            empty_data = np.zeros(self.bars.cluster_count)
            self.bars.update(empty_data)
            self.canvas.update()  # Force one final redraw to show flat bars
        
    def update(self, new_data: np.ndarray):
        """Called at 50 FPS by the clusterer. Only saves the data, does NOT draw."""
        self.latest_data = new_data

    def _render_frame(self):
        """Called at 30 FPS by the timer. Does the actual heavy GUI drawing."""
        if self.latest_data is not None:
            self.bars.update(self.latest_data)
            self.latest_data = None  # Clear until new data arrives

class MainWindow(QtWidgets.QMainWindow):
    """Main application window."""

    closing = QtCore.pyqtSignal()
    start_clustering = QtCore.pyqtSignal()
    stop_clustering = QtCore.pyqtSignal()

    def __init__(self, canvas: VisualizationCanvas, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setWindowTitle("Real-time Motion Clustering")
        self.setWindowIcon(QtGui.QIcon())

        # Create central widget and main layout
        central_widget = QtWidgets.QWidget()
        main_layout = QtWidgets.QVBoxLayout()

        # Add canvas to layout
        self.canvas = canvas
        main_layout.addWidget(self.canvas.canvas.native)

        # Create control buttons
        self._create_controls(main_layout)

        # Set up central widget
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        # Connect signals
        self._connect_signals()

    def _create_controls(self, main_layout: QtWidgets.QVBoxLayout):
        controls_layout = QtWidgets.QHBoxLayout()

        # Create buttons
        self.start_button = QtWidgets.QPushButton("Start Clustering", self)
        self.stop_button = QtWidgets.QPushButton("Stop Clustering", self)
        self.exit_button = QtWidgets.QPushButton("Exit", self)
        self.vis_toggle = QtWidgets.QCheckBox("Enable Visualization", self)
        self.vis_toggle.setChecked(True)  # On by default
        
        self.fps_label = QtWidgets.QLabel("0 FPS", self)
        font = self.fps_label.font()
        font.setBold(True)
        self.fps_label.setFont(font)
        self.fps_label.setMinimumWidth(50)
        self.fps_label.setAlignment(QtCore.Qt.AlignCenter)

        # Set button properties
        for button in [self.start_button, self.stop_button, self.exit_button]:
            button.setMinimumWidth(120)
            button.setMinimumHeight(30)

        # Add buttons to layout
        controls_layout.addWidget(self.start_button)
        controls_layout.addWidget(self.stop_button)
        controls_layout.addWidget(self.vis_toggle)
        controls_layout.addWidget(self.fps_label)
        controls_layout.addWidget(self.exit_button)
        controls_layout.addStretch()

        main_layout.addLayout(controls_layout)
        
        self.processed_frames = 0
        self.fps_timer = QtCore.QTimer(self)
        self.fps_timer.timeout.connect(self._update_fps_display)
        self.fps_timer.start(1000)  # Update label once per second

    def _connect_signals(self):
        self.start_button.clicked.connect(self.start_clustering.emit)
        self.stop_button.clicked.connect(self.stop_clustering.emit)
        self.vis_toggle.toggled.connect(self.canvas.set_visualization_enabled)
        self.exit_button.clicked.connect(self.close)
        
    def _update_fps_display(self):
        """Update the label and reset counter every second."""
        self.fps_label.setText(f"{self.processed_frames} FPS")
        self.processed_frames = 0
        
    def increment_fps_counter(self, *args):
        """Tally a frame whenever new clustering data is emitted."""
        self.processed_frames += 1

    def closeEvent(self, event):
        print("Closing main window")
        self.closing.emit()
        event.accept()

# =============================================================================
# Live Clustering Component
# =============================================================================

class LiveClusterer(QtCore.QObject):
    """Handles real-time motion clustering using the trained KMeans model."""

    new_data = QtCore.pyqtSignal(np.ndarray)

    def __init__(self, km, features_mean, features_std, beta, parent=None):
        super().__init__(parent=parent)
        self.km = km
        self.features_mean = features_mean
        self.features_std = features_std
        self.beta = beta

    def update(self, input_data: np.ndarray):
        try:
            # Normalize input data
            input_norm = (input_data - self.features_mean) / (self.features_std + 1e-8)

            # Predict distances to clusters
            distances = self.km.transform(input_norm)

            scores = -self.beta * distances**2        # or just -beta * distances

            # subtract row-wise max for numerical stability
            scores = scores - scores.max(axis=1, keepdims=True)

            probs = np.exp(scores)
            probs /= probs.sum(axis=1, keepdims=True)  # shape (n_samples, K)

            class_probs = probs[0]
            self.new_data.emit(class_probs)

        except Exception as e:
            print(f"Error in clustering: {e}")

# =============================================================================
# OSC Communication
# =============================================================================
import re
from pythonosc import dispatcher, osc_server
from pythonosc.udp_client import SimpleUDPClient

class OSCReceiver(QtCore.QObject):
    new_data = QtCore.pyqtSignal(dict)

    def __init__(self, ip: str, port: int, parent=None):
        super().__init__(parent=parent)
        self.ip = ip
        self.port = port
        self.server = None
        self._setup_server()

    def _setup_server(self):
        self.dispatcher = dispatcher.Dispatcher()
        self.dispatcher.map("/*", self._handle_message)
        self.server = osc_server.BlockingOSCUDPServer(
            (self.ip, self.port), self.dispatcher
        )

    def start(self):
        print(f"Starting OSC receiver on {self.ip}:{self.port}")
        try:
            self.server.serve_forever()
        except Exception as e:
            print(f"OSC server error: {e}")

    def stop(self):
        print("Stopping OSC receiver")
        if self.server:
            self.server.shutdown()

    def _handle_message(self, addr: str, *args):
        try:
            values_dict = {
                addr: np.array(args, dtype=np.float32)
            }
            self.new_data.emit(values_dict)
        except Exception as e:
            print(f"Error handling OSC message: {e}")

class MotionDataReceiver(QtCore.QObject):
    new_data = QtCore.pyqtSignal(np.ndarray)

    def __init__(self, data_ids: list, window_length: int, joint_weights: np.ndarray, parent=None):
        super().__init__(parent=parent)
        self.data_ids = data_ids
        self.window_length = window_length
        self.joint_weights = joint_weights

        self.data_dims = [None] * len(self.data_ids)
        self.data_dim_total = None
        self.data_updated = [False] * len(self.data_ids)
        self.data_window = None
        self.is_running = False
        
        self.regexes = [re.compile('^' + re.escape(p).replace('\\*', '.*') + '$') for p in self.data_ids]
        
    def start(self):
        self.is_running = True
        print("Motion data receiver started")

    def stop(self):
        self.is_running = False
        print("Motion data receiver stopped")

    def receive(self, new_data: dict):
        if not self.is_running:
            return

        try:
            data_id = list(new_data.keys())[0]
            
            matched_index = None
            for idx, regex in enumerate(self.regexes):
                if regex.match(data_id):
                    matched_index = idx
                    break

            if matched_index is None:
                return

            data_values = list(new_data.values())[0]
            data_index = matched_index

            if self.data_dim_total is None:
                self._initialize_data_structures(data_values, data_index)
                if self.data_dim_total is None:
                    return

            self._update_sliding_window(data_values, data_index)

        except Exception as e:
            print(f"Error processing motion data: {e}")

    def _initialize_data_structures(self, data_values: np.ndarray, data_index: int):
        self.data_dims[data_index] = data_values.shape[0]

        if None not in self.data_dims:
            self.data_dim_total = sum(self.data_dims)
            self.data_window = np.zeros((self.window_length, self.data_dim_total), dtype=np.float32)
            print(f"Initialized data window: {self.data_window.shape}")

    def _update_sliding_window(self, data_values: np.ndarray, data_index: int):
        start_pos = sum(self.data_dims[:data_index])
        end_pos = start_pos + self.data_dims[data_index]

        self.data_window[-1, start_pos:end_pos] = data_values
        self.data_updated[data_index] = True

        if False not in self.data_updated:
            try:
                pos_world_idx = self.data_ids.index("/mocap/0/joint/pos_world")
                pw_start = sum(self.data_dims[:pos_world_idx])
                pw_end = pw_start + self.data_dims[pos_world_idx]

                pos_world_raw = self.data_window[:, pw_start:pw_end]
                joint_count_rt = pos_world_raw.shape[1] // 3
                pos_world_3d = pos_world_raw.reshape(self.window_length, joint_count_rt, 3)

                # Dictionary to cache features so they are only calculated once per frame
                computed_features = {}
                
                # Recursive lazy-evaluator
                def get_feature(name):
                    # If already calculated for this frame, return the cached result
                    if name in computed_features:
                        return computed_features[name]
                    
                    # Otherwise, calculate it and its dependencies on the fly
                    if name == "pos_world": res = pos_world_3d
                    elif name == "pos_world_m": res = get_feature("pos_world") / 100.0
                    elif name == "pos_world_smooth": res = ma.smooth(get_feature("pos_world_m"), 25)
                    elif name == "pos_scalar": res = ma.scalar(get_feature("pos_world_smooth"), "norm")
                    elif name == "vel_world": res = ma.derivative(get_feature("pos_world_smooth"), 1.0 / 50.0)
                    elif name == "vel_world_smooth": res = ma.smooth(get_feature("vel_world"), 25)
                    elif name == "vel_world_scalar": res = ma.scalar(get_feature("vel_world_smooth"), "norm")
                    elif name == "accel_world": res = ma.derivative(get_feature("vel_world_smooth"), 1.0 / 50.0)
                    elif name == "accel_world_smooth": res = ma.smooth(get_feature("accel_world"), 25)
                    elif name == "accel_world_scalar": res = ma.scalar(get_feature("accel_world_smooth"), "norm")
                    elif name == "jerk_world": res = ma.derivative(get_feature("accel_world_smooth"), 1.0 / 50.0)
                    elif name == "jerk_world_smooth": res = ma.smooth(get_feature("jerk_world"), 25)
                    elif name == "jerk_world_scalar": res = ma.scalar(get_feature("jerk_world_smooth"), "norm")
                    elif name == "qom": res = ma.quantity_of_motion(get_feature("vel_world_scalar"), self.joint_weights)
                    elif name == "bbox": res = ma.bounding_box(get_feature("pos_world_m"))
                    elif name == "bsphere": res = ma.bounding_sphere(get_feature("pos_world_m"))
                    elif name == "weight_effort": res = ma.weight_effort(get_feature("vel_world_scalar"), self.joint_weights, 25)
                    elif name == "space_effort": res = ma.space_effort_v2(get_feature("pos_world_m"), self.joint_weights, 25)
                    elif name == "time_effort": res = ma.time_effort(get_feature("accel_world_scalar"), self.joint_weights, 25)
                    elif name == "flow_effort": res = ma.flow_effort(get_feature("jerk_world_scalar"), self.joint_weights, 25)
                    else: raise ValueError(f"Unknown feature requested: {name}")
                    
                    # Store in cache and return
                    computed_features[name] = res
                    return res

                # Build the final feature vector using the requested names
                feature_blocks = []
                for f_name in motion_feature_names:
                    feat = get_feature(f_name)
                    feat = np.reshape(feat, (feat.shape[0], -1))
                    feature_blocks.append(feat)

                features = np.concatenate(feature_blocks, axis=1)

                if motion_feature_average:
                    features = np.mean(features, axis=0, keepdims=True)

                features = features.flatten().reshape(1, -1)
                features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
                
                self.new_data.emit(features)

            except Exception as e:
                import traceback
                print(f"Error extracting realtime features: {e}")
                traceback.print_exc()

            self.data_window = np.roll(self.data_window, shift=-1, axis=0)
            self.data_window[-1, :] = 0.0
            self.data_updated = [False] * len(self.data_updated)

class ClusteringSender:
    def __init__(self, ip: str, port: int):
        self.osc_sender = SimpleUDPClient(ip, port)
        print(f"Clustering sender initialized: {ip}:{port}")

    def send(self, class_probs: np.ndarray):
        try:
            osc_values = class_probs.tolist()
            self.osc_sender.send_message("/motion/cluster", osc_values)
        except Exception as e:
            print(f"Error sending clustering data: {e}")

# =============================================================================
# Main Application
# =============================================================================
import colorsys

class MotionClusteringApp(QtCore.QObject):
    def __init__(self):
        super().__init__()
        self.osc_thread = None
        self.components = {}

    def initialize(self):
        try:
            # Initialize live clusterer
            self.components['clusterer'] = LiveClusterer(
                km=km,
                features_mean=mocap_features_mean.flatten(), # Use calculated mean
                features_std=mocap_features_std.flatten(),
                beta=cluster_distance_beta
            )

            # Initialize OSC components
            # For this example, assuming data_ids match what is requested
            mocap_data_ids = [
                "/mocap/0/joint/rot_local",
                "/mocap/0/joint/rot_world",
                "/mocap/0/joint/pos_world",
            ]
            
            self.components['osc_receiver'] = OSCReceiver(
                osc_receive_ip, osc_receive_port
            )
            
            self.components['mocap_receiver'] = MotionDataReceiver(
                data_ids=mocap_data_ids,
                window_length=cluster_excerpt_length,
                joint_weights=joint_weights
            )
            self.components['cluster_sender'] = ClusteringSender(
                osc_send_ip, osc_send_port
            )

            # Initialize GUI
            cluster_labels_list = [f"C{i}" for i in range(cluster_count)]
            bar_colors = [
                colorsys.hsv_to_rgb(1.0 / cluster_count * i, 1.0, 1.0)
                for i in range(cluster_count)
            ]
            self.components['canvas'] = VisualizationCanvas(
                cluster_labels_list, bar_colors, (800, 400)
            )
            self.components['main_window'] = MainWindow(self.components['canvas'])

            # Connect signals
            self._connect_components()
            print("Application initialized successfully")
            return True

        except Exception as e:
            print(f"Failed to initialize application: {e}")
            return False

    def _connect_components(self):
        self.osc_thread = QtCore.QThread(parent=self.components['main_window'])
        self.components['osc_receiver'].moveToThread(self.osc_thread)

        self.components['osc_receiver'].new_data.connect(
            self.components['mocap_receiver'].receive
        )
        self.components['mocap_receiver'].new_data.connect(
            self.components['clusterer'].update
        )
        self.components['clusterer'].new_data.connect(
            self.components['cluster_sender'].send
        )
        self.components['clusterer'].new_data.connect(
            self.components['canvas'].update
        )
        self.components['clusterer'].new_data.connect(
            self.components['main_window'].increment_fps_counter
        )

        self.components['main_window'].start_clustering.connect(
            self.components['mocap_receiver'].start
        )
        self.components['main_window'].stop_clustering.connect(
            self.components['mocap_receiver'].stop
        )

        self.osc_thread.started.connect(self.components['osc_receiver'].start)
        self.components['main_window'].closing.connect(self._shutdown)

    def run(self):
        if not self.initialize():
            return False
        self.components['main_window'].show()
        self.osc_thread.start()
        print("Application started")
        return True

    def _shutdown(self):
        print("Shutting down application...")
        if 'mocap_receiver' in self.components:
            self.components['mocap_receiver'].stop()
        if 'osc_receiver' in self.components:
            self.components['osc_receiver'].stop()
        if self.osc_thread and self.osc_thread.isRunning():
            self.osc_thread.quit()
            self.osc_thread.wait(3000)
        print("Application shutdown complete")

def main():
    qt_app = use_app("pyqt5")
    qt_app.create()

    app = MotionClusteringApp()

    if app.run():
        try:
            qt_app.run()
        except KeyboardInterrupt:
            print("Application interrupted by user")
        except Exception as e:
            print(f"Application error: {e}")
    else:
        print("Failed to start application")
        sys.exit(1)

if __name__ == "__main__":
    main()
