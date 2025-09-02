"""
Real-time Motion Capture Classifier
===================================

This application receives motion capture data via OSC (Open Sound Control), 
classifies motion patterns using a trained LSTM neural network, and displays 
the classification results in real-time through a bar chart visualization.

Features:
- Real-time motion data reception via OSC
- LSTM-based motion classification
- Interactive visualization with class labels
- Graceful application shutdown
- Multi-threaded OSC communication
"""

"""
imports
"""

# general
import os
import sys
import time
import pickle
import logging
from typing import List, Dict, Optional, Tuple
import numpy as np
import colorsys
import re

#mocap
from common import utils
from common import bvh_tools as bvh
from common import fbx_tools as fbx
from common import mocap_tools as mocap
from common.quaternion import qmul, qrot, qnormalize_np, slerp, qfix

#osc
from pythonosc import dispatcher, osc_server
from pythonosc.udp_client import SimpleUDPClient

# pytorch
import torch
from torch import nn
import torch.nn.functional as F

#gui
from PyQt5 import QtWidgets, QtCore, QtGui
from vispy import scene
from vispy.app import use_app
from vispy.scene import SceneCanvas, visuals

"""
Confgurations
"""


# Device Settings

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

# Mocap Settings

mocap_data_file_path = "E:/Data/mocap/Daniel/Zed/fbx/"
mocap_data_file_extensions = [".fbx"] 
mocap_joint_count = 34
mocap_joint_indices = [ 3, 4, 5, 6, 7 ] # right arm only
mocap_data_ids = ["/mocap/*/joint/rot_local"]
mocap_data_window_length = 90
mocap_data_window_offset = 15
mocap_pos_scale = 1.0
mocap_data_norm_path = "../MocapClassifier/results/data/"

# Model Settings

class_count = 5

model_input_dim = None
model_hidden_dim = 128
model_layer_count = 3
model_dropout = 0.3

model_weights_file = "../MocapClassifier/results/weights/classifier_weights_epoch_200.pth"

# OSC Settings

osc_receive_ip = "0.0.0.0"
osc_receive_port = 9007
osc_send_ip = "127.0.0.1"
osc_send_port = 10000

# GUI Settings

canvas_size = (600, 400)
class_labels = ["Class 1", "Class 2", "Class 3", "Class 4", "Class 5"]

"""
Load Mocap Normalization Parameters
"""

try:
    with open(mocap_data_norm_path + "mean.pkl", 'rb') as f:
        data_mean = pickle.load(f)
    with open(mocap_data_norm_path + "std.pkl", 'rb') as f:
        data_std = pickle.load(f)
    input_dim = data_mean.shape[0]
except FileNotFoundError as e:
    print(f"Could not load normalization data: {e}")
    sys.exit(1)

model_input_dim = data_mean.shape[0]

"""
Create Model
"""

class MotionClassifier(nn.Module):
    """
    LSTM-based neural network for motion classification.
    
    Architecture:
    - LSTM layers for temporal feature extraction
    - Fully connected layers for classification
    - Dropout for regularization
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, layer_count: int, 
                 class_count: int, dropout: float = 0.3):
        """
        Initialize the motion classifier.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            layer_count: Number of LSTM layers
            class_count: Number of output classes
            dropout: Dropout probability
        """
        super().__init__()
        
        self.rnn = nn.LSTM(input_dim, hidden_dim, layer_count, 
                          batch_first=True, dropout=0.0)
        self.dropout1 = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.dropout2 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim // 2, class_count)
        self.relu = nn.ReLU()
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize network weights using appropriate distributions."""
        for module in [self.fc1, self.fc2]:
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                module.bias.data.fill_(0.01)
                
        for param in self.rnn.parameters():
            if len(param.shape) >= 2:
                torch.nn.init.orthogonal_(param.data)
            else:
                torch.nn.init.normal_(param.data)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_dim)
            
        Returns:
            Output tensor of shape (batch_size, class_count)
        """
        x, (h, c) = self.rnn(x)
        x = h[-1]  # Take final hidden state from last layer
        x = self.dropout1(x)
        x = self.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return x

"""
Live Classification Components
"""

class LiveClassifier(QtCore.QObject):
    """Handles real-time motion classification using the trained model."""
    
    new_data = QtCore.pyqtSignal(np.ndarray)
    
    def __init__(self, classifier: MotionClassifier, parent=None):
        """
        Initialize the live classifier.
        
        Args:
            classifier: Trained motion classifier model
            parent: Parent Qt object
        """
        super().__init__(parent=parent)
        self.classifier = classifier.eval()
        
    def update(self, input_data: np.ndarray):
        """
        Classify input motion data and emit results.
        
        Args:
            input_data: Motion data array of shape (window_length, input_dim)
        """
        try:
            with torch.no_grad():
                # Normalize input data
                input_norm = (input_data - data_mean) / (data_std + 1e-8)
                
                # Convert to tensor and add batch dimension
                input_tensor = torch.tensor(input_norm, dtype=torch.float32).unsqueeze(0).to(device)
                
                # Get model predictions
                outputs = self.classifier(input_tensor)
                class_probs = F.softmax(outputs, dim=1)
                class_probs = class_probs.detach().cpu().numpy()[0]
                
                self.new_data.emit(class_probs)
                
        except Exception as e:
            print(f"Error in classification: {e}")

"""
OSC Communication
"""

class OSCReceiver(QtCore.QObject):
    """Handles incoming OSC messages in a separate thread."""
    
    new_data = QtCore.pyqtSignal(dict)
    
    def __init__(self, ip: str, port: int, parent=None):
        """
        Initialize OSC receiver.
        
        Args:
            ip: IP address to listen on
            port: Port number to listen on
            parent: Parent Qt object
        """
        super().__init__(parent=parent)
        self.ip = ip
        self.port = port
        self.server = None
        self._setup_server()
        
    def _setup_server(self):
        """Setup OSC server with message dispatcher."""
        self.dispatcher = dispatcher.Dispatcher()
        self.dispatcher.map("/*", self._handle_message)
        self.server = osc_server.BlockingOSCUDPServer(
            (self.ip, self.port), self.dispatcher
        )
        
    def start(self):
        """Start the OSC server."""
        print(f"Starting OSC receiver on {self.ip}:{self.port}")
        try:
            self.server.serve_forever()
        except Exception as e:
            print(f"OSC server error: {e}")
            
    def stop(self):
        """Stop the OSC server."""
        print("Stopping OSC receiver")
        if self.server:
            self.server.shutdown()
            
    def _handle_message(self, addr: str, *args):
        """
        Handle incoming OSC message.
        
        Args:
            addr: OSC address
            *args: OSC message arguments
        """
        try:
            values_dict = {
                addr: np.array(args, dtype=np.float32)
            }
            self.new_data.emit(values_dict)
        except Exception as e:
            print(f"Error handling OSC message: {e}")

class MotionDataReceiver(QtCore.QObject):
    """Processes motion capture data and maintains a sliding window."""
    
    new_data = QtCore.pyqtSignal(np.ndarray)
    
    def __init__(self, data_ids: List[str], joint_count: int, 
                 joint_indices: List[int], window_length: int, parent=None):
        """
        Initialize motion data receiver.
        
        Args:
            data_ids: List of expected OSC addresses
            joint_count: Total number of joints in motion data
            joint_indices: Indices of joints to use for classification
            window_length: Length of sliding window
            parent: Parent Qt object
        """
        super().__init__(parent=parent)
        self.data_ids = data_ids
        self.joint_count = joint_count
        self.joint_indices = joint_indices
        self.window_length = window_length
        
        # Initialize data structures
        self.data_dims = [None] * len(self.data_ids)
        self.data_dim_total = None
        self.data_updated = [False] * len(self.data_ids)
        self.data_window = None
        self.is_running = False
        
    def start(self):
        """Start receiving motion data."""
        self.is_running = True
        print("Motion data receiver started")
        
    def stop(self):
        """Stop receiving motion data."""
        self.is_running = False
        print("Motion data receiver stopped")
        
    def receive(self, new_data: Dict[str, np.ndarray]):
        """
        Process incoming motion data.
        
        Args:
            new_data: Dictionary containing OSC address and data values
        """
        if not self.is_running:
            return
            
        try:
            data_id = list(new_data.keys())[0]

            def osc_pattern_to_regex(pattern):
                # Convert OSC wildcard pattern to regex
                return re.compile('^' + re.escape(pattern).replace('\\*', '.*') + '$')

            matched_index = None
            for idx, pattern in enumerate(self.data_ids):
                regex = osc_pattern_to_regex(pattern)
                if regex.match(data_id):
                    matched_index = idx
                    break

            if matched_index is None:
                return


            data_values = list(new_data.values())
            
            # Filter data based on joint indices
            data_values = np.reshape(data_values, (self.joint_count, -1))
            data_values = data_values[self.joint_indices, :]
            data_values = np.reshape(data_values, (-1))
            
            data_index = matched_index
            
            # Initialize data structures on first reception
            if self.data_dim_total is None:
                self._initialize_data_structures(data_values, data_index)
                if self.data_dim_total is None:
                    return
                    
            # Update sliding window
            self._update_sliding_window(data_values, data_index)
            
        except Exception as e:
            print(f"Error processing motion data: {e}")
            
    def _initialize_data_structures(self, data_values: np.ndarray, data_index: int):
        """Initialize data structures based on received data dimensions."""
        self.data_dims[data_index] = data_values.shape[0]
        
        if None not in self.data_dims:
            self.data_dim_total = sum(self.data_dims)
            self.data_window = np.zeros((self.window_length, self.data_dim_total), dtype=np.float32)
            print(f"Initialized data window: {self.data_window.shape}")
            
    def _update_sliding_window(self, data_values: np.ndarray, data_index: int):
        """Update the sliding window with new data."""
        start_pos = sum(self.data_dims[:data_index])
        end_pos = start_pos + self.data_dims[data_index]
        
        self.data_window[-1, start_pos:end_pos] = data_values
        self.data_updated[data_index] = True
        
        # Check if all data streams have been updated
        if False not in self.data_updated:
            self.new_data.emit(self.data_window.copy())
            
            # Slide window and reset update flags
            self.data_window = np.roll(self.data_window, shift=-1, axis=0)
            self.data_window[-1, :] = 0.0
            self.data_updated = [False] * len(self.data_updated)

class ClassificationSender:
    """Sends classification results via OSC."""
    
    def __init__(self, ip: str, port: int):
        """
        Initialize classification sender.
        
        Args:
            ip: Destination IP address
            port: Destination port
        """
        self.osc_sender = SimpleUDPClient(ip, port)
        print(f"Classification sender initialized: {ip}:{port}")
        
    def send(self, class_probs: np.ndarray):
        """
        Send classification probabilities via OSC.
        
        Args:
            class_probs: Array of class probabilities
        """
        try:
            osc_values = class_probs.tolist()
            self.osc_sender.send_message("/motion/class", osc_values)
        except Exception as e:
            print(f"Error sending classification data: {e}")

"""
GUI Components
"""

class ClassificationBarView:
    """Visualization component for classification results with labels."""
    
    def __init__(self, class_count: int, class_labels: List[str], 
                 colors: List[Tuple[float, float, float]], parent_view):
        """
        Initialize bar view with labels.
        
        Args:
            class_count: Number of classification classes
            class_labels: List of class label strings
            colors: List of RGB color tuples for bars
            parent_view: Parent VisPy view object
        """
        self.class_count = class_count
        self.class_labels = class_labels
        self.parent_view = parent_view
        
        # Calculate bar positions and dimensions
        self.bar_width = 0.8 / class_count
        self.bar_spacing = 1.0 / class_count
        bar_centers_x = np.linspace(self.bar_spacing / 2, 1.0 - self.bar_spacing / 2, class_count)
        
        # Create bars
        self.bars = []
        self.labels = []
        
        for i in range(class_count):
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
                text=class_labels[i],
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
        """
        Update bar heights with new classification values.
        
        Args:
            values: Array of classification probabilities
        """
        for bar, value in zip(self.bars, values):
            bar.center = (bar.center[0], value / 2)
            bar.height = max(abs(value), 0.001)  # Prevent zero height

class VisualizationCanvas:
    """Main visualization canvas containing the bar chart."""
    
    def __init__(self, class_labels: List[str], colors: List[Tuple[float, float, float]], size: Tuple[int, int]):
        """
        Initialize visualization canvas.
        
        Args:
            class_labels: List of class label strings
            colors: List of RGB color tuples
            size: Canvas size (width, height)
        """
        self.size = size
        self.canvas = SceneCanvas(size=size, keys="interactive")
        self.grid = self.canvas.central_widget.add_grid()
        
        # Create main view for bars
        self.bar_view = self.grid.add_view(0, 0, bgcolor="white")
        
        # Initialize bar visualization
        self.bars = ClassificationBarView(
            class_count=len(colors),
            class_labels=class_labels,
            colors=colors,
            parent_view=self.bar_view.scene
        )
        
        # Configure camera
        self.bar_view.camera = "panzoom"
        self.bar_view.camera.set_range(x=(0.0, 1.0), y=(-0.2, 1.0))
        
        # Enable FPS counter
        self.canvas.measure_fps(window=1, callback='%1.1f FPS')
        
    def update(self, new_data: np.ndarray):
        """
        Update visualization with new classification data.
        
        Args:
            new_data: Array of classification probabilities
        """
        self.bars.update(new_data)

class MainWindow(QtWidgets.QMainWindow):
    """Main application window."""
    
    closing = QtCore.pyqtSignal()
    start_classification = QtCore.pyqtSignal()
    stop_classification = QtCore.pyqtSignal()
    
    def __init__(self, canvas: VisualizationCanvas, *args, **kwargs):
        """
        Initialize main window.
        
        Args:
            canvas: Visualization canvas object
        """
        super().__init__(*args, **kwargs)
        self.setWindowTitle("Real-time Motion Classifier")
        self.setWindowIcon(QtGui.QIcon())  # You can add an icon path here
        
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
        """Create control buttons and add to layout."""
        controls_layout = QtWidgets.QHBoxLayout()
        
        # Create buttons
        self.start_button = QtWidgets.QPushButton("Start Classification", self)
        self.stop_button = QtWidgets.QPushButton("Stop Classification", self)
        self.exit_button = QtWidgets.QPushButton("Exit", self)
        
        # Set button properties
        for button in [self.start_button, self.stop_button, self.exit_button]:
            button.setMinimumWidth(120)
            button.setMinimumHeight(30)
            
        # Add buttons to layout
        controls_layout.addWidget(self.start_button)
        controls_layout.addWidget(self.stop_button)
        controls_layout.addWidget(self.exit_button)
        controls_layout.addStretch()  # Add stretch to push buttons to the left
        
        main_layout.addLayout(controls_layout)
        
    def _connect_signals(self):
        """Connect button signals to appropriate slots."""
        self.start_button.clicked.connect(self.start_classification.emit)
        self.stop_button.clicked.connect(self.stop_classification.emit)
        self.exit_button.clicked.connect(self.close)
        
    def closeEvent(self, event):
        """Handle window close event."""
        print("Closing main window")
        self.closing.emit()
        event.accept()

# =============================================================================
# Main Application
# =============================================================================

class MotionClassifierApp(QtCore.QObject):
    """Main application controller that coordinates all components."""
    
    def __init__(self):
        """Initialize the motion classifier application."""
        super().__init__()
        self.osc_thread = None
        self.components = {}
        
    def initialize(self):
        """Initialize all application components."""
        try:
            # Initialize neural network model
            self._initialize_model()
            
            # Initialize OSC components
            self._initialize_osc_components()
            
            # Initialize GUI
            self._initialize_gui()
            
            # Connect signals between components
            self._connect_components()
            
            print("Application initialized successfully")
            return True
            
        except Exception as e:
            print(f"Failed to initialize application: {e}")
            return False
            
    def _initialize_model(self):
        """Initialize and load the trained neural network model."""
        classifier = MotionClassifier(
            input_dim=model_input_dim,
            hidden_dim=model_hidden_dim,
            layer_count=model_layer_count,
            class_count=class_count,
            dropout=model_dropout
        )
        
        classifier.to(device)
        
        # Load trained weights
        if device == 'cuda':
            classifier.load_state_dict(torch.load(model_weights_file))
        else:
            classifier.load_state_dict(
                torch.load(model_weights_file, map_location=torch.device("cpu"))
            )
            
        self.components['classifier'] = LiveClassifier(classifier)
        print(f"Model loaded on {device}")
        
    def _initialize_osc_components(self):
        """Initialize OSC communication components."""
        # OSC receiver
        self.components['osc_receiver'] = OSCReceiver(
            osc_receive_ip, osc_receive_port
        )
        
        # Motion data processor
        self.components['mocap_receiver'] = MotionDataReceiver(
            mocap_data_ids,
            mocap_joint_count,
            mocap_joint_indices,
            mocap_data_window_length
        )
        
        # Classification result sender
        self.components['classify_sender'] = ClassificationSender(
            osc_send_ip, osc_send_port
        )
        
    def _initialize_gui(self):
        """Initialize GUI components."""
        # Create color palette for bars
        bar_colors = [
            colorsys.hsv_to_rgb(1.0 / class_count * i, 1.0, 1.0)
            for i in range(class_count)
        ]
        
        # Create visualization canvas
        self.components['canvas'] = VisualizationCanvas(
            class_labels, bar_colors, canvas_size
        )
        
        # Create main window
        self.components['main_window'] = MainWindow(self.components['canvas'])
        
    def _connect_components(self):
        """Connect signals between all components."""
        # OSC thread setup
        self.osc_thread = QtCore.QThread(parent=self.components['main_window'])
        self.components['osc_receiver'].moveToThread(self.osc_thread)
        
        # OSC data flow
        self.components['osc_receiver'].new_data.connect(
            self.components['mocap_receiver'].receive
        )
        
        # Classification data flow
        self.components['mocap_receiver'].new_data.connect(
            self.components['classifier'].update
        )
        self.components['classifier'].new_data.connect(
            self.components['classify_sender'].send
        )
        self.components['classifier'].new_data.connect(
            self.components['canvas'].update
        )
        
        # GUI controls
        self.components['main_window'].start_classification.connect(
            self.components['mocap_receiver'].start
        )
        self.components['main_window'].stop_classification.connect(
            self.components['mocap_receiver'].stop
        )
        
        # Thread management
        self.osc_thread.started.connect(self.components['osc_receiver'].start)
        self.components['main_window'].closing.connect(self._shutdown)
        
    def run(self):
        """Start the application."""
        if not self.initialize():
            return False
            
        # Show main window
        self.components['main_window'].show()
        
        # Start OSC thread
        self.osc_thread.start()
        
        print("Application started")
        return True
        
    def _shutdown(self):
        """Gracefully shutdown the application."""
        print("Shutting down application...")
        
        # Stop motion data receiver
        if 'mocap_receiver' in self.components:
            self.components['mocap_receiver'].stop()
            
        # Stop OSC receiver
        if 'osc_receiver' in self.components:
            self.components['osc_receiver'].stop()
            
        # Wait for OSC thread to finish
        if self.osc_thread and self.osc_thread.isRunning():
            self.osc_thread.quit()
            self.osc_thread.wait(3000)  # Wait up to 3 seconds
            
        print("Application shutdown complete")

# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Main entry point for the application."""
    # Initialize Qt application
    qt_app = use_app("pyqt5")
    qt_app.create()
    
    # Create and run motion classifier application
    app = MotionClassifierApp()
    
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
