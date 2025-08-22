"""
Real-time Sensor Classifier
===================================

This application receives sensor data via OSC (Open Sound Control), 
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

#osc
from pythonosc import dispatcher
from pythonosc import osc_server
from pythonosc.udp_client import SimpleUDPClient

# pytorch
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from collections import OrderedDict

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

# Sensor data settings

sensor_data_norm_path = "data/results/data/"
sensor_data_ids = ["/accelerometer", "/gyroscope"]
sensor_data_dims = [3, 3]
sensor_data_window_length = 60

# load sensor dara mean and std
with open(sensor_data_norm_path + "mean.pkl", 'rb') as f:
    data_mean = pickle.load(f)
with open(sensor_data_norm_path + "std.pkl", 'rb') as f:
    data_std = pickle.load(f)  
    
# Classification Settings

class_names = [ "fluidity", "staccato", "thrusting"]
class_count = len(class_names)

# Model settings

model_input_dim = data_mean.shape[0]
model_hidden_dim = 128
model_layer_count = 3
model_dropout = 0.3

model_weights_file = "../SensorClassifier/results/weights/classifier_weights_epoch_200.pth"

# OSC Settings

osc_receive_ip = "0.0.0.0"
osc_receive_port = 9007
osc_send_ip = "127.0.0.1"
osc_send_port = 10000

# GUI Settings

canvas_size = (600, 400)
class_labels = class_names

"""
Load Sensor Normalization Parameters
"""

try:
    with open(sensor_data_norm_path + "mean.pkl", 'rb') as f:
        data_mean = pickle.load(f)
    with open(sensor_data_norm_path + "std.pkl", 'rb') as f:
        data_std = pickle.load(f)
    input_dim = data_mean.shape[0]
except FileNotFoundError as e:
    print(f"Could not load normalization data: {e}")
    sys.exit(1)


"""
Create Model
"""

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

class OscReceiver(QtCore.QObject):
    
    new_data = QtCore.pyqtSignal(dict)
    
    def __init__(self, ip, port, parent=None):
        super().__init__(parent=parent)
        
        self.ip = ip
        self.port = port
        
        self.dispatcher = dispatcher.Dispatcher()
        self.dispatcher.map("/*", self.receive)
        self.server = osc_server.BlockingOSCUDPServer((self.ip, self.port), self.dispatcher)
        
    def start(self):
        print("OscReceiver start")
        self.server.serve_forever()
    
    def stop(self):
        print("OscReceiver stop")
        self.server.shutdown()
    
    def receive(self, addr, *args):
        
        osc_address = addr
        osc_values = args
        
        #print("osc_address ", osc_address, " osc_values ", osc_values)
        
        values_dict = {
            osc_address: osc_values
            }
        
        self.new_data.emit(values_dict)

class SensorDataReceiver(QtCore.QObject):
    
    new_data = QtCore.pyqtSignal(np.ndarray)
    
    def __init__(self, sensor_ids, sensor_data_dims, window_length, parent=None):
        super().__init__(parent=parent)
        
        self.sensor_ids = sensor_ids
        self.sensor_data_dims = sensor_data_dims
        self.window_length = window_length
        self.sensor_values = [ np.zeros((self.window_length, self.sensor_data_dims[sI])) for sI in range(len(self.sensor_data_dims)) ]
        self.sensor_updated = [ False ] * len(self.sensor_ids)
        
        self.running = False
        
    def start(self):
        self.running = True
        
    def stop(self):
        self.running = False

    def receive(self, new_data):
        
        if self.running  == False:
            return

        sensor_id = list(new_data.keys())[0]
        sensor_value = list(new_data.values())[0]
        
        if sensor_id not in self.sensor_ids:
            return
    
        sensor_index = self.sensor_ids.index(sensor_id)
        
        self.sensor_values[sensor_index] = np.roll(self.sensor_values[sensor_index], -1, axis=0)
        self.sensor_values[sensor_index][-1] = sensor_value
        self.sensor_updated[sensor_index] = True
 
        if self.sensor_updated.count(True) == len(self.sensor_ids):
            
            sensor_values_combined = np.concatenate(self.sensor_values, axis=1)
            self.new_data.emit(sensor_values_combined)
            
            self.sensor_updated = [ False ] * len(self.sensor_ids)

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
        self.components['osc_receiver'] = OscReceiver(
            osc_receive_ip, osc_receive_port
        )
        
        # Motion data processor
        self.components['sensor_receiver'] = SensorDataReceiver(
            sensor_data_ids,
            sensor_data_dims,
            sensor_data_window_length
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
            self.components['sensor_receiver'].receive
        )
        
        # Classification data flow
        self.components['sensor_receiver'].new_data.connect(
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
            self.components['sensor_receiver'].start
        )
        self.components['main_window'].stop_classification.connect(
            self.components['sensor_receiver'].stop
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
        
        # Stop sensor data receiver
        if 'sensor_receiver' in self.components:
            self.components['sensor_receiver'].stop()
            
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