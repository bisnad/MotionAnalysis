"""
imports
"""

# general
import os, sys, time, subprocess
import numpy as np
import pickle

#osc
from pythonosc import dispatcher
from pythonosc import osc_server
from pythonosc.udp_client import SimpleUDPClient

# pytorch
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch import nn
import torch.nn.functional as nnF
import torch.optim as optim
from collections import OrderedDict

#gui
from PyQt5 import QtWidgets, QtCore
from vispy import scene
from vispy.app import use_app, Timer
from vispy.scene import SceneCanvas, visuals


"""
Device
"""

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

"""
Data
"""

data_norm_path = "../MocapClassifier/results/data/"
data_sensor_ids = ["/accelerometer", "/gyroscope"]
data_sensor_dims = [3, 3]
data_window_length = 60

# load sensor dara mean and std
with open(data_norm_path + "mean.pkl", 'rb') as f:
    data_mean = pickle.load(f)
with open(data_norm_path + "std.pkl", 'rb') as f:
    data_std = pickle.load(f)  
    
"""
Model settings
"""

input_dim = sum(data_sensor_dims)
hidden_dim = 32
layer_count = 3
class_count = 3
model_weights_file = "../MocapClassifier/results/weights/classifier_weights_epoch_100.pth"

"""
Create Model
"""

# create model
class Classifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_count, class_count):
        super().__init__()

        self.rnn = nn.LSTM(input_dim, hidden_dim, layer_count, batch_first=True)
        self.fc = nn.Linear(hidden_dim, class_count)
        self.sm = nn.LogSoftmax(dim=1)

    def forward(self, x):

        x, (h, c) = self.rnn(x)
        x = x[:, -1, :] # only last time step 
        x = self.fc(x)
        y = self.sm(x)
        return y

class LiveClassifier(QtCore.QObject):
    
    new_data = QtCore.pyqtSignal(np.ndarray)

    def __init__(self, classifier, parent=None):
        super().__init__(parent=parent)
        
        self.classifier = classifier.eval()
        
    def update(self, input_data):
        
        input_norm = (input_data - data_mean) / data_std
        
        #print("input_data s ", input_data.shape)
        
        input = torch.tensor(input_norm, dtype=torch.float32).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = classifier(input)
            
        class_prob = torch.exp(outputs).detach().cpu().numpy()[0]
        
        self.new_data.emit(class_prob)

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

class SensorReceiver(QtCore.QObject):
    
    new_data = QtCore.pyqtSignal(np.ndarray)
    
    def __init__(self, sensor_ids, data_sensor_dims, window_length, parent=None):
        super().__init__(parent=parent)
        
        self.sensor_ids = sensor_ids
        self.data_sensor_dims = data_sensor_dims
        self.window_length = window_length
        self.sensor_values = [ np.zeros((self.window_length, self.data_sensor_dims[sI])) for sI in range(len(self.data_sensor_dims)) ]
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

class ClassifySender:
    def __init__(self, ip, port):
        self.osc_sender = SimpleUDPClient(ip, port)
    
    def send(self, class_probs):
        osc_values = np.reshape(class_probs, (-1)).tolist()
        self.osc_sender.send_message("/motion/class", osc_values) 
        

"""
GUI
"""

class BarView:
    def __init__(self, value_count, colors, parent_view=None):
        
        self.value_count = value_count
        self.parent_view = parent_view
        self.values = np.zeros(self.value_count)
        
        bar_width = 1.0 / value_count
        bar_centers_x = np.linspace(bar_width / 2, 1.0 - bar_width / 2, self.value_count)
        
        self.bars = [ visuals.Rectangle(center=(bar_centers_x[i], 0.0), width=bar_width, height=0.01, color=colors[i]) for i in range(self.value_count) ]
        self.compound = visuals.Compound(self.bars, parent=self.parent_view)
        
    def update(self, values):

        for bar, value in zip(self.bars, values):
            bar.center = (bar.center[0], value / 2)
            bar.height = abs(value) + 0.0001
            
class Canvas:
    def __init__(self, colors, size):
        self.size = size
        self.canvas = SceneCanvas(size = size, keys="interactive")
        self.grid = self.canvas.central_widget.add_grid()

        self.bar_view = self.grid.add_view(0, 0, bgcolor="white")
        self.bars = BarView(len(colors), colors, self.bar_view.scene)
        
        self.bar_view.camera = "panzoom"
        self.bar_view.camera.set_range(x=(0.0, 1.0), y=(0.0, 1.0))
        
        self.canvas.measure_fps(window=1, callback='%1.1f FPS')

    def update(self, new_data):
        
        #print("new_data ", new_data)
        
        self.bars.update(new_data)
            
class MainWindow(QtWidgets.QMainWindow):
    
    closing = QtCore.pyqtSignal()
    
    def __init__(self, canvas, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.setWindowTitle("Mocap Classifier")
        
        # main layout
        central_widget = QtWidgets.QWidget()
        main_layout = QtWidgets.QVBoxLayout()
        
        #self._controls = Controls()
        #main_layout.addWidget(self._controls)
        self.canvas = canvas
        main_layout.addWidget(self.canvas.canvas.native)
        
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)
        
        # controls layout
        controls_layout = QtWidgets.QHBoxLayout()
        
        self.start_buttom = QtWidgets.QPushButton("start", self)
        self.stop_buttom = QtWidgets.QPushButton("stop", self)
        self.spacer = QtWidgets.QSpacerItem(0, 10, QtWidgets.QSizePolicy.MinimumExpanding)
        
        self.start_buttom.setFixedWidth(80)
        self.stop_buttom.setFixedWidth(80)
        
        controls_layout.addWidget(self.start_buttom)
        controls_layout.addWidget(self.stop_buttom)
        controls_layout.addItem(self.spacer)
        
        main_layout.addLayout(controls_layout)

    def closeEvent(self, event):
        print("Closing main window!")
        self.closing.emit()
        return super().closeEvent(event)

if __name__ == "__main__":
    
    # model
    
    classifier = Classifier(input_dim, hidden_dim, layer_count, class_count)
    classifier.to(device)
    classifier.load_state_dict(torch.load(model_weights_file))
    
    liveClassifier = LiveClassifier(classifier)
    
    # osc
    osc_receive_ip = "0.0.0.0"
    osc_receive_port = 9000
    osc_record_active = False
    
    osc_send_ip = "127.0.0.1"
    osc_send_port = 10000
    
    osc_receiver = OscReceiver(osc_receive_ip, osc_receive_port)
    sensor_receiver = SensorReceiver(data_sensor_ids, data_sensor_dims, data_window_length)
    classify_sender = ClassifySender(osc_send_ip, osc_send_port)

    #gui
    app = use_app("pyqt5")
    app.create()
    
    canvas = Canvas(((1.0, 0.0, 0.0, 1.0), (0.0, 1.0, 0.0, 1.0), (0.0, 0.0, 1.0, 1.0)), (400, 300))
    win = MainWindow(canvas)
    win.start_buttom.clicked.connect(sensor_receiver.start)
    win.stop_buttom.clicked.connect(sensor_receiver.stop)
    
    osc_thread = QtCore.QThread(parent=win)
    osc_receiver.moveToThread(osc_thread)
    
    osc_receiver.new_data.connect(sensor_receiver.receive)
    sensor_receiver.new_data.connect(liveClassifier.update)
    liveClassifier.new_data.connect(classify_sender.send)
    liveClassifier.new_data.connect(canvas.update)
    
    osc_thread.started.connect(osc_receiver.start)
    
    win.closing.connect(osc_receiver.stop)
    osc_thread.finished.connect(osc_receiver.deleteLater)
    
    win.show()
    #win.setFocus()
    osc_thread.start()
    app.run()


