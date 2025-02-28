import torch
import numpy as np

from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt

from vispy import scene
from vispy.app import use_app, Timer
from vispy.scene import SceneCanvas, visuals


from pathlib import Path
from threading import Thread, Event
import time
from time import sleep
import datetime

import motion_pipeline

config = {"pipeline": None,
          "sender": None}

class BarView:
    def __init__(self, max_value_count, colors, parent_view=None):
        
        self.max_value_count = max_value_count
        self.value_count = max_value_count
        self.colors = colors
        self.parent_view = parent_view
        
        bar_width = 1.0 / self.value_count
        bar_centers_x = np.linspace(bar_width / 2, 1.0 - bar_width / 2, self.value_count)
        
        self.bars = [ visuals.Rectangle(center=(bar_centers_x[i], 0.0), width=bar_width, height=0.01, color=colors[i]) for i in range(self.value_count) ]
        self.compound = visuals.Compound(self.bars, parent=self.parent_view)
        
    def resetValueCount(self, value_count):
        
        #print("BarView resetValueCount value_count ", value_count)
        
        self.value_count = value_count
        
        bar_width = 0.9 / self.value_count
        bar_centers_x = np.linspace(bar_width / 2, 1.0 - bar_width / 2, self.value_count)
        
        for vI in range(self.max_value_count ):
            
            if vI < self.value_count:
                self.bars[vI].center = (bar_centers_x[vI], 0.0)
                self.bars[vI].width = bar_width
                self.bars[vI].color = self.colors[vI]
            
            if vI >= self.value_count:
                self.bars[vI].center = (2.0, 0.0)
                self.bars[vI].width = 0.01
                self.bars[vI].color = [0.0, 0.0, 0.0, 0.0]
        
    def update(self, values):
        
        #print("BarView update values ", values)
        
        value_count = values.shape[0]
        value_count = min(value_count, self.max_value_count )
        
        #print("value_count ", value_count, " self.value_count ", self.value_count)
        
        if value_count != self.value_count:
            self.resetValueCount(value_count)
            
        for vI in range(self.value_count):
            bar = self.bars[vI]
            value = values[vI]

            bar.center = (bar.center[0], value / 2)
            bar.height = abs(value) + 0.0001
            
class DataView:
    def __init__(self, title, max_value_dim, value_range, time_steps, colors):
        
        self.title = title
        self.max_value_dim = max_value_dim
        self.value_dim = max_value_dim
        self.value_range = value_range
        self.time_steps = time_steps
        self.colors = colors
        
        if value_range[0] > value_range[1]:
            self.autoscale = True
        else:
            self.autoscale = False
        
        self.canvas = SceneCanvas()
        self.grid = self.canvas.central_widget.add_grid()

        yaxis = scene.AxisWidget(orientation='left',
                                 axis_font_size=12,
                                 axis_label_margin=50,
                                 tick_label_margin=5)
        yaxis.width_max = 80
        self.grid.add_widget(yaxis, row=0, col=0)
        
        self.bar_view = self.grid.add_view(0, 1, bgcolor="black")

        self.bars = BarView(self.max_value_dim, self.colors, self.bar_view.scene)

        self.bar_view.camera = "panzoom"
        self.bar_view.camera.set_range(x=(0.0, 1.0), y=(self.value_range[0], self.value_range[1]))

        yaxis.link_view(self.bar_view)
        
    def set_value_range(self, value_range):

        self.value_range = value_range
        if value_range[0] > value_range[1]:
            self.autoscale = True
        else:
            self.autoscale = False
        
        self.bar_view.camera.set_range(x=(0.0, 1.0), y=(value_range[0], value_range[1]))
        
    def update_data(self, data):
        
        data_dim = min(data.shape[0], self.max_value_dim)
        data = data[:data_dim]
        
        #print("DataView update_data ", data)

        if self.autoscale == True:
            
            range_changed = False
            min_value = np.min(data)
            max_value = np.max(data)

            if self.value_range[0] > min_value:
                self.value_range[0] = min_value
                range_changed = True
            if self.value_range[1] < max_value:
                self.value_range[1] = max_value
                range_changed = True

            if range_changed == True:
                self.bar_view.camera.set_range(x=(0.0, 1.0), y=(self.value_range[0], self.value_range[1]))
        
        self.bars.update(data)

class Canvas:
    def __init__(self, size):
        self.size = size
        #self.canvas = SceneCanvas(size = size, keys="interactive")
        self.canvas = SceneCanvas(size = size)
        self.grid = self.canvas.central_widget.add_grid()
        self.views = {}
        
        #self.canvas.measure_fps(window=1, callback='%1.1f FPS')
        
    def add_sensor_view(self, name, max_value_dim, value_range, time_steps, colors):
        sensor_view = DataView(name, max_value_dim, value_range, time_steps, colors)
        self.grid.add_widget(sensor_view.canvas.central_widget, len(self.views), 0)
        self.views[name] = sensor_view
        
    def set_value_range(self, name, value_range):

        self.views[name].set_value_range(value_range)

    def update_data(self, new_data):
        
        key = list(new_data.keys())[0]
        value = list(new_data.values())[0]
        
        if key in self.views:
            self.views[key].update_data(value)
            

class MotionGui(QtWidgets.QWidget):
    
    def __init__(self, config):
        super().__init__()
        
        self.pipeline = config["pipeline"]
        self.sender = config["sender"]
        
        self.q_start_buttom = QtWidgets.QPushButton("start", self)
        self.q_start_buttom.clicked.connect(self.start)  
        
        self.q_stop_buttom = QtWidgets.QPushButton("stop", self)
        self.q_stop_buttom.clicked.connect(self.stop)  
        
        fps = 1.0 / self.pipeline.updateInterval
        
        self.q_fps = QtWidgets.QSpinBox(self)
        self.q_fps.setMinimum(0)
        self.q_fps.setMaximum(200)
        self.q_fps.setValue(int(fps))
        self.q_fps.valueChanged.connect(self.change_fps)  

        self.q_button_grid = QtWidgets.QGridLayout()
        self.q_button_grid.addWidget(self.q_start_buttom,0,0)
        self.q_button_grid.addWidget(self.q_stop_buttom,0,1)
        self.q_button_grid.addWidget(self.q_fps,0,2)
        
        # canvas
        self.canvas = Canvas((400, 400))
        self.canvas.add_sensor_view("data", 100, [1000.0, -1000.0], 100, [(1.0, 1.0, 1.0, 0.8)] * 100)
        
        self.canvas_active = True
        
        self.q_canvas_toggle = QtWidgets.QCheckBox("canvas", self)
        self.q_canvas_toggle.stateChanged.connect(lambda:self.toggle_canvas(self.q_canvas_toggle))  
        self.q_canvas_toggle.setChecked(self.canvas_active)
        
        self.q_canvas_grid = QtWidgets.QGridLayout()
        self.q_canvas_grid.addWidget(self.canvas.canvas.native,0,0)
        self.q_canvas_grid.addWidget(self.q_canvas_toggle,1,0)
        
 
        # send items
        self.sendItems = {"pos_scaled": False,
                          "pos_smooth": False,
                          "velocity": False,
                          "velocity_smooth": False,
                          "acceleration": False,
                          "acceleration_smooth": False,
                          "jerk": False,
                          "jerk_smooth": False,
                          "pos_scalar": False,
                          "velocity_scalar": False,
                          "acceleration_scalar": False,
                          "jerk_scalar": False,
                          "qom": False,
                          "bbox": False,
                          "bsphere": False,
                          "flow_effort": False,
                          "time_effort": False,
                          "weight_effort": False,
                          "space_effort": False}
        self.showItem = ""
        
        self.q_sendItems = QtWidgets.QListWidget()
        for key, value in self.sendItems.items():
            QtWidgets.QListWidgetItem(key, self.q_sendItems)

    
        self.q_sendItems_layout = QtWidgets.QVBoxLayout()
        self.q_sendItems_layout.addWidget(self.q_sendItems)
        
        for i in range(self.q_sendItems.count()):
            q_sendItem = self.q_sendItems.item(i)
            q_sendItem.setFlags(q_sendItem.flags() | Qt.ItemIsUserCheckable)
            
            q_checked = self.sendItems[q_sendItem.text()]
            
            if q_checked == False:
                q_sendItem.setCheckState(Qt.Unchecked)
            else:
                q_sendItem.setCheckState(Qt.Checked);
        
        self.q_sendItems.itemChanged.connect(lambda:self.change_send_item(self.q_sendItems))  
        
        # osc sender
        self.sender_active = True
        
        self.q_sender_toggle = QtWidgets.QCheckBox("osc", self)
        self.q_sender_toggle.stateChanged.connect(lambda:self.toggle_sender(self.q_sender_toggle))  
        self.q_sender_toggle.setChecked(self.sender_active)
          
        sender_ip_string, sender_port = self.sender.get_address()
        sender_ip_list = sender_ip_string.split(".")
        sender_ip = [ int(el) for el in sender_ip_list ]
        
        self.sender_ip = sender_ip
        self.sender_port = sender_port
  
        self.q_sender_ip = []
        for i in range(4):
            _w = QtWidgets.QSpinBox(self)
            _w.setMinimum(0)
            _w.setMaximum(255)
            _w.setValue(self.sender_ip[i])
            _w.valueChanged.connect(lambda:self.change_sender_ip(_w))  

            self.q_sender_ip.append(_w)
        self.q_sender_port = QtWidgets.QSpinBox(self)
        self.q_sender_port.setMinimum(0)
        self.q_sender_port.setMaximum(65535)
        self.q_sender_port.setValue(self.sender_port)
        self.q_sender_port.valueChanged.connect(lambda:self.change_sender_port(self.q_sender_port))  
        
        self.q_sender_grid = QtWidgets.QGridLayout()
        self.q_sender_grid.addWidget(self.q_sender_toggle,0,0)
        self.q_sender_grid.addWidget(self.q_sender_ip[0],0,1)
        self.q_sender_grid.addWidget(self.q_sender_ip[1],0,2)
        self.q_sender_grid.addWidget(self.q_sender_ip[2],0,3)
        self.q_sender_grid.addWidget(self.q_sender_ip[3],0,4)
        self.q_sender_grid.addWidget(self.q_sender_port,0,5)
        
        self.q_grid = QtWidgets.QGridLayout()
        #self.q_grid.addWidget(self.pose_canvas,0,0)
        self.q_grid.addLayout(self.q_button_grid,0,0)
        self.q_grid.addLayout(self.q_canvas_grid,1,0)
        self.q_grid.addLayout(self.q_sendItems_layout,2,0)
        self.q_grid.addLayout(self.q_sender_grid,3,0)
        self.q_grid.addLayout(self.q_sendItems_layout,4,0)        
        
        self.q_grid.setRowStretch(0, 0)
        self.q_grid.setRowStretch(1, 0)
        self.q_grid.setRowStretch(2, 0)
        self.q_grid.setRowStretch(3, 0)
        self.q_grid.setRowStretch(4, 0)
        
        self.setLayout(self.q_grid)
        
        self.setGeometry(50,50,512,612)
        self.setWindowTitle("Motion Analysis")
        
    def start(self):
        self.data_thread_event = Event()
        self.data_thread = Thread(target = self.update)
        
        self.data_thread.start()
        
    def stop(self):
        self.data_thread_event.set()
        self.data_thread.join()
        
    def change_fps(self, fps):
        
        update_interval = 1.0 / int(fps)
        self.pipeline.setUpdateInterval(update_interval)
                
    def update(self):
        
        while self.data_thread_event.is_set() == False:
            
            
            start_time = time.time() 
            
            self.pipeline.update()
            self.update_osc()
            if self.canvas_active  == True:
                self.update_view()
            
            end_time = time.time()   
            
            #print("update time ", end_time - start_time, " interval ", self.pose_thread_interval)
            
            next_update_interval = max(self.pipeline.updateInterval - (end_time - start_time), 0.0)
            
            #print("start_time ", start_time, " end_time ", end_time, " next_update_interval ", next_update_interval)
            
            sleep(next_update_interval)

    def update_osc(self):
        
        if self.sendItems["pos_scaled"] == True:
            osc_values = np.reshape(self.pipeline.posScaled, (-1)).tolist()
            self.sender.send("/mocap/0/joint/pos_world", osc_values)
        if self.sendItems["pos_smooth"] == True:
            osc_values = np.reshape(self.pipeline.posSmooth, (-1)).tolist()
            self.sender.send("/mocap/0/joint/pos_smooth", osc_values) 
        if self.sendItems["velocity"] == True:
            osc_values = np.reshape(self.pipeline.vel, (-1)).tolist()
            self.sender.send("/mocap/0/joint/velocity", osc_values) 
        if self.sendItems["velocity_smooth"] == True:
            osc_values = np.reshape(self.pipeline.velSmooth, (-1)).tolist()
            self.sender.send("/mocap/0/joint/velocity_smooth", osc_values) 
        if self.sendItems["acceleration"] == True:
            osc_values = np.reshape(self.pipeline.accel, (-1)).tolist()
            self.sender.send("/mocap/0/joint/accel", osc_values)             
        if self.sendItems["acceleration_smooth"] == True:
            osc_values = np.reshape(self.pipeline.accelSmooth, (-1)).tolist()
            self.sender.send("/mocap/0/joint/acceleration_smooth", osc_values)    
        if self.sendItems["jerk"] == True:
            osc_values = np.reshape(self.pipeline.jerk, (-1)).tolist()
            self.sender.send("/mocap/0/joint/jerk", osc_values)    
        if self.sendItems["jerk_smooth"] == True:
            osc_values = np.reshape(self.pipeline.jerkSmooth, (-1)).tolist()
            self.sender.send("/mocap/0/joint/jerk_smooth", osc_values)   
        if self.sendItems["pos_scalar"] == True:
            osc_values = np.reshape(self.pipeline.pos_scalar, (-1)).tolist()
            self.sender.send("/mocap/0/joint/position_scalar", osc_values)   
        if self.sendItems["velocity_scalar"] == True:
            osc_values = np.reshape(self.pipeline.vel_scalar, (-1)).tolist()
            self.sender.send("/mocap/0/joint/velocity_scalar", osc_values)   
        if self.sendItems["acceleration_scalar"] == True:
            osc_values = np.reshape(self.pipeline.accel_scalar, (-1)).tolist()
            self.sender.send("/mocap/0/joint/acceleration_scalar", osc_values)              
        if self.sendItems["jerk_scalar"] == True:
            osc_values = np.reshape(self.pipeline.jerk_scalar, (-1)).tolist()
            self.sender.send("/mocap/0/joint/jerk_scalar", osc_values)      
        if self.sendItems["qom"] == True:
            osc_values = np.reshape(self.pipeline.qom, (-1)).tolist()
            self.sender.send("/mocap/0/qom", osc_values)
        if self.sendItems["bbox"] == True:
            osc_values = np.reshape(self.pipeline.bbox, (-1)).tolist()
            self.sender.send("/mocap/0/bbox", osc_values)
        if self.sendItems["bsphere"] == True:
            osc_values = np.reshape(self.pipeline.bsphere, (-1)).tolist()
            self.sender.send("/mocap/0/bsphere", osc_values)
        if self.sendItems["flow_effort"] == True:
            osc_values = np.reshape(self.pipeline.flow_effort, (-1)).tolist()
            self.sender.send("/mocap/0/flow_effort", osc_values)
        if self.sendItems["time_effort"] == True:
            osc_values = np.reshape(self.pipeline.time_effort, (-1)).tolist()
            self.sender.send("/mocap/0/time_effort", osc_values)
        if self.sendItems["weight_effort"] == True:
            osc_values = np.reshape(self.pipeline.weight_effort, (-1)).tolist()
            self.sender.send("/mocap/0/weight_effort", osc_values)
        if self.sendItems["space_effort"] == True:
            osc_values = np.reshape(self.pipeline.space_effort, (-1)).tolist()
            self.sender.send("/mocap/0/space_effort", osc_values)

    def update_view(self):
        
        for i in range(self.q_sendItems.count()):
            q_sendItem = self.q_sendItems.item(i)
            
            if q_sendItem.isSelected() == True:
                
                q_text = q_sendItem.text()
                
                # reset value scale if selected item changes
                if self.showItem != q_text:
                    self.canvas.set_value_range("data", [1000, -1000])
                    self.showItem = q_text
                
                # show values of select item
                if self.showItem == "position_scaled":
                    view_data = {"data": self.pipeline.posScaled.flatten()}
                    self.canvas.update_data(view_data)
                elif self.showItem == "position_smooth":
                    view_data = {"data": self.pipeline.posSmooth.flatten()}
                    self.canvas.update_data(view_data)                
                elif self.showItem == "velocity":
                    view_data = {"data": self.pipeline.vel.flatten()}
                    self.canvas.update_data(view_data)       
                elif self.showItem == "velocity_smooth":
                    view_data = {"data": self.pipeline.velSmooth.flatten()}
                    self.canvas.update_data(view_data)     
                elif self.showItem == "acceleration":
                    view_data = {"data": self.pipeline.accel.flatten()}
                    self.canvas.update_data(view_data)  
                elif self.showItem == "acceleration_smooth":
                    view_data = {"data": self.pipeline.accelSmooth.flatten()}
                    self.canvas.update_data(view_data)  
                elif self.showItem == "jerk":
                    view_data = {"data": self.pipeline.jerk.flatten()}
                    self.canvas.update_data(view_data)             
                elif self.showItem == "jerk_smooth":
                    view_data = {"data": self.pipeline.jerkSmooth.flatten()}
                    self.canvas.update_data(view_data)   
                elif self.showItem == "position_scalar":
                    view_data = {"data": self.pipeline.pos_scalar.flatten()}
                    self.canvas.update_data(view_data)   
                elif self.showItem == "velocity_scalar":
                    view_data = {"data": self.pipeline.vel_scalar.flatten()}
                    self.canvas.update_data(view_data)    
                elif self.showItem == "acceleration_scalar":
                    view_data = {"data": self.pipeline.accel_scalar.flatten()}
                    self.canvas.update_data(view_data)          
                elif self.showItem == "jerk_scalar":
                    view_data = {"data": self.pipeline.jerk_scalar.flatten()}
                    self.canvas.update_data(view_data)          
                if self.showItem == "qom":
                    view_data = {"data": self.pipeline.qom.flatten()}
                    self.canvas.update_data(view_data)
                elif self.showItem == "bbox":
                    view_data = {"data": self.pipeline.bbox.flatten()}                    
                    self.canvas.update_data(view_data)
                elif self.showItem == "bsphere":
                    view_data = {"data": self.pipeline.bsphere.flatten()}
                    self.canvas.update_data(view_data)
                elif self.showItem == "flow_effort":
                    view_data = {"data": self.pipeline.flow_effort.flatten()}
                    self.canvas.update_data(view_data)
                elif self.showItem == "time_effort":
                    view_data = {"data": self.pipeline.time_effort.flatten()}
                    self.canvas.update_data(view_data)
                elif self.showItem == "weight_effort":
                    view_data = {"data": self.pipeline.weight_effort.flatten()}
                    self.canvas.update_data(view_data)
                elif self.showItem == "space_effort":
                    view_data = {"data": self.pipeline.space_effort.flatten()}
                    self.canvas.update_data(view_data)

                
                break
        
        #view_data = {"data": self.pipeline.space_effort}
        #self.canvas.update_data(view_data)
    

    def change_send_item(self, widget):
        #print("change_send_item")
        
        for i in range(widget.count()):
            q_sendItem = widget.item(i)
            
            #print("i ", i, " text ", q_sendItem.text())

            q_text = q_sendItem.text()
            q_state = q_sendItem.checkState()
            
            if q_state == Qt.Checked:
                self.sendItems[q_text] = True
            else:
                self.sendItems[q_text] = False

    def toggle_canvas(self, widget):
        self.canvas_active = widget.isChecked()

    def toggle_sender(self, widget):
        self.sender_on = widget.isChecked()

        self.sender.set_active(self.sender_on)

    def change_sender_ip(self, widget):
        
        sender_widget = super().sender()

        if sender_widget == self.q_sender_ip[0]:
            self.sender_ip[0] = sender_widget.value()
        elif sender_widget == self.q_sender_ip[1]:
            self.sender_ip[1] = sender_widget.value()
        elif sender_widget == self.q_sender_ip[2]:
            self.sender_ip[2] = sender_widget.value()
        else:
            self.sender_ip[3] = sender_widget.value()
            
        sender_active = self.sender.get_active()
        sender_ip = "{}.{}.{}.{}".format(self.sender_ip[0], self.sender_ip[1], self.sender_ip[2], self.sender_ip[3])
        sender_port = self.sender_port
        
        if sender_active == True:
            self.sender.set_active(False)
            
        self.sender.set_address(sender_ip, sender_port)
            
        if sender_active == True:
            self.sender.set_active(True)
    
    
    def change_sender_port(self, widget):
        self.sender_port = widget.value()  

        sender_active = self.sender.get_active()
        sender_ip = "{}.{}.{}.{}".format(self.sender_ip[0], self.sender_ip[1], self.sender_ip[2], self.sender_ip[3])
        sender_port = self.sender_port
        
        if sender_active == True:
            self.sender.set_active(False)
            
        self.sender.set_address(sender_ip, sender_port)
            
        if sender_active == True:
            self.sender.set_active(True)
            
