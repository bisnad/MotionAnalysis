import torch
import numpy as np

from PyQt5 import QtWidgets
from PyQt5 import QtGui
from PyQt5.QtCore import Qt
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from pathlib import Path

from threading import Thread, Event
import time
from time import sleep
import datetime

# 2d pose
# currently fixed for coco (see https://mmpose.readthedocs.io/en/latest/dataset_zoo/2d_body_keypoint.html)
# 0 -> 1
# 0 -> 2
# 1 -> 2
# 1 -> 3
# 2 -> 4
# 3 -> 5
# 4 -> 6
# 5 -> 6
# 5 -> 7
# 7 -> 9
# 6 -> 8
# 8 -> 10
# 5 -> 11
# 6 -> 12
# 11 -> 12
# 11 -> 13
# 13 -> 15
# 12 -> 14
# 14 -> 16
# currently fixed for coco (see https://mmpose.readthedocs.io/en/latest/dataset_zoo/2d_body_keypoint.html)
pose2d_topology = np.array(
[[0,1],[0,2],[1,2],[1,3],[2,4],[3,5],[4,6],[5,6],[5,7],[7,9],[6,8],[8,10],[5,11],[6,12],[11,12],[11,13],[13,15],[12,14],[14,16]]
)

# 3d pose
# currently fixed for h36m (see https://mmpose.readthedocs.io/en/latest/dataset_zoo/3d_body_keypoint.html)
# 0 -> 1
# 0 -> 4
# 0 -> 7
# 1 -> 2
# 2 -> 3
# 4 -> 5
# 5 -> 6
# 7 -> 8
# 8 -> 9
# 8 -> 11
# 8 -> 14
# 9 -> 10
# 11 -> 12
# 12 -> 13
# 14 -> 15
# 15 -> 16
pose3d_topology = np.array(
[[0,1],[0,4],[0,7],[1,2],[2,3],[4,5],[5,6],[7,8],[8,9],[8,11],[8,14],[9,10],[11,12],[12,13],[14,15],[15,16]]
)

config = {"motion_model": None,
          "pose2d_size": (500, 300),
          "pose3d_view_min": np.array([-100, -100, -100], dtype=np.float32),
          "pose3d_view_max": np.array([100, 100, 100], dtype=np.float32),
          "pose3d_view_ele": 0,
          "pose3d_view_azi": 180,
          "pose3d_view_dist": 250,
          "update_interval": 0.02,
          "sender": None
          }

class Pose2DCanvas(QtWidgets.QWidget):
    
    def __init__(self, config):
        
        super().__init__()
        self.setGeometry(0, 0, config["pose2d_size"][0], config["pose2d_size"][1])
        self.image = None
        self.pixmap = None
        self.keypoints = None
        self.pose_topology = pose2d_topology # should come from config
        
    def updateImage(self, image):
        self.image = QtGui.QImage(image.data, image.shape[1], image.shape[0], image.strides[0], QtGui.QImage.Format_BGR888)
        self.pixmap = QtGui.QPixmap.fromImage(self.image)
        
    def updateKeypoints(self, keypoints, keypoints_visible):
        self.keypoints = keypoints
        self.keypoints_visible = keypoints_visible
        
        #print("Pose2DCanvas self.keypoints s ", self.keypoints.shape)
        
    def paintEvent(self, event):

        if self.pixmap is None:
            return

        painter = QtGui.QPainter(self)
        painter.drawPixmap(self.rect(), self.pixmap)
        
        if self.keypoints is None:
            return
        
        pen = QtGui.QPen(QtGui.QColor(255.0, 0.0, 0.0, 150), 1.5)
        painter.setPen(pen)
        
        image_size = self.pixmap.size()
        render_size = self.rect().size()

        render_scale = np.array([render_size.width(), render_size.height()]) / np.array([image_size.width(), image_size.height()])
        
        #print("render_scale ", render_scale)
        #print("self.pixmap ", self.pixmap.size())
        #print("self.rect ", self.rect().size())
        
        edge_count = self.pose_topology.shape[0]
        
        for keypoints, visible in zip(self.keypoints, self.keypoints_visible):
            
            #joint_count = keypoints.shape[0] 
            #print("joint_count ", joint_count)
            
            keypoints_scaled = keypoints * render_scale

            for eI in range(edge_count):
                
                point1 = keypoints_scaled[self.pose_topology[eI][0]]
                point2 = keypoints_scaled[self.pose_topology[eI][1]]
                
                painter.drawLine(point1[0], point1[1], point2[0], point2[1])
        

class Pose3DCanas:
    
    def __init__(self, config):
        
        self.pose_canvas = gl.GLViewWidget()
        self.pose_canvas_lines = gl.GLLinePlotItem()
        self.pose_canvas_points = gl.GLScatterPlotItem()
        self.pose_canvas.addItem(self.pose_canvas_lines)
        self.pose_canvas.addItem(self.pose_canvas_points)
        self.pose_canvas.setCameraParams(distance=config["pose3d_view_dist"])
        self.pose_canvas.setCameraParams(azimuth=config["pose3d_view_azi"])
        self.pose_canvas.setCameraParams(elevation=config["pose3d_view_ele"])
        
        self.pose_topology = pose3d_topology # should come from config
        
        self.keypoints = None
        self.keypoint_scores = None
        self.pose_edges = None

    def updateKeypoints(self, keypoints, keypoint_scores):
        self.keypoints = keypoints
        self.keypoint_scores = keypoint_scores
        
        #print("Pose3DCanas self.keypoints s ", self.keypoints.shape)
    
        pose_count = self.keypoints.shape[0]
        edge_count = self.pose_topology.shape[0]
        
        #print("keypoints min ", np.min(keypoints.reshape(-1, 3), axis=0), " max ", np.max(keypoints.reshape(-1, 3), axis=0))
        
        pose_min_posX = -100.0
        pose_max_posX = 100.0
        pose_posX_offset = (pose_max_posX - pose_min_posX) / (pose_count + 1)
        pose_posZ_offset = -50
        
        self.pose_edges = np.zeros([pose_count, edge_count, 2, 3])
        
        for pI, (keypoints, scores) in enumerate(zip(self.keypoints, self.keypoint_scores)):
            
            render_scale = 100.0
            keypoints_scaled = keypoints * render_scale
            #print("keypoints_scaled s ", keypoints_scaled.shape)
            
            keypoints_scaled[:, 0] += pose_min_posX + pose_posX_offset * (pI + 1)
            keypoints_scaled[:, 2] += pose_posZ_offset
            
            for eI in range(edge_count):
                
                point1 = keypoints_scaled[self.pose_topology[eI][0]]
                point2 = keypoints_scaled[self.pose_topology[eI][1]]
                
                self.pose_edges[pI, eI, 0] = point1
                self.pose_edges[pI, eI, 1] = point2
                
        #print("self.pose_edges ", self.pose_edges)
    
    def update(self):
        
        if self.pose_edges is None:
            return
        
        lines_data = self.pose_edges.reshape(-1, 3)
        self.pose_canvas_lines.setData(pos=lines_data, mode="lines", color=(1.0, 0.0, 0.0, 0.5), width=1.5)
            
        
class MotionGui(QtWidgets.QWidget):
    
    def __init__(self, config):
        super().__init__()
        
        self.setFixedWidth(1000)
        self.setFixedHeight(410)
        
        self.motion_model = config["motion_model"]
        self.sender = config["sender"]
        
        self.pose2d_canvas = Pose2DCanvas(config)
        self.pose3d_canvas = Pose3DCanas(config)
        
        self.pose2d_canvas.setFixedWidth(640)
        self.pose2d_canvas.setFixedHeight(360)
        
        self.pose3d_canvas.pose_canvas.setFixedWidth(360)
        self.pose3d_canvas.pose_canvas.setFixedHeight(360)
        
        self.input_image = None
        self.pose2d_keypoints = None
        self.pose3d_keypoints = None
        self.pose2d_keypoints_visible = None
        self.pose3d_keypoint_scores = None
        
        """
        self.synthesis = config["synthesis"]
        self.sender = config["sender"]
        
        self.edges = self.synthesis.edge_list
        """
        
        self.pose_thread_interval = config["update_interval"]
        
        """
        self.view_min = config["view_min"]
        self.view_max = config["view_max"]
        self.view_ele = config["view_ele"]
        self.view_azi = config["view_azi"]
        self.view_dist = config["view_dist"]
        self.view_line_width = config["view_line_width"]
        
        # dynamic canvas
        self.pose_canvas = gl.GLViewWidget()
        self.pose_canvas_lines = gl.GLLinePlotItem()
        self.pose_canvas_points = gl.GLScatterPlotItem()
        self.pose_canvas.addItem(self.pose_canvas_lines)
        self.pose_canvas.addItem(self.pose_canvas_points)
        self.pose_canvas.setCameraParams(distance=self.view_dist)
        self.pose_canvas.setCameraParams(azimuth=self.view_azi)
        self.pose_canvas.setCameraParams(elevation=self.view_ele)
        """

        self.q_start_buttom = QtWidgets.QPushButton("start", self)
        self.q_start_buttom.clicked.connect(self.start)  
        
        self.q_stop_buttom = QtWidgets.QPushButton("stop", self)
        self.q_stop_buttom.clicked.connect(self.stop)  
        self.q_button_spacer = QtWidgets.QSpacerItem(0, 10, QtWidgets.QSizePolicy.MinimumExpanding)
        self.q_fps_label =QtWidgets.QLabel()
        
        self.q_start_buttom.setFixedWidth(80)
        self.q_stop_buttom.setFixedWidth(80)
        
        self.q_button_grid = QtWidgets.QGridLayout()
        self.q_button_grid.addWidget(self.q_start_buttom,0,0)
        self.q_button_grid.addWidget(self.q_stop_buttom,0,1)
        self.q_button_grid.addItem(self.q_button_spacer,0,2)
        self.q_button_grid.addWidget(self.q_fps_label,0,3)
        
        self.q_pose_grid = QtWidgets.QGridLayout()
        self.q_pose_grid.addWidget(self.pose2d_canvas,0,0)
        self.q_pose_grid.addWidget(self.pose3d_canvas.pose_canvas,0,1)

        self.q_grid = QtWidgets.QGridLayout()
        #self.q_grid.addWidget(self.pose_canvas,0,0)
        #self.q_grid.addLayout(self.q_button_grid,1,0)
        
        #self.q_grid.addWidget(self.pose2d_canvas,0,0)
        
        self.q_grid.addLayout(self.q_pose_grid,0,0)
        self.q_grid.addLayout(self.q_button_grid,1,0)
        
        self.q_grid.setRowStretch(0, 0)
        self.q_grid.setRowStretch(1, 0)
        
        self.setLayout(self.q_grid)
        
        self.setGeometry(50,50,512,612)
        self.setWindowTitle("Motion Capture")
    
    def start(self):
        self.pose_thread_event = Event()
        self.pose_thread = Thread(target = self.update)
        
        self.pose_thread.start()
        
    def stop(self):
        self.pose_thread_event.set()
        self.pose_thread.join()
    
    def update(self):
        while self.pose_thread_event.is_set() == False:

            start_time = time.time()  
            
            self.update_motion()
            self.update_display()

            end_time = time.time()   
            time_diff = end_time - start_time
            time_fps = 1.0 / time_diff
            self.q_fps_label.setText("fps {:10.2f}".format(time_fps))
            
            #print("update time ", end_time - start_time, " interval ", self.pose_thread_interval)
            
            self.update_osc()
            
            next_update_interval = max(self.pose_thread_interval - (end_time - start_time), 0.0)
            
            sleep(next_update_interval)

    def update_motion(self):
        
        self.motion_model.update()
        
        self.input_image = self.motion_model.results["image"]
        pose2d_results = self.motion_model.results["pose2d_results"]
        pose3d_results = self.motion_model.results["pose3d_results"]
        
        if pose2d_results is not None:
        
            pose2d_pred_instances = pose2d_results.pred_instances
            self.pose2d_keypoints = pose2d_pred_instances.keypoints
            self.pose2d_keypoints_visible = pose2d_pred_instances.keypoints_visible
        
        if pose3d_results is not None:
            
            pose3d_pred_instances = pose3d_results.pred_instances
            self.pose3d_keypoints = pose3d_pred_instances.keypoints
            self.pose3d_keypoint_scores = pose3d_pred_instances.keypoint_scores
        
    def update_display(self):
        
        if self.pose2d_keypoints is not None:
            self.pose2d_canvas.updateImage(self.input_image)
            self.pose2d_canvas.updateKeypoints(self.pose2d_keypoints, self.pose2d_keypoints_visible)
            self.pose2d_canvas.update()

        if self.pose3d_keypoints is not None:
            
            self.pose3d_canvas.updateKeypoints(self.pose3d_keypoints, self.pose3d_keypoint_scores)
            self.pose3d_canvas.update()
        

    def update_osc(self):
        
        if self.pose2d_keypoints is not None:
            
            pose_count = self.pose2d_keypoints.shape[0]
            
            for pI in range(pose_count):
                
                keypoints = self.pose2d_keypoints[pI]
                visibility = self.pose2d_keypoints_visible[pI]
                
                self.sender.send(f"/mocap/{pI}/joint/pos2d_world", keypoints)
                self.sender.send(f"/mocap/{pI}/joint/visibility", visibility)

        if self.pose3d_keypoints is not None:
            
            pose_count = self.pose3d_keypoints.shape[0]
            
            for pI in range(pose_count):
                
                keypoints = self.pose3d_keypoints[pI]
                scores = self.pose3d_keypoint_scores[pI]
                
                self.sender.send(f"/mocap/{pI}/joint/pos3d_world", keypoints)
                self.sender.send(f"/mocap/{pI}/joint/scores", scores)
    
    """
    def update_pred_seq(self):
        
        self.synthesis.update()       
        self.synth_pose_wpos = self.synthesis.synth_pose_wpos
        self.synth_pose_wrot = self.synthesis.synth_pose_wrot
    """
        
    """
    def update_osc(self):
        
        # convert from left handed bvh coordinate system to right handed standard coordinate system
        self.synth_pose_wpos_rh = np.copy(self.synth_pose_wpos)

        self.synth_pose_wpos_rh[:, 0] = self.synth_pose_wpos[:, 0] / 100.0
        self.synth_pose_wpos_rh[:, 1] = -self.synth_pose_wpos[:, 2] / 100.0
        self.synth_pose_wpos_rh[:, 2] = self.synth_pose_wpos[:, 1] / 100.0

        self.synth_pose_wrot_rh = np.copy(self.synth_pose_wrot)
        
        self.synth_pose_wrot_rh[:, 1] = self.synth_pose_wrot[:, 1]
        self.synth_pose_wrot_rh[:, 2] = -self.synth_pose_wrot[:, 3]
        self.synth_pose_wrot_rh[:, 3] = self.synth_pose_wrot[:, 2]

        
        self.sender.send("/mocap/joint/pos_world", self.synth_pose_wpos_rh)
        self.sender.send("/mocap/joint/rot_world", self.synth_pose_wrot_rh)
    """

    """
    def update_seq_plot(self):
        
        pose = self.synth_pose_wpos

        points_data = pose
        lines_data = pose[np.array(self.edges).flatten()]
        
        self.pose_canvas_lines.setData(pos=lines_data, mode="lines", color=(1.0, 1.0, 1.0, 0.5), width=self.view_line_width)
        #self.pose_canvas_lines.setData(pos=lines_data, mode="lines", color=(0.0, 0.0, 0.0, 1.0), width=self.view_line_width)
        #self.pose_canvas_points.setData(pos=pose, color=(1.0, 1.0, 1.0, 1.0))

        #self.pose_canvas.show()
        
        #print(self.pose_canvas.cameraParams())
    """