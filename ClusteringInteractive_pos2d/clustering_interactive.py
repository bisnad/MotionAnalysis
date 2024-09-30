import motion_analysis
import motion_model
import motion_synthesis
import motion_sender
import motion_gui
import motion_control


import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch import nn
from collections import OrderedDict
import networkx as nx
import scipy.linalg as sclinalg
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.cluster import DBSCAN

import os, sys, time, subprocess
import numpy as np
import math

from common import utils
from common import bvh_tools as bvh
from common import fbx_tools as fbx
from common import mocap_tools as mocap
from common.quaternion import qmul, qrot, qnormalize_np, slerp, qfix
#from common.quaternion_np import slerp
from common.pose_renderer import PoseRenderer


from matplotlib import pyplot as plt
import numpy as np
import pickle

"""
Mocap Settings
"""

mocap_file_path = "C:/Users/dbisig/Projects/Premiere/Software_Git2/MotionUtilities/MocapRecorder/recordings"
mocap_files = ["CCL_SaoPaulo_2024-09-10-16-27-19_pose2d.pkl"]
mocap_sensor_id = "/mocap/0/joint/pos2d_world"
mocap_joint_count = 17
mocap_joint_dim = 2

mocap_joint_children = [
    [1, 2],
    [3],
    [4],
    [5],
    [6],
    [7, 11],
    [8, 12],
    [9],
    [10],
    [],
    [],
    [13],
    [14],
    [15],
    [16],
    [],
    []
    ]

# joint weight percentages for COCO data
mocap_joint_weight_percentages = [
    2.28,
    1.14,
    1.14,
    1.52,
    1.52,
    23.8,
    23.8,
    3,
    1.8,
    0.7,
    3,
    1.8,
    0.7,
    11.5,
    5.4,
    11.5,
    5.4
    ]

mocap_body_weight = 60

mocap_seq_window_length = 48 # 8
mocap_seq_window_overlap = 24 # 4

"""
Load Mocap Data
"""


all_mocap_data = []

for mocap_file in mocap_files:
    
    mocap_data = {}
    mocap_data["skeleton"] = {}
    mocap_data["motion"] = {}
    
    with open(mocap_file_path + "/" + mocap_file, "rb") as input_file:
        file_data = pickle.load(input_file)

    sensor_indices = [i for i, x in enumerate(file_data["sensor_ids"]) if x == mocap_sensor_id]
    sensor_values = [file_data["sensor_values"][i] for i in sensor_indices]
    sensor_values = np.array(sensor_values)
    sensor_values = np.reshape(sensor_values, (-1, mocap_joint_count, mocap_joint_dim))
    
    mocap_data["skeleton"]["children"] = mocap_joint_children
    mocap_data["motion"]["pos_world"] = sensor_values
    
    all_mocap_data.append(mocap_data)

# retrieve mocap properties

mocap_data = all_mocap_data[0]

joint_count = mocap_joint_count
joint_dim = mocap_joint_dim
pose_dim = joint_count * joint_dim

mocap_joint_weight_percentages = np.array(mocap_joint_weight_percentages)
mocap_joint_weight_percentages_total = np.sum(mocap_joint_weight_percentages)
joint_weights = mocap_joint_weight_percentages * mocap_body_weight / 100.0



"""
Mocap Analysis
"""

mocap_data["motion"]["pos_world_m"] = mocap_data["motion"]["pos_world"] * 10.0

mocap_data["motion"]["pos_world_smooth"] = motion_analysis.smooth(mocap_data["motion"]["pos_world_m"], 25)
mocap_data["motion"]["pos_scalar"] = motion_analysis.scalar(mocap_data["motion"]["pos_world_smooth"], "norm")
mocap_data["motion"]["vel_world"] = motion_analysis.derivative(mocap_data["motion"]["pos_world_smooth"], 1.0 / 50.0)
mocap_data["motion"]["vel_world_smooth"] = motion_analysis.smooth(mocap_data["motion"]["vel_world"], 25)
mocap_data["motion"]["vel_world_scalar"] = motion_analysis.scalar(mocap_data["motion"]["vel_world_smooth"], "norm")
mocap_data["motion"]["accel_world"] = motion_analysis.derivative(mocap_data["motion"]["vel_world_smooth"], 1.0 / 50.0)
mocap_data["motion"]["accel_world_smooth"] = motion_analysis.smooth(mocap_data["motion"]["accel_world"], 25)
mocap_data["motion"]["accel_world_scalar"] = motion_analysis.scalar(mocap_data["motion"]["accel_world_smooth"], "norm")
mocap_data["motion"]["jerk_world"] = motion_analysis.derivative(mocap_data["motion"]["accel_world_smooth"], 1.0 / 50.0)
mocap_data["motion"]["jerk_world_smooth"] = motion_analysis.smooth(mocap_data["motion"]["jerk_world"], 25)
mocap_data["motion"]["jerk_world_scalar"] = motion_analysis.scalar(mocap_data["motion"]["jerk_world_smooth"], "norm")
mocap_data["motion"]["qom"] = motion_analysis.quantity_of_motion(mocap_data["motion"]["vel_world_scalar"], joint_weights)
mocap_data["motion"]["bbox"] = motion_analysis.bounding_box(mocap_data["motion"]["pos_world_m"])
mocap_data["motion"]["bsphere"] = motion_analysis.bounding_sphere(mocap_data["motion"]["pos_world_m"])
mocap_data["motion"]["weight_effort"] = motion_analysis.weight_effort(mocap_data["motion"]["vel_world_scalar"], joint_weights, 25)
mocap_data["motion"]["space_effort"] = motion_analysis.space_effort_v2(mocap_data["motion"]["pos_world_m"], joint_weights, 25)
mocap_data["motion"]["time_effort"] = motion_analysis.time_effort(mocap_data["motion"]["accel_world_scalar"], joint_weights, 25)
mocap_data["motion"]["flow_effort"] = motion_analysis.flow_effort(mocap_data["motion"]["jerk_world_scalar"], joint_weights, 25)

"""
Load Model
"""

mocap_features = {"qom": mocap_data["motion"]["qom"],
                  "bsphere": mocap_data["motion"]["bsphere"], 
                  "weight_effort": mocap_data["motion"]["weight_effort"],
                  "space_effort": mocap_data["motion"]["space_effort"],
                  "time_effort": mocap_data["motion"]["time_effort"],
                  "flow_effort": mocap_data["motion"]["flow_effort"]}

motion_model.config = {
    "mocap_data": mocap_data["motion"]["pos_world"],
    "features_data": mocap_features,
    "mocap_window_length": mocap_seq_window_length,
    "mocap_window_offset": mocap_seq_window_overlap,
    "cluster_method": "kmeans",
    "cluster_count": 20,
    "cluster_random_state": 170
    }

clustering = motion_model.createModel(motion_model.config) 

"""
Setup Motion Synthesis
"""


motion_synthesis.config = {"skeleton": mocap_data["skeleton"],
          "model": clustering,
          "seq_window_length": mocap_seq_window_length,
          "seq_window_overlap": mocap_seq_window_overlap
          }

synthesis = motion_synthesis.MotionSynthesis(motion_synthesis.config)

"""
OSC Sender
"""

motion_sender.config["ip"] = "127.0.0.1"
motion_sender.config["port"] = 9004

osc_sender = motion_sender.OscSender(motion_sender.config)


"""
GUI
"""

from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from pathlib import Path

motion_gui.config["synthesis"] = synthesis
motion_gui.config["sender"] = osc_sender

app = QtWidgets.QApplication(sys.argv)
gui = motion_gui.MotionGui(motion_gui.config)

# set close event
def closeEvent():
    QtWidgets.QApplication.quit()
app.lastWindowClosed.connect(closeEvent) # myExitHandler is a callable


"""
OSC Control
"""


motion_control.config["synthesis"] = synthesis
motion_control.config["model"] = clustering
motion_control.config["gui"] = gui
motion_control.config["ip"] = "127.0.0.1"
motion_control.config["port"] = 9002

osc_control = motion_control.MotionControl(motion_control.config)

"""
Start Application
"""

osc_control.start()
gui.show()
app.exec_()

osc_control.stop()
