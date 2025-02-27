"""
Things that are unclear at the moment:
    
How the travel distance is calculated (especially the mean, std, max, min), at the moment, I calculate only the mean
Same issue applies to the area covered

"""


"""
imports
"""

import analysis as ma
import motion_receiver
import motion_sender
import motion_pipeline
import motion_gui

import json
import sys

from matplotlib import pyplot as plt
import numpy as np

"""
Mocap Settings
"""

#mocap_joint_weights_path = "configs/joint_weights_xsens_fbx.json"
#mocap_joint_weights_path = "configs/joint_weights_captury_fbx.json"
#mocap_joint_weights_path = "configs/joint_weights_zed34_fbx.json"
#mocap_joint_weights_path = "configs/joint_weights_qualisys_hands_bvh.json"
#mocap_joint_weights_path = "configs/joint_weights_xsens_fbx.json"
#mocap_joint_weights_path = "configs/joint_weights_captury_fbx.json"
#mocap_joint_weights_path = "configs/joint_weights_zed34_fbx.json"
#mocap_joint_weights_path = "configs/joint_weights_qualisys_hands_bvh.json"
mocap_joint_weights_path = "configs/joint_weights_coco.json"
mocap_pos_dim = 3
mocap_fps = 50

# load joint weights 
if mocap_joint_weights_path is None: # 
    mocap_joint_count = 17 # COCO
    mocap_joint_weights = [1] * mocap_joint_count
else:
    with open(mocap_joint_weights_path) as json_data:
        mocap_joint_weights = json.load(json_data)["jointWeights"]
    mocap_joint_weights = np.array(mocap_joint_weights, dtype=np.float32)
    
    mocap_joint_count = len(mocap_joint_weights)

"""
OSC Receiver
"""

input_pos_data = np.zeros((mocap_joint_count, mocap_pos_dim), dtype=np.float32)

motion_receiver.config["ip"] = "0.0.0.0"
motion_receiver.config["port"] = 9007
motion_receiver.config["data"] = [ input_pos_data ]
motion_receiver.config["messages"] = ["/mocap/0/joint/pos_world"]

osc_receiver = motion_receiver.MotionReceiver(motion_receiver.config)

"""
OSC Sender
"""

#motion_sender.config["ip"] = "127.0.0.1"
motion_sender.config["ip"] = "192.168.0.249"
# motion_sender.config["port"] = 9008
motion_sender.config["port"] = 10000

osc_sender = motion_sender.OscSender(motion_sender.config)

"""
Data Pipeline
"""

pipeline = motion_pipeline.MotionPipeline(osc_receiver, mocap_joint_weights, 0.02)
pipeline.posScale = 1.0

"""
GUI
"""

from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from pathlib import Path

motion_gui.config["pipeline"] = pipeline
motion_gui.config["sender"] = osc_sender

app = QtWidgets.QApplication(sys.argv)
gui = motion_gui.MotionGui(motion_gui.config)


"""
Start Application
"""

osc_receiver.start()
gui.show()
app.exec_()

osc_receiver.stop()


