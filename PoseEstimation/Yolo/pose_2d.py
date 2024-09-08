from ultralytics import YOLO
import cv2
import numpy as np

import motion_sender

motion_sender.config["ip"] = "127.0.0.1"
motion_sender.config["port"] = 9004
osc_sender = motion_sender.OscSender(motion_sender.config)

def on_predict_batch_end(predictor):

    if len(predictor.results) == 0: 
        return
    
    results = predictor.results[0]

    boxes = results.boxes
    probs = results.probs
    keypoints = results.keypoints

    keypoints_xyn = keypoints.xyn
    keypoints_np = keypoints_xyn.cpu().numpy()

    for skel_index in range(keypoints_np.shape[0]):

        osc_sender.send("/mocap/skel{}/joint/pos_world".format(skel_index), keypoints_np[skel_index])


# load a pretrained YOLOv8m model
model = YOLO("yolov8x-pose.pt")

# Add the custom callback to the model
model.add_callback("on_predict_batch_end", on_predict_batch_end)

"""
# Run inference from image
im2 = cv2.imread("dance2.jpg")
results = model.predict(source=im2, save=True, save_txt=True)  # save predictions as labels

#print("results ", results)
"""

# Run inference from webcam
#results = model(source=0, show=True, conf=0.3, save=True)
results = model(source=0, show=True, conf=0.3)