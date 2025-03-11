from argparse import ArgumentParser

from ultralytics import YOLO
import torch
import cv2
import numpy as np

import motion_sender

"""
Compute Device
"""

device = 'cpu'
if torch.cuda.is_available():
    device="cuda"
elif torch.backends.mps.is_available():
    device="mps"

print('Using {} device'.format(device))

"""
OSC Settings
"""

motion_sender.config["ip"] = "127.0.0.1"
motion_sender.config["port"] = 9007
osc_sender = motion_sender.OscSender(motion_sender.config)

"""
Parse arguments
"""

def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        'inputs',
        type=str,
        nargs='?',
        help='Input image/video path or webcam.')

    parser.add_argument(
        '--pose2d',
        type=str,
        default='yolov8x-pose.pt',
        help='Pretrained 2D pose estimation algorithm')

    args, _ = parser.parse_known_args()

    return args.inputs, args.pose2d

input_args, model_args = parse_args()

print("input_args ", input_args)
print("model_args ", model_args)

"""
Pose Estimation and OSC Sending
"""

def on_predict_batch_end(predictor):

    if len(predictor.results) == 0: 
        return
    
    results = predictor.results[0]

    boxes = results.boxes
    probs = results.probs
    keypoints = results.keypoints

    keypoints_xyn = keypoints.xyn
    keypoints_np = keypoints_xyn.cpu().numpy()

    keypoints_conf = keypoints.conf
    keypoints_conf_np = keypoints_conf.cpu().numpy()

    for skel_index in range(keypoints_np.shape[0]):

        osc_sender.send("/mocap/{}/joint/pos_world".format(skel_index), keypoints_np[skel_index])
        osc_sender.send("/mocap/{}/joint/visibility".format(skel_index), keypoints_conf_np[skel_index])


# load a pretrained YOLOv8m model
model = YOLO("models/{}".format(model_args)).to(device)

# Add the custom callback to the model
model.add_callback("on_predict_batch_end", on_predict_batch_end)


if input_args == "webcam":
    # Run inference from webcam
    print("Run inference from webcam")
    results = model(source=0, show=True, conf=0.3)
else:
    # Run inference from image or video
    print("Run inference from image or video")
    results = model(source=input_args, show=True, conf=0.3)
