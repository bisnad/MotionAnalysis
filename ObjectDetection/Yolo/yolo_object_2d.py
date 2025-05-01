from ultralytics import YOLO
import cv2
import numpy as np

import motion_sender

"""
OSC Settings
"""

motion_sender.config["ip"] = "127.0.0.1"
motion_sender.config["port"] = 9004
osc_sender = motion_sender.OscSender(motion_sender.config)

def on_predict_batch_end(predictor):


    if len(predictor.results) == 0: 
        return
    
    results = predictor.results[0]
    
    class_names = results.names
    
    #print("class_names ", class_names)
    
    boxes = results.boxes
    
    boxes_cls = boxes.cls.cpu().numpy() # class
    boxes_conf = boxes.conf.cpu().numpy() # confidence
    boxes_xyxyn = boxes.xyxyn.cpu().numpy() # normalised bounding box

    """
    print("boxes_cls ", boxes_cls)
    print("boxes_conf ", boxes_conf)
    print("boxes_xyxyn ", boxes_xyxyn) 
    """
    
    for box_index in range(boxes_xyxyn.shape[0]):
        
        osc_parameters = np.concatenate(( boxes_cls[box_index:box_index+1], boxes_conf[box_index:box_index+1], boxes_xyxyn[box_index]), axis=0)
        osc_sender.send("/object/box/{}/bbox".format(box_index), osc_parameters)

        osc_parameters = class_names[box_index]
        osc_sender.send("/object/box/{}/class".format(box_index), osc_parameters)



# load a pretrained YOLOv8m model
model = YOLO("yolov8x.pt")

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