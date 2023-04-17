'''create-HOTA.py generates a set of formatted tracking labels that can be evaluated
with Higher Order Tracking Accuracy (HOTA)
Once the appropriate directory had been generated, HOTA can be evaluated by using the TrackEval
kit included in the root directory'''
#HOTA data has already been created and can be found in:
#.\TrackEval\data\trackers\mot_challenge\MOT17-train

import torch
from ultralytics import YOLO
from ultralytics import nn
import numpy as np
import os

#The following block of code is only used for weight averaging between models (if required)
baseline = YOLO('path\to\baseline\model.pt').model.state_dict()
mixed = YOLO('path\to\mixed\model.pt').model.state_dict()
result = YOLO('yolov8n.pt').model.state_dict()

for key in baseline:
    result[key] = (baseline[key]+mixed[key])/2.0

model = YOLO('F:\CMPT828Project\\runs\detect\\baseline-large-model\weights\last.pt')
model.reset_weights()
model.model.load_state_dict(result)

#if weight averaging is not required, run the line below
#model = YOLO('path\to\model.pt)

path_to_vids = '.\Project_Data\MOT17 Train'
for vid in os.listdir(path_to_vids):
    video_path = os.path.join(path_to_vids,vid)
    sequence = model.track(video_path, show=False, tracker='bytetrack.yaml')
    results = []
    for index, frame_results in enumerate(sequence):
        for detection in frame_results:
            #print(vars(detection))
            #print(detection)
            if detection.boxes.is_track == True:
                box_id = int(detection.boxes.id.tolist()[0])
                box_xyxy = detection.boxes.xyxy.tolist()[0]
                box_xywh = detection.boxes.xywh.tolist()[0]
                result = [index+1, box_id, box_xyxy[0], box_xyxy[1], box_xywh[2], box_xywh[3], -1, -1, -1, -1]
                results.append(result)
        label_target_path1 = os.path.join(path_to_vids,vid.split('SDP-raw')[0]+'DPM'+'.txt')
        label_target_path2 = os.path.join(path_to_vids,vid.split('SDP-raw')[0]+'FRCNN'+'.txt')
        label_target_path3 = os.path.join(path_to_vids,vid.split('SDP-raw')[0]+'SDP'+'.txt')
        if results:
            np.savetxt(label_target_path1,results,fmt='%.2f', delimiter=' ')
            np.savetxt(label_target_path2,results,fmt='%.2f', delimiter=' ')
            np.savetxt(label_target_path3,results,fmt='%.2f', delimiter=' ')