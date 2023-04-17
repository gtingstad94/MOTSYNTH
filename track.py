'''track.py loads a pytorch or YOLO model and runs detections on frames in a video
using ByteTrack'''
import torch
from ultralytics import YOLO
from ultralytics import nn
import numpy as np
import inspect

#Baseline and Mixed models are loaded
#Load baseline
baseline = YOLO('.\Final Models\\baseline-large-model\weights\last.pt').model.state_dict()
#Load Mixed
mixed = YOLO('.\Final Models\\mixed-large-model\weights\last.pt').model.state_dict()

#load dummy model and reset weights
model = YOLO('.\Final Models\\baseline-large-model\weights\last.pt')
model.reset_weights()
averaged = model.model.state_dict()
#Average weights to produce new model
for key in averaged:
    averaged[key] = (baseline[key]+mixed[key])/2.0

#Reload adjusted weights into model
model.model.load_state_dict(averaged)

#Path to video for tracking
video_path = '.\EvalData\MOT17 Train\MOT17-04-SDP-raw.webm'
#Run tracking. replace "model" with custom model if required
sequence = model.track(video_path, show=True, conf=0.3, iou=0.5, tracker='bytetrack.yaml')