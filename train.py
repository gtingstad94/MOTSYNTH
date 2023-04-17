'''train.py trains a model. Some of the training settings are also
specified in the custom.yaml file'''

import torch
from ultralytics import YOLO

if __name__ == '__main__':

    model = YOLO('path\to\model.pt')
    #Training a model
    results = model.train(
        data='path\to\custom.yaml',
        imgsz=640,
        patience=30,
        epochs=30,
        batch=16,
        optimizer='SGD',
        name='model_output',
        mixup = 0.1,
        #dropout = 0.5,
        device=0
    )