'''create-pseudo.py uses a baseline synthetic/mixed model
to generate pseudolabels on MOT16 (real) data.
High confidence labels are then used to generate a "transfer set"
which includes only high confidence image-label pairs
'''

from ultralytics import YOLO
import numpy as np
import os
import shutil

trans_imgs = 'path\to\image output\root'
trans_labels = 'path\to\label output\root'

model = YOLO('path\to\model.pt')

mot16 = 'path\to\mot16\root'    #'.\CMPT828Project\Project_Data\MOT16'
for folder in [i for i in os.listdir(mot16) if os.path.isdir(os.path.join(mot16,i))]:
    #make required directories
    new_trans_imgs = os.path.join(trans_imgs,folder)
    new_trans_labels = os.path.join(trans_labels,folder)
    if not os.path.isdir(new_trans_imgs):
        os.mkdir(new_trans_imgs)
    if not os.path.isdir(new_trans_labels):
        os.mkdir(new_trans_labels)

    for img in [os.path.join(mot16,folder,'img1',i)
                 for i in os.listdir(os.path.join(mot16,folder,'img1'))
                 if os.path.join(mot16,folder,'img1',i).split('.')[-1] == 'jpg']:
        results = model(img)
        output = []
        img_target_path = os.path.join(new_trans_imgs,img.split('\\')[-1])
        label_target_path = os.path.join(new_trans_labels,img.split('\\')[-1].split('.')[0]+'.txt')
        boxes = results[0].boxes
        for box in boxes:
            #print(box.xywhn.tolist())
            #print(box.conf.tolist())
            if box.conf > 0.7:
                output.append([0]+box.xywhn.tolist()[0])
        if output:
            print(output)
            shutil.copyfile(img,img_target_path)
            np.savetxt(label_target_path,output,fmt='%.8f', delimiter=' ')