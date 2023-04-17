'''inference.py runs detections with no trackers. You may either do this with single images
or with video files. Options are given for both'''

import cv2
from ultralytics import YOLO

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


#This code runs inference on a single image
'''image_to_eval = '.\ProjectData\\test\images\MOT17-01-FRCNN\\00085.jpg'          #path to image
results = model(image_to_eval, show=True)'''


#Below is optional code for running inference on video frames

# Open the video file
video_path = '.\EvalData\MOT17 Train\MOT17-04-SDP-raw.webm'
cap = cv2.VideoCapture(video_path)

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 inference on the frame
        results = model(frame)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Display the annotated frame
        cv2.imshow("YOLOv8 Inference", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()