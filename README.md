# CMPT828-CourseProject
Note: This project was developed on a Windows machine. Path formatting may be different than your machine.
# Dependencies
Required dependencies:
Ultralytics: Use pip install ultralytics to install the ultralytics environemnt and any of its dependencies.

HOTA evaluations:
Minimum requirements for HOTA evaluations using TrackEval include:
scipy==1.4.1
numpy==1.18.1

# Included Files
The project repository includes all models, model results and some of the data used during the project. See a description of each subdirectory below:

./Final Models: Includes data for all of the models created during the project. Trainig performance curves, training and validation examples, and arguments used to produce the model are included in these folders. The models can be loaded into pyTorch or ultralytics from ./Final Models/model-name/weights/last.pt.

./ProjectData: includes sample data for training, validation, and testing. Demonstrates required folder structure for training with ultralytics. custom.yaml specifies training parameters for ultralytics models.

./EvalData: Includes a single video file from the MOT17 dataset which may be used to run a tracking example with track.py in the root directory

./TrackEval: The evaluation kit used for calculating HOTA. The MOT17 training dataset was used as an evaluation baseline. The evaluation files were created with create-HOTA.py, and are stored in ./TrackEval/data/trackers/mot_challenge/MOT17-train. Due to the number of video files needed to generate the HOTA evaluation data, all videos are not included in this repo.

./: The root directory contains scripts used to train and evaluate my models.
Custom model architectures are also contained here, which were used for the ablation study. 'yolov8n-default.yaml' was used to generate an ablation groundtruth model, and 'yolov8n-custom.yaml' generates the ablation model with fewer channels in the backbone of the model. 

# Scripts
For reference, the following scripts are included in the repository:
train.py - used for training ultralytics models
track.py - uses an ultralytics model with ByteTrack on a video sequence
create-pseudo.py - generates pseudolabels and creates a high-confidence set for transfer learning
mixup.py - takes high-conf dataset from create-pseudo.py and creates mixed synth/real dataset for transfer learning
inference.py - used to run predictions on image or video
create-HOTA.py - used to create the evaluation files for HOTA. This script uses ByteTrack with a model to generate annotations on video files. These annotations are then compared with ground-truth annotations to evaluate the HOTA score. 

# Demo
Since most of my functions use very large datasets, the data required to run them could not be included in this repository; however, I have included the main scripts I used along with comments and pseudocode. I have also included one sample video, which can be used to visually inspect the tracking performance of the weight-averaged model created during my main experiment. To see this output, simply run track.py with python.

The main metric used to evaluate performance in my experiment was HOTA. I have pre-generated the HOTA evaluation files for each model on the MOT17 training set, and you will be able to run the evaluation by entering the following code in your terminal:
python TrackEval/scripts/run_mot_challenge.py --BENCHMARK MOT17 --SPLIT_TO_EVAL train --TRACKERS_TO_EVAL AveragedModels --METRICS HOTA --USE_PARALLEL False --NUM_PARALLEL_CORES 1

Note that the --TRACKERS_TO_EVAL argument specifies the model you would like to test. You can replace this variable with any of the folder names in the ./TrackEval/data/trackers/mot_challenge/MOT17-train folder.
