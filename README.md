# Face_Recognition
 Image_Processing

Using Python 
libraries: OpenCV, Haarcascade,  os, numpy, PIL.
OpenC2 for image processing, Haarcascade Frontal Face for detecting object in video stream (FaceDetector) , os for file path,  numpy for matrix calculation, Python Image Library (PIL).

Idea: There are  3 files Dataset, training and  recognition.
All photos turned into grayscale to be manipulated.
Dataset file: Open camera then take multiple photos  to work with and get the features of them.
Training file: Work with the dataset file by getting features from them and describe them at train.yml file that contain all desired features.
Recognition file: Open camera again and take live photos then compare their features with the train.yml file if it match to make a specific decison 
if it doesn't make another decision.
