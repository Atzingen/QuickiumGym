# QuickiumGym - assistance and management of gym exercises

This project involves a solution developed by the Quickium team for the [OpenCV AI Competition 2021](https://opencv.org/opencv-ai-competition-2021/) in the “Health and fitness area”.

The solution is a computer vision system for assistance and management of gym exercises. The system integrates with the gym network, having access to all user training plans, data and history. Through an Artificial Intelligence model, the system is able to detect some key points (Human Pose Estimation) at the user body during exercises moviment, and with these points the software can infer wrong executed movements using the angles between body parts, amplitude of movement, inclination, and other factors that can be informed by physical professionals. 

On a totem tha will be in front of the exercite area, the user is recognized by their face, get the training parameters, do the exercises while a software logic monitor their movements and at the end the user gets their score.

More information on the submission [video](https://www.youtube.com/watch?v=FV9vyA8N8rk)


## About this repository

This repository contais the pose recognition, exercise feature extraction [this subfolder](face_recognition/) and face identification and recognition [this subfolder](scores/) and works as a stand alone demonstration using the OAK-D camera with blazepose 33 body points model. 

Code developed for the totem interface and [API](https://apigym.pages.quickium.com/docs) is not shared and will not be opensourced. 

## Instalation

This code was tested under Python 3.8 and using the OAK-D. Necessary libraries can be installed using pip and the requeriments.txt file on both subfolders.



