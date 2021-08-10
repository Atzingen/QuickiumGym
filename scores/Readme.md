This project involves a solution developed by the Quickium team for the OpenCV AI Competition 2021 in the “Health and fitness area”, 
the solution is a computer vision system for assistance and management of gym exercises. 

This directory is a small part of the project, the focus is to provide a example of the score metrics involved in the solution
The project uses depthai_blazepose (https://github.com/geaxgx/depthai_blazepose) as base to determine human pose estimation
The additional files are:
    quickium_solution.py: A python script that get the frames from the OAK-D camera and shows the images beside a matplotlib graph plotting the height wrist of the person in the frame
    calc_score.py: A python script containing all auxiliary functions to calculate the user scores 
    requirements.txt: Containing python modules dependencies

Just execute quickium_solution.py to test the solution
