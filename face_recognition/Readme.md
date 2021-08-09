Face Recognition is a directory that contains the respective scripts and functions:
    haarcascade_frontalface_default.xml:  OpenCV Cascade Classifier used by auxiliar functions to detect faces in images
    compare_faces.py: A python script example to send 2 images faces to Quickium Gym API and get the percent match of their
    face_recognition.py: A python script containing all auxiliar functios to compare a received face to faces presents in true_faces folders
    true_faces: A directory that must contain folders with people name and their face pictures (jpg format) inside. 