import sys
import os
sys.path.append(os.path.abspath(os.getcwd()))
sys.path.append(os.path.abspath(os.getcwd() + '/blazepose/'))
import cv2
from blazepose.BlazeposeDepthaiEdge import BlazeposeDepthai
from blazepose.blazeposescript import get_frame_body_points

# Load the cascade
face_cascade = cv2.CascadeClassifier('face_detection/haarcascade_frontalface_default.xml')
# Load DepthAI reqs
pose = BlazeposeDepthai()
#Main loop
while True:
    img, _, = get_frame_body_points(pose)
    # Convert into grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    # Draw rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    # Display the output
    cv2.imshow('img', img)
    if cv2.waitKey(1) == ord('q'):
        break
