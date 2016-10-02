#!/usr/bin/python
#start from docker instance (run_docker.sh)

import openface
import dlib
import os, cv2
import numpy as np
import time

fileDir = os.path.dirname(os.path.realpath(__file__))
modelDir = os.path.join(fileDir, '..', 'openface/models')
dlibModelDir = os.path.join(modelDir, 'dlib')
openfaceModelDir = os.path.join(modelDir, 'openface')
os.path.join(dlibModelDir,"shape_predictor_68_face_landmarks.dat")

cap = cv2.VideoCapture(0) #We will be capturing from the camera with id 0

print dlibModelDir
alib = openface.AlignDlib(dlibModelDir + "/shape_predictor_68_face_landmarks.dat")

while(True):
    #Capture frame by frame
    ret, frame = cap.read()

    face_box = alib.getLargestFaceBoundingBox(frame)
    #Display the resulting frame

    if face_box is not None:
        top_left = (face_box.left(), face_box.top())
        bot_right = (face_box.right(), face_box.bottom())
        aligned_frame = alib.align(96, frame)

        cv2.rectangle(frame, top_left, bot_right, (0,255,0), 3)

    cv2.imshow('Hello', frame)
    cv2.imshow('aligned', aligned_frame)



    if cv2.waitKey(1) & 0xFF == ord('q'):
        break



