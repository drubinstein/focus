#!/bin/python2

import numpy as np
import cv2

cap = cv2.VideoCapture(0) #We will be capturing from the camera with id 0

while(True):
    #Capture frame by frame
    ret, frame = cap.read()

    # Operate on the frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #Display the resulting frame
    cv2.imshow('GrayFrame',gray)
    cv2.imshow('ColorFrame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
	break

#When everything is done, end the capture
cap.release()
cap.destroyAllWindows()
