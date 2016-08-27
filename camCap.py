#!/bin/python2

import numpy as np
from scipy import misc
import cv2
import math

cap = cv2.VideoCapture(0) #We will be capturing from the camera with id 0

square_size = 96

while(True):
    #Capture frame by frame
    ret, frame = cap.read()

    # Operate on the frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #create 96x96 pixels from the center
    y_size, x_size = gray.shape
    #cut off the left and right sides (cause wide screen is stupid)
    gray_downsampled = gray[:,range(x_size/2-y_size/2,x_size/2+y_size/2)]
    gray_downsampled = misc.imresize(gray_downsampled,(square_size, square_size))

    #Display the resulting frame
    cv2.imshow('Gray', gray)
    cv2.imshow('GrayFrame',gray_downsampled)
    if cv2.waitKey(1) & 0xFF == ord('q'):
	break

#When everything is done, end the capture
cap.release()
cap.destroyAllWindows()
