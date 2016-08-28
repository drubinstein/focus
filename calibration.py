#file calibration.py
import os
import cv2
import numpy as np
import Tkinter as tk
import time
from random import randint

from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
import matplotlib.pyplot as pyplot

def main():
    root = tk.Tk()

#use 1/2 screen width because I personally use dual monitors
    screen_width = root.winfo_screenwidth()/4
    screen_height = root.winfo_screenheight()/2

    img = np.zeros((screen_height, screen_width,3), np.uint8)
    print screen_width, screen_height
    fs = 100
#TODO: Instead of looping through every w and h instead choose 25 random points and
#get captures for those with half-second delays

    print('starting video')
    cap = cv2.VideoCapture(0)
    #get video resolution
    ret, frame = cap.read()
    print frame.shape
    frame_h, frame_w, _ = frame.shape
    print "Frame dims are '{0}' x '{1}'".format(frame_w,frame_h)

    print('Initializing neural net')
    net1 = NeuralNet(
        layers=[  # three layers: two hidden layers
            ('input', layers.InputLayer),
            ('hidden', layers.DenseLayer),
            ('hidden', layers.DenseLayer),
            ('output', layers.DenseLayer),
            ],
        # layer parameters:
        input_shape=(None, frame_h*frame_w),
        hidden_num_units=100,  # number of units in hidden layer
        output_nonlinearity=None,  # output layer uses identity function
        output_num_units=2,  #  target values

        # optimization method:
        update=nesterov_momentum,
        update_learning_rate=0.01,
        update_momentum=0.9,

        regression=True,  # flag to indicate we're dealing with regression problem
        max_epochs=400,  # we want to train this many epochs
        verbose=1,
        )

    print('generating 25 random points to choose from')
    npts = 25
    for pt in xrange(0,25):
        img[:] = (0,0,0) # clear

        x = randint(0,screen_width-1)
        y = randint(0,screen_height-1)
        print x,y

        target = np.array([[np.float32(x), np.float32(y)]])
        #create a white circle at the randomly selected point
        cv2.circle(img, (x,y), 10, (255,255,255), -1)
        cv2.imshow('calibration', img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        #capture for 5 seconds
        t_end = time.time() + 5
        while time.time() < t_end:
            ret, frame = cap.read()
            # Operate on the frame
            # We're not doing anything special so grayscale should be good enough
            # Otherwise we'd just take the luminance values from the frame read
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_sc = np.float32(gray / 255.)
            gray_rs = np.reshape(gray_sc,(1,frame_h*frame_w))

            train_fn = theano.function([
            net1.fit(gray_rs,np.array([[np.float32(x),np.float32(y)]]))

if __name__ == '__main__':
    main()
