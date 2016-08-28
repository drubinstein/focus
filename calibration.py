#file calibration.py
import os
import cv2
import numpy as np
import Tkinter as tk
import time
from random import randint

import lasagne
import matplotlib.pyplot as pyplot

import theano
import theano.tensor as T

from six.moves import cPickle

NUM_HIDDEN_UNITS=30
BATCH_SIZE=1

def build_model(frame_w, frame_h, output_dim,
                batch_size=BATCH_SIZE, num_hidden_units=NUM_HIDDEN_UNITS, input_var=None):
    """Create a symbolic representation of the neural network with
    input_dim input nodes
    output_dim output nodes
    num_hidden_units per hidden layer
    input_var symbolic input variable

    return: A theano expression representing the network
    """
    l_in = lasagne.layers.InputLayer(shape=(None, 1, frame_h,frame_w), input_var=input_var)
    l_hidden1 = lasagne.layers.DenseLayer(
        l_in,
        num_units=num_hidden_units,
        nonlinearity=lasagne.nonlinearities.rectify,
    )
    l_hidden2 = lasagne.layers.DenseLayer(
        l_hidden1,
        num_units=num_hidden_units,
        nonlinearity=lasagne.nonlinearities.rectify,
    )
    l_out = lasagne.layers.DenseLayer(
        l_hidden2,
        num_units=output_dim,
        nonlinearity=lasagne.nonlinearities.linear,
        b=None,
    )
    return l_out


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
    input_var = T.tensor4('inputs')
    target_var = T.ivector('targets')
    network = build_model(frame_w, frame_h, 2, 1, 30, input_var)
    prediction = lasagne.layers.get_output(network)
    predict_fn = theano.function([input_var], prediction)
    #loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    #loss = loss.mean()
    loss = T.sum(lasagne.objectives.squared_error(target_var,prediction))
    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.nesterov_momentum(loss, params, learning_rate=0.01, momentum=0.9)
    train_fn = theano.function([input_var, target_var], loss, updates=updates)

    print('generating 25 random points to choose from')
    npts = 5
    for _ in xrange(0,npts):
        img[:] = (0,0,0) # clear

        x = randint(0,screen_width-1)
        y = randint(0,screen_height-1)
        print x,y

        target = np.array([[np.float32(x), np.float32(y)]])
        #create a white circle at the randomly selected point
        cv2.circle(img, (x,y), 10, (255,255,255), -1)

        cv2.waitKey(100)
        cv2.imshow('calibration', img)
        cv2.waitKey(100)


        #capture for 5 seconds
        t_end = time.time() + 5
        while time.time() < t_end:
            ret, frame = cap.read()
            # Operate on the frame
            # We're not doing anything special so grayscale should be good enough
            # Otherwise we'd just take the luminance values from the frame read
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_sc = np.float32(gray / 255.)

            #Now go train!
            coord=np.int32(np.array([x,y]))
            train_fn([[gray_sc]], coord)

    print('Saving the model to focus.mdl')
    f = open('focus.mdl', 'wb')
    cPickle.dump(network, f, protocol = cPickle.HIGHEST_PROTOCOL)
    f.close()

    print('Now continuing onto testing')
    img = np.zeros((screen_height, screen_width,3), np.uint8)

    while(True):
        #Capture frame by frame
        ret, frame = cap.read()

        # Operate on the frame
        # We're not doing anything special so grayscale should be good enough
        # Otherwise we'd just take the luminance values from the frame read
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        #cut off the left and right sides (cause wide screen is stupid)
        gray_sc = np.float32(gray / 255.)


        y_pred = lasagne.layers.get_output(network, inputs=gray_sc)
        print eval(y_pred)


        cv2.imshow('GrayFrame',gray)
        img[:] = (0,0,0) # clear
        cv2.circle(img, y_pred, 10, (255,255,255), -1)
        #Display the resulting frame
        plt.show(block=False)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == '__main__':
    main()
