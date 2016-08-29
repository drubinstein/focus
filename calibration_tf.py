#file calibration.py
import os
import cv2
import numpy as np
import Tkinter as tk
import time
from random import randint

import matplotlib.pyplot as pyplot
import tensorflow as tf


from six.moves import cPickle

NUM_HIDDEN_UNITS=100
printing=True

def calibrate(sess, optimizer, cam, dur, n_input, X, Y, x, y):
    #capture for 5 seconds
    t_end = time.time() + dur
    while time.time() < t_end:
        ret, frame = cam.read()
        # Operate on the frame
        # We're not doing anything special so grayscale should be good enough
        # Otherwise we'd just take the luminance values from the frame read
        gray = np.float32(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        gray_rs = np.reshape(gray,(1,n_input)) - 127.5
        #gray_sc = np.float32(gray / 255.)

        #Now go train!
        sess.run(optimizer, feed_dict={X: gray_rs/255., Y: [[x,y]]})


def test(sess, pred, cam, n_input, X, screen_width, screen_height, x_tf, y_tf, x, y):
    ret, frame = cam.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_rs = np.reshape(gray,(1,n_input)) - 127.5
    p = sess.run(pred, feed_dict={X: gray_rs/255.})
    print 'Reference: %d -> %f, %d -> %f' % (x,x_tf,y,y_tf)
    print 'Actual: %f -> %d, %f -> %d' % (p[0][0],p[0][0]*screen_width/2.+screen_width/2.,p[0][1],p[0][1]*screen_height/2.+screen_height/2.)

def mlp(x, weights, biases, dropout):
    #Hidden Layer with tanh activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1  = tf.nn.l2_normalize(layer_1, dim=0)
    layer_1 = tf.nn.relu(layer_1)
    layer_1 = tf.nn.dropout(layer_1, dropout)
    if printing: layer_1 = tf.Print(layer_1, [layer_1], 'layer 1: ')
    #Hidden Layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2  = tf.nn.l2_normalize(layer_2, dim=0)
    layer_2 = tf.nn.relu(layer_2)
    layer_2 = tf.nn.dropout(layer_2, dropout)
    if printing: layer_2 = tf.Print(layer_2, [layer_2], 'layer 2: ')
    #layer 3
    layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
    layer_3  = tf.nn.l2_normalize(layer_3, dim=0)
    layer_3 = tf.nn.relu(layer_3)
    layer_3 = tf.nn.dropout(layer_3, dropout)
    if printing: layer_3 = tf.Print(layer_3, [layer_3], 'layer 3: ')
    #Output layer
    out_layer = tf.matmul(layer_3, weights['out']) + biases['out']
    #out_layer = tf.maximum(out_layer,[[-1,-1]])
    #out_layer = tf.minimum(out_layer,[[1,1]])
    if printing: out_layer = tf.Print(out_layer,[out_layer], 'output layer: ')
    return out_layer

def main():
    root = tk.Tk()

    fs = 100

    print('starting video')
    cap = cv2.VideoCapture(0)
    #get video resolution
    ret, frame = cap.read()
    if ret == False: os.sys.exit("No camera detected")
    print frame.shape
    frame_h, frame_w, _ = frame.shape
    npxls = frame_h*frame_w
    print "Frame dims are '{0}' x '{1}'".format(frame_w,frame_h)

    #setup the calibration window
    screen_width = frame_w  #root.winfo_screenwidth()/2
    screen_height = frame_h #root.winfo_screenheight()/2
    img = np.zeros((screen_height, screen_width,3), np.uint8)
    print 'window width: %d, window height: %d' % (screen_width, screen_height)

    print('Initializing neural net')
    learning_rate = .01
    dropout = .75
    n_input = npxls
    n_hidden_1 = NUM_HIDDEN_UNITS
    n_hidden_2 = NUM_HIDDEN_UNITS
    n_hidden_3 = NUM_HIDDEN_UNITS
    n_out = 2
    X = tf.placeholder(tf.float32, [None, frame_h*frame_w])
    Y = tf.placeholder(tf.float32, [None, n_out])
    # Store layers weight & bias
    weights = {
        'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
        'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
        'h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3])),
        #'out': tf.Variable(tf.zeros([n_hidden_3, n_out]))
        'out': tf.Variable(tf.random_normal([n_hidden_3, n_out]))
        }
    biases = {
        'b1': tf.Variable(tf.random_normal([n_hidden_1])),
        'b2': tf.Variable(tf.random_normal([n_hidden_2])),
        'b3': tf.Variable(tf.random_normal([n_hidden_3])),
        #'out': tf.Variable(tf.random_normal([n_out],.5))
        'out': tf.Variable(tf.random_normal([n_out]))
        }

    #create multilayer perceptron
    pred = mlp(X,weights, biases, dropout)

    #define cost function
    #define cost function
    cost = tf.pow(pred-Y,2)
    #if printing: cost = tf.Print(cost,[cost],'Sq.Err.: ')
    cost = tf.reduce_mean(cost)
    if printing: cost = tf.Print(cost,[cost],'MSE: ')
    #cost = tf.sqrt(cost)
    #if printing: cost = tf.Print(cost,[cost],'RMSE: ')
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

    init = tf.initialize_all_variables()

    with tf.Session() as sess:
        sess.run(init)

        print('Calibrating')
        """
        npts = 20
        print "generating '{0}' random points to choose from".format(npts)
        for _ in xrange(0,npts):
            img[:] = (0,0,0) # clear

            x = randint(0,screen_width-1)#/screen_width
            y = randint(0,screen_height-1)#/screen_height
            print x,y

            #create a white circle at the randomly selected point
            cv2.circle(img, (x,y), 10, (255,255,255), -1)

            cv2.waitKey(100)
            cv2.imshow('calibration', img)
            cv2.waitKey(100)

            calibrate(sess, optimizer, cap, 1, n_input, X, Y, x/float(screen_width) , y/float(screen_height))
            test(sess, pred, cap, n_input, X, screen_width, screen_height,x,y)
        """
        #alternative calibration
        for x in xrange(0,screen_width,100):
            for y in xrange(0,screen_height,100):
                #create a white circle at the randomly selected point
                img[:] = (0,0,0) # clear
                cv2.circle(img, (x,y), 10, (255,255,255), -1)

                cv2.waitKey(100)
                cv2.imshow('calibration', img)
                cv2.waitKey(100)

                #normalize x and y to be between -1 and 1
                x_tf = (x-screen_width/2.)/(screen_width/2.)
                y_tf = (y-screen_height/2.)/(screen_height/2.)
                calibrate(sess, optimizer, cap, .1,n_input,X,Y,x_tf,y_tf)
                test(sess, pred, cap, n_input, X, float(screen_width), float(screen_height), x_tf,y_tf,x, y)
        """
        #Lets see if it can figure out a dot....
        x = screen_width/4
        y = screen_height/4
        while True:
            img[:] = (0,0,0) # clear
            cv2.circle(img, (x,y), 10, (255,255,255), -1)

            cv2.waitKey(100)
            cv2.imshow('calibration', img)
            cv2.waitKey(100)

            #normalize x and y to be between -1 and 1
            x_tf = (float(x)-screen_width/2)/(screen_width/2.)
            y_tf = (float(y)-screen_height/2)/(screen_height/2.)
            calibrate(sess, optimizer, cap, .1,n_input,X,Y,x_tf,y_tf)
            test(sess, pred, cap, n_input, X, float(screen_width), float(screen_height), x_tf,y_tf,x,y)
        """


        cv2.destroyWindow('calibration')
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
            gray_rs = np.reshape(gray,(1,n_input)) - 127.5

            feed_dict = {X: gray_rs/255.}
            p = sess.run(pred, feed_dict)
            print p[0][0]*screen_width/2+screen_width/2, p[0][1]*screen_height/2+screen_height/2

            p_rnd_x, p_rnd_y = np.int32([p[0][0]*screen_width+screen_width/2, p[0][1]*screen_height+screen_height/2])
            print p_rnd_x, p_rnd_y

            cv2.imshow('GrayFrame',gray)
            img[:] = (0,0,0) # clear
            cv2.circle(img, (p_rnd_x, p_rnd_y), 10, (255,255,255), -1)
            cv2.imshow('Focus', img)
            #Display the resulting frame
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

if __name__ == '__main__':
    main()
