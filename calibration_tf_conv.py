#file calibration.py
#calibration with a convolutional neural net
import os
import cv2
import numpy as np
import Tkinter as tk
import time
from random import randint

import tensorflow as tf

from six.moves import cPickle

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
        gray_rs = np.reshape(gray,(1,n_input))
        #gray_sc = np.float32(gray / 255.)

        #Now go train!
        sess.run(optimizer, feed_dict={X: gray_rs/255., Y: [[x,y]]})

def test(sess, pred, cam, n_input, X, screen_width, screen_height)
    ret, frame = cam.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_rs = np.reshape(gray,(1,n_input))
    p = sess.run(pred, feed_dict={X: (gray_rs-127.5)/255.})
    print x,y
    print p[0][0]*screen_width+screen_width/2., p[0][1]*screen_height+screen_height/2.

def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

def maxpool2d(x, k=2):
   return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
        padding='SAME')

def mlp(x, weights, biases, dropout, w, h):
    #Resahpe for convolution layer
    x = tf.reshape(x, shape=[-1, h, w, 1])

    #Convolution layer
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    #Max pooling (down-sampling)
    conv1 = maxpool2d(conv1, k=2)
    if printing: conv1 = tf.Print(conv1, [conv1], 'conv1: ', summarize=10)

    #Convolution layer
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    #Max pooling (down-sampling)
    conv2 = maxpool2d(conv2, k=2)
    if printing: conv2 = tf.Print(conv2, [conv2], 'conv2: ', summarize=10)

    #Fully connected layer
    #Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    #Apply dropout
    fc1 = tf.nn.relu(fc1)
    fc1 = tf.nn.dropout(fc1, dropout)
    if printing: fc1 = tf.Print(fc1, [fc1], 'fc1: ', summarize=10)

    #Output, class prediction
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    if printing: out = tf.Print(out, [out], 'out: ')
    return out

def main():
    root = tk.Tk()

#use 1/2 screen width because I personally use dual monitors
    screen_width = root.winfo_screenwidth()/4
    screen_height = root.winfo_screenheight()/2

    img = np.zeros((screen_height, screen_width,3), np.uint8)
    print screen_width, screen_height
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

    print('Initializing neural net')
    learning_rate = 0.0001
    dropout = .75
    n_input = npxls
    n_out = 2
    X = tf.placeholder(tf.float32, [None, frame_h*frame_w])
    Y = tf.placeholder(tf.float32, [None, n_out])
    # Store layers weight & bias
    #TODO: Figure out why output is not 1x2 and instead is 75x2. Must have something to do with wd1
    weights = {
        # 5x5 conv, 1 input, 32 outputs
        'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
        # 5x5 conv, 32 inputs, 64 outputs
        'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
        # fully connected, 8*8*64 inputs, 1024 outputs
        'wd1': tf.Variable(tf.random_normal([256*64, 1024])),
        # 1024 inputs, 2 outputs (class prediction)
        'out': tf.Variable(tf.random_normal([1024, n_out]))
    }

    biases = {
        'bc1': tf.Variable(tf.random_normal([32])),
        'bc2': tf.Variable(tf.random_normal([64])),
        'bd1': tf.Variable(tf.random_normal([1024])),
        'out': tf.Variable(tf.random_normal([n_out]))
    }

    #create multilayer perceptron
    pred = mlp(X, weights, biases, dropout, frame_w, frame_h)

    #define cost function
    cost = tf.pow(pred-Y,2)
    #if printing: cost = tf.Print(cost,[cost],'Sq.Err.: ')
    cost = tf.reduce_mean(cost)
    #if printing: cost = tf.Print(cost,[cost],'MSE: ')
    cost = tf.sqrt(cost)
    if printing: cost = tf.Print(cost,[cost],'RMSE: ')

    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

    init = tf.initialize_all_variables()

    with tf.Session() as sess:
        sess.run(init)

        print('Calibrating')
        npts = 10
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

            calibrate(sess, optimizer, cap, 2, n_input, X, Y, x/float(screen_width) , y/float(screen_height))
            ret, frame = cap.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_rs = np.reshape(gray,(1,n_input))
            print sess.run(pred, feed_dict={X: gray_rs/255.}).shape
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

                calibrate(sess, optimizer, cap, .1,n_input,X,Y,x/float(screen_width),y/float(screen_height))
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
            gray_rs = np.reshape(gray,(1,n_input))

            feed_dict = {X: gray_rs/255.}
            p = sess.run(pred, feed_dict)
            print p[0]

            p_rnd = np.int32(p[0])

            cv2.imshow('GrayFrame',gray)
            img[:] = (0,0,0) # clear
            cv2.circle(img, (p_rnd[0]*screen_width, p_rnd[1]*screen_height), 10, (255,255,255), -1)
            cv2.imshow('Focus', img)
            #Display the resulting frame
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

if __name__ == '__main__':
    main()
