#file calibration.py
#calibration with a convolutional neural net
import os
import cv2
import numpy as np
import Tkinter as tk
import time
from random import randint

import tensorflow as tf
import openface
import dlib

from six.moves import cPickle

printing=False

def scale(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #perform histogram equalization
    img = cv2.equalizeHist(img)
    #now convert to [-1,1]
    img = (np.float32(img)/255.-.5)*2.

    return img

def calibrate(sess, optimizer, dur, cam, alib, face_dim, X, Y, F, x_tf, y_tf, screen_width = None, screen_height = None, x = None, y = None, pred = None):
    #capture for 10 seconds
    t_end = time.time() + dur
    while time.time() < t_end:
        ret, frame = cam.read()

        face_box = alib.getLargestFaceBoundingBox(frame)
        #Display the resulting frame

        if face_box is not None:
            top_left = (face_box.left(), face_box.top())
            bot_right = (face_box.right(), face_box.bottom())
            aligned_face = alib.align(face_dim, frame)
            cv2.imshow('aligned', aligned_face)

            #rescale and center from 0-255 to [-1,1]
            gray_face = scale(aligned_face)
            gray_face = [np.reshape(gray_face, (face_dim, face_dim, 1))]
            f = [[face_box.left(), face_box.top(), face_box.right(), face_box.bottom()]]

            #Now go train!
            sess.run(optimizer, feed_dict={X: gray_face, F: f, Y: [[x_tf,y_tf]]})

            """
            if pred is not None and x is not None and y is not None and screen_width is not None and screen_height is not None:
                p = sess.run(pred, feed_dict={X: gray_face, F: f})
                print 'On cal reference: %d -> %f, %d -> %f' % (x,x_tf,y,y_tf)
                print 'On cal actual: %f -> %d, %f -> %d' % (p[0][0],p[0][0]*screen_width/2.+screen_width/2.,p[0][1],p[0][1]*screen_height/2.+screen_height/2.)
            """




def test(sess, pred, cam, alib, face_dim, X, F, screen_width, screen_height, x_tf, y_tf, x, y):
    ret, frame = cam.read()
    face_box = alib.getLargestFaceBoundingBox(frame)
    #Display the resulting frame

    if face_box is not None:
        top_left = (face_box.left(), face_box.top())
        bot_right = (face_box.right(), face_box.bottom())
        aligned_face = alib.align(face_dim, frame)

        #rescale and center from 0-255 to [-1,1]
        gray_face = np.reshape(scale(aligned_face), (-1, face_dim, face_dim, 1))
        f = [[face_box.left(), face_box.top(), face_box.right(), face_box.bottom()]]

        p = sess.run(pred, feed_dict={X: gray_face, F: f})
        print 'Reference: %d -> %f, %d -> %f' % (x,x_tf,y,y_tf)
        print 'Actual: %f -> %d, %f -> %d' % (p[0][0],p[0][0]*screen_width/2.+screen_width/2.,p[0][1],p[0][1]*screen_height/2.+screen_height/2.)
        return p

def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)
    #return x

def maxpool2d(x, k=2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

def mlp(x, f, weights, biases, conv_drop, hidden_drop, face_dim):
    l1 = conv2d(x, weights['wc1'], biases['bc1'])
    l1 = maxpool2d(l1)
    l1 = tf.nn.l2_normalize(l1,0)
    l1 = tf.nn.dropout(l1, conv_drop)
    if printing: l1 = tf.Print(l1, [l1], 'l1: ')

    l2 = conv2d(l1, weights['wc2'], biases['bc2'])
    l2 = maxpool2d(l2)
    l2 = tf.nn.l2_normalize(l2,0)
    #l2 = tf.reshape(l2, [-1, weights['wd1'].get_shape().as_list()[0]]) #reshape to (?, 2048)
    l2 = tf.nn.dropout(l2, conv_drop)
    if printing: l2 = tf.Print(l2, [l2], 'l2: ')

    l3 = conv2d(l2, weights['wc3'], biases['bc3'])
    l3 = maxpool2d(l3)
    l3 = tf.reshape(l3, [-1, weights['wd1'].get_shape().as_list()[0]]) #reshape to (?, 2048)
    l3 = tf.nn.l2_normalize(l3,0)
    l3 = tf.nn.dropout(l3, conv_drop)
    if printing: l3 = tf.Print(l3, [l3], 'l3: ')

    l4 = tf.add(tf.matmul(l3, weights['wd1']), biases['bd1'])
    #l4 = tf.add(tf.matmul(l2, weights['wd1']), biases['bd1'])
    l4 = tf.nn.relu(l4)
    l4 = tf.nn.l2_normalize(l4,0)
    l4 = tf.nn.dropout(l4, hidden_drop)
    if printing: out = tf.Print(l4, [l4], 'l4: ')

    #append the bounding box locations
    #out = tf.concat(1, [l4, f])
    out = l4
    out = tf.add(tf.matmul(out, weights['out']), biases['out'])
    if printing: out = tf.Print(out, [out], 'out: ')
    return out

def mlp_dense(x, f, weights, biases, hidden_drop, face_dim):

    #The idea heare is that the first two layers will try to backproject and the final layer
    #will use the face location information to figure out where the origin truly was

    l1 = tf.reshape(x, [-1, face_dim**2]) #reshape to (?, face_dim*2)
    l1 = tf.add(tf.matmul(l1, weights['w1']), biases['b1'])
    l1 = tf.nn.l2_normalize(l1,0)
    l1 = tf.nn.relu(l1)
    l1 = tf.nn.dropout(l1, hidden_drop)

    l2 = tf.add(tf.matmul(l1, weights['w2']), biases['b2'])
    l2 = tf.nn.l2_normalize(l2,0)
    l2 = tf.nn.relu(l2)
    l2 = tf.nn.dropout(l2, hidden_drop)

    #add in the bounding box information
    out = tf.concat(1, [l2, f])
    out = tf.add(tf.matmul(l2, weights['out']), biases['out'])
    if printing: out = tf.Print(out, [out], 'out: ')
    return out


def main():
    root = tk.Tk()

    #setup openface
    fileDir = os.path.dirname(os.path.realpath(__file__))
    modelDir = os.path.join(fileDir, '..', 'openface/models')
    dlibModelDir = os.path.join(modelDir, 'dlib')
    openfaceModelDir = os.path.join(modelDir, 'openface')
    os.path.join(dlibModelDir,"shape_predictor_68_face_landmarks.dat")

    print dlibModelDir
    alib = openface.AlignDlib(dlibModelDir + "/shape_predictor_68_face_landmarks.dat")

    #use 1/2 screen width because I personally use dual monitors
    screen_width = root.winfo_screenwidth()/4
    screen_height = root.winfo_screenheight()/2

    img = np.zeros((screen_height, screen_width,3), np.uint8)
    print screen_width, screen_height

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
    learning_rate = .01
    conv_drop = .8
    hidden_drop = .8
    n_out = 2 #coordinates of where the user is gazing
    face_dim = 96 #size of face bounding box
    X = tf.placeholder(tf.float32, [None, face_dim, face_dim, 1])
    F = tf.placeholder(tf.float32, [None, 6]) #4 coordinates of a face + 2 for scaling
    Y = tf.placeholder(tf.float32, [None, n_out])

    # Store layers weight & bias
    weights = {
            'wc1': tf.Variable(tf.random_uniform([3, 3, 1, 32], minval=-.00001, maxval=.00001)),  # 4x4 conv,   1 input, 32 outputs
            'wc2': tf.Variable(tf.random_uniform([3, 3, 32, 64], minval=-.00001, maxval=.00001)),  # 4x4 conv, 32 input, 64 outputs
            'wc3': tf.Variable(tf.random_uniform([3, 3, 64, 128])), # 4x4 conv, 64 input, 128 outputs
            'wd1': tf.Variable(tf.random_uniform([128*4*4, 1024])), # fully connected. 128*4*4 inputs from conv layer
            #'out': tf.Variable(tf.random_uniform([1024+4, n_out])) #625 outputs from the conv layer + 4 inputs representing the location of the head within the original iamge
            'out': tf.Variable(tf.random_uniform([1024, n_out], minval=-.00001, maxval=.00001)) #625 outputs from the conv layer + 4 inputs representing the location of the head within the original iamge
            }

    biases = {
            'bc1': tf.Variable(tf.random_uniform([32], minval=-.00001, maxval=.00001)),
            'bc2': tf.Variable(tf.random_uniform([64], minval=-.00001, maxval=.00001)),
            'bc3': tf.Variable(tf.random_uniform([128], minval=-.00001, maxval=.00001)),
            'bd1': tf.Variable(tf.random_uniform([1024], minval=-.00001, maxval=.00001)),
            'out': tf.Variable(tf.random_uniform([n_out], minval=-.00001, maxval=.00001))
            }

    weights_d = {
            'w1' : tf.Variable(tf.random_uniform([face_dim**2, 2048], minval=-.00001, maxval=.00001)),
            'w2' : tf.Variable(tf.random_uniform([2048, 512], minval=-.00001, maxval=.00001)),
            'out': tf.Variable(tf.random_uniform([512, n_out], minval=-.00001, maxval=.00001)),
            }

    biases_d = {
            'b1' : tf.Variable(tf.random_uniform([2048], minval=-.00001, maxval=.00001)),
            'b2' : tf.Variable(tf.random_uniform([512], minval=-.00001, maxval=.00001)),
            'out': tf.Variable(tf.random_uniform([n_out], minval=-.00001, maxval=.00001))
            }

    #create multilayer perceptron
    #pred = mlp(X, F, weights, biases, conv_drop, hidden_drop, face_dim)
    pred = mlp_dense(X, F, weights_d, biases_d, hidden_drop, face_dim)

    #define cost function
    #for now the cost function is the MSE
    #figure out how to do cross entropy with logits maybe? I mean this is a regression problem so...
    cost = tf.pow(pred-Y,2)
    cost = tf.reduce_mean(cost)

    train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    #train_op = tf.train.RMSPropOptimizer(.1, .9).minimize(cost)
    #train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
    #pred_op = tf.argmax(pred, 1)


    init = tf.initialize_all_variables()

    with tf.Session() as sess:
        sess.run(init)

        print('Calibrating')
        """
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

            calibrate(sess, train_op, cap, 2, n_input, X, Y, x/float(screen_width) , y/float(screen_height))
            ret, frame = cap.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_rs = np.reshape(gray,(1,n_input))
            print sess.run(pred, feed_dict={X: gray_rs/255.}).shape
        #alternative calibration
        for x in xrange(0,screen_width,100):
            for y in xrange(0,screen_height,100):
                #create a white circle at the randomly selected point
                img[:] = (0,0,0) # clear
                cv2.circle(img, (x,y), 10, (255,255,255), -1)

                cv2.waitKey(100)
                cv2.imshow('calibration', img)
                cv2.waitKey(100)

                x_tf = (float(x)/screen_width-.5)*2.
                y_tf = (float(y)/screen_height-.5)*2.
                calibrate(sess, train_op, .5, cap, alib, face_dim, X, Y, F, x_tf, y_tf)
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
            x_tf = (float(x)/screen_width-.5)*2.
            y_tf = (float(y)/screen_height-.5)*2.
            calibrate(sess, train_op, .5, cap, alib, face_dim, X, Y, F, x_tf, y_tf, screen_width, screen_height, x, y, pred)
            p_tf = test(sess, pred, cap, alib, face_dim, X, F, float(screen_width), float(screen_height), x_tf, y_tf, x, y)
            if p_tf is not None:
                p = np.multiply(np.array(p_tf)/2.+.5, np.array([screen_width, screen_height])).astype(int)

                #draw
                cv2.circle(img, (p[0][0], p[0][1]), 10, (255,0,0), -1)
                cv2.imshow('Focus', img)
                #Display the resulting frame
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        cv2.destroyWindow('calibration')
        print('Now continuing onto testing')

        """
        img = np.zeros((screen_height, screen_width,3), np.uint8)

        while(True):
            #generate the dot
            x = randint(0,screen_width-1)#/screen_width
            y = randint(0,screen_height-1)#/screen_height
            x_tf = (float(x)/screen_width-.5)*2.
            y_tf = (float(y)/screen_height-.5)*2.

            #attempt to perform the backprojection
            p_tf = test(sess, pred, cap, alib, face_dim, X, F, float(screen_width), float(screen_height), x_tf, y_tf, x, y)
            if p_tf is not None:
                p = np.multiply(np.array(p_tf)/2.+.5, np.array([screen_width, screen_height])).astype(int)

                #draw
                img[:] = (0,0,0) # clear
                cv2.circle(img, (p[0][0], p[0][1]), 10, (255,255,255), -1)
                cv2.imshow('Focus', img)
                #Display the resulting frame
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        """

if __name__ == '__main__':
    main()
