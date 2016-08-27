# file detectMyFace.py
import os

import numpy as np
from pandas.io.parsers import read_csv
from sklearn.utils import shuffle

from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
import matplotlib.pyplot as pyplot
import cv2

import matplotlib.pyplot as plt
from scipy import misc
import math

from six.moves import cPickle

def plot_sample(x, y, axis):
    img = x.reshape(96, 96)
    axis.imshow(img, cmap='gray')
    axis.scatter(y[0::2] * 48 + 48, y[1::2] * 48 + 48, marker='x', s=10)

#load the newly trained model
print('Loading nn.mdl')
f = open('nn.mdl', 'rb')
net1 = cPickle.load(f)
f.close()

print('Now continuing onto video capture')
cap = cv2.VideoCapture(0)

plt.ion()
fig = plt.figure()
ax = plt.gca()
square_size = 96

while(True):
    #Capture frame by frame
    ret, frame = cap.read()

    # Operate on the frame
    # We're not doing anything special so grayscale should be good enough
    # Otherwise we'd just take the luminance values from the frame read
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #cut off the left and right sides (cause wide screen is stupid)
    y_size, x_size = gray.shape
    gray_downsampled = gray[:,range(x_size/2-y_size/2,x_size/2+y_size/2)]
    #downsample the image to a 96x96 image
    gray_downsampled = misc.imresize(gray_downsampled,(square_size, square_size))

    #now fit the downsampled image to what we just classified
    #Get it we're focusing on the face :p
    gray_downsampled_sc = np.float32(gray_downsampled / 255.)
    gray_downsampled_rs = np.reshape(gray_downsampled_sc,(1,square_size*square_size))

    y_pred = net1.predict(gray_downsampled_rs)


    cv2.imshow('GrayFrame',gray_downsampled)
    #Display the resulting frame
    plot_sample(gray_downsampled_rs,y_pred[0],ax)
    plt.show(block=False)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    plt.pause(0.01)
    plt.cla()

#When everything is done, end the capture
cap.release()
cap.destroyAllWindows()
