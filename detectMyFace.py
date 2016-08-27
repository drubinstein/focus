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


FTRAIN = '~/data/kaggle-facial-keypoint-detection/training.csv'
FTEST  = '~/data/kaggle-facial-keypoint-detection/test.csv'

def plot_sample(x, y, axis):
    img = x.reshape(96, 96)
    axis.imshow(img, cmap='gray')
    axis.scatter(y[0::2] * 48 + 48, y[1::2] * 48 + 48, marker='x', s=10)

def load(test=False, cols=None):
    """Loads data from FTEST if *test* is True, otherwise from FTRAIN.
    Pass a list of *cols* if you're only interested in a subset of the
    target columns.
    """
    fname = FTEST if test else FTRAIN
    df = read_csv(os.path.expanduser(fname))  # load pandas dataframe

    # The Image column has pixel values separated by space; convert
    # the values to numpy arrays:
    df['Image'] = df['Image'].apply(lambda im: np.fromstring(im, sep=' '))

    if cols:  # get a subset of columns
        df = df[list(cols) + ['Image']]

    print(df.count())  # prints the number of values for each column
    df = df.dropna()  # drop all rows that have missing values in them

    X = np.vstack(df['Image'].values) / 255.  # scale pixel values to [0, 1]
    X = X.astype(np.float32)

    if not test:  # only FTRAIN has any target columns
        y = df[df.columns[:-1]].values
        y = (y - 48) / 48  # scale target coordinates to [-1, 1]
        X, y = shuffle(X, y, random_state=42)  # shuffle train data
        y = y.astype(np.float32)
    else:
        y = None

    return X, y


"""
X, y = load()
print("X.shape == {}; X.min == {:.3f}; X.max == {:.3f}".format(
    X.shape, X.min(), X.max()))
print("y.shape == {}; y.min == {:.3f}; y.max == {:.3f}".format(
    y.shape, y.min(), y.max()))

net1 = NeuralNet(
    layers=[  # three layers: one hidden layer
        ('input', layers.InputLayer),
        ('hidden', layers.DenseLayer),
        ('output', layers.DenseLayer),
        ],
    # layer parameters:
    input_shape=(None, 9216),  # 96x96 input pixels per batch
    hidden_num_units=100,  # number of units in hidden layer
    output_nonlinearity=None,  # output layer uses identity function
    output_num_units=30,  # 30 target values

    # optimization method:
    update=nesterov_momentum,
    update_learning_rate=0.01,
    update_momentum=0.9,

    regression=True,  # flag to indicate we're dealing with regression problem
    max_epochs=400,  # we want to train this many epochs
    verbose=1,
    )

#load and train based off the training set
X, y = load()
net1.fit(X, y)
"""

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
