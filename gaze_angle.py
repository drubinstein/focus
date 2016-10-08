#file gaze_angle.py
#Use the Columbia Gaze Data Set to estimate the angle of someone's gaze

import os
import cv2
import numpy as np
import Tkinter as tk
import time
from random import randint

import tensorflow as tf
import openface
import dlib

import re

printing=False

def train(path_to_dataset='/home/david/git/focus/Columbia Gaze Data Set'):
    for root, dirs, files in os.walk(path_to_dataset):
        for d in dirs:
            subdir_path = os.path.join(root, d)
            #now get out all the files and train
            for subroot, subdirs, subfiles in os.walk(subdir_path):
                for f in subfiles:
                    filepath = os.path.join(subroot, f)
                    #check to make sure it is in image
                    if f[-3:] == 'jpg':
                        #parse the filename
                        #file name is in the format person_distance_
                        split_fname = re.split('_|H|V|P|\.',f)

                        h_pose = int(split_fname[2])
                        v_gaze = int(split_fname[4])
                        h_gaze = int(split_fname[6])




train()

