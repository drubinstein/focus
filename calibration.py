#file calibration.py
import os
import cv2
import numpy as np
import Tkinter as tk
import time

root = tk.Tk()

screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

img = np.zeros((screen_height, screen_width/2,3), np.uint8)
print screen_width, screen_height
fs = 100
#TODO: Instead of looping through every w and h instead choose 25 random points and
#get captures for those with half-second delays
for w in xrange(0,screen_width/2,fs):
    for h in xrange(0,screen_height,fs):
        img[:] = (0,0,0) # clear

        cv2.circle(img, (w,h), 25, (255,255,255), -1)
        cv2.imshow('',img)
        time.sleep(.1)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

