# focus
Project to track where my focus/gaze is is
Currently using tensorflow (used to be theano + lasagne)
Requirements:
-OpenCv2.4
-an nvidia graphics card
-CUDA and cudnn
-TensorFlow

There are two files that can be tested with
Both currently dont work. Any advice would be appreciated
1) calibration_tf.py : Uses a densely connected 3-layer neural net to track where you are looking after calibration
2)calibration_tf_conf.py : Uses a convolutional neural net to perform the same task as in (1) 
