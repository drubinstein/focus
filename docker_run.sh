#!/bin/sh
#Title
#Description
#By David Rubinstein
#alt --user=$USER

#figure out way to get tensorflow inside f the instance
#for CPU
# export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.11.0rc0-cp27-none-linux_x86_64.whl
#for GPU
# export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow-0.11.0rc0-cp27-none-linux_x86_64.whl
#sudo pip install --upgrade $TF_BINARY_URL
docker run -it -p 55555:22 -p 9000:9000 -p 8000:8000 \
	--user $(id -u) \
	--env="DISPLAY" \
	--workdir="/home/$USER" \
	--volume="/home/$USER:/home/$USER" \
	--volume="/etc/group:/etc/group:ro" \
    --volume="/etc/passwd:/etc/passwd:ro" \
    --volume="/etc/shadow:/etc/shadow:ro" \
    --volume="/etc/sudoers.d:/etc/sudoers.d:ro" \
    --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
	--volume="/home/yotta/david/git/focus:/mount/focus" \
	--volume="/dev/video0:/dev/video0" \
	--privileged \
	bamos/openface \
	/bin/bash
	#osrf/ros:indigo-desktop-full \
    #rqt

