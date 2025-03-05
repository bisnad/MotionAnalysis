# 2D Pose Tracker #

Daniel Bisig - Instituto Stocos, Spain - daniel@stocos.com, Zurich University of the Arts, Switzerland - daniel.bisig@zhdk.ch

### Overview ###

A simple wrapper for the Yolo V8 2d pose tracker. This wrapper employs Yolo for tracking 2d poses of humans in a single live video image. Each pose of each tracked human is represented by the 2D positions of 17 joints. 

## OSC-Communication

The tracker communicates with other software applications by using the OSC (Open Sound Control) protocol. During tracking, the tracker sends for each detected human the 2d positions of the skeleton joints. These positions are normalised with regards to the size of the video image. 

#### The following OSC messages are sent by the tracker:

- /mocap/skelX/joint/pos_world <Float> <Float> <Float> <Float> <Float>: sends the 2d joint positions for each tracked human. The "X" in the message translates into the index of the tracked human. 

The code has been tested on Windows and MacOS. Anaconda environments for these two operating systems are provided as part of this repository. 