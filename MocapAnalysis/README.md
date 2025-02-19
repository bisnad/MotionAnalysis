# AI-Toolbox - Motion Analysis - Motion Capture Analysis

![analysis_screenshot](./data/media/analysis_screenshot.png)

## Summary

This C++-based tool provides a set of algorithms for analysing motion capture recordings in real-time. The following analysis functions are provided: positional and rotational derivatives (velocity, acceleration, and jerk) and Laban Effort Factors (Flow, Time,  Space, Weight). The software receives motion capture data via OSC and sends the analysis results also via OSC. 

## Usage
**Select Data for Plotting**

The first pull down menu allows to chose which of the incoming motion capture data or analysis results is displayed as time-varying graph. Only one type of data can be displayed at a time. One graph will be shown for each dimension of the data. Each graph displays the data for either all the individual skeleton joints (position, rotation, velocity, acceleration, jerk) or groups of skeleton joints (Flow, Time, Space, Weight). 

**Select Data for OSC Sending**

The second pull down menu allows to chose which of the incoming motion capture data or analysis results is sent as OSC message to a client. Any number and combination can be sent at the same time. The items in this menu indicate the addresses of the corresponding OSC messages, with the exception that the menu displays them in upper case while the actual addresses are lowercase. The IP-address and port to which these messages are sent are currently fixed to: 127.0.0.1:9004. The analysis software receives OSC data from the MocapPlayer on a fixed port which is 9003. The data and analysis results sent from the Analysis software are in interleaved format. For example the positions of all skeleton joints are sent as follows, with the number indicating the index of the joint: x0, y0, z0, x1, y1, z1, .... 

**Note on Laban Effort Factors**

Laban Effort Factors are calculated for five different groups of skeleton joints: 1. all joints 2. torso and head joints, left arm joints, right arm joints, left leg joints, right leg joints. These values are all sent together via OSC. 

## OSC Communication

The tool sends OSC messages representing the the motion descriptors that have been selected by the user for sending. For each motion descriptor, all descriptor values are grouped together into a single OSC message. With the exception of motion descriptors based on Laban Effort Factors, each message contains all the joint properties grouped together as follows: j1_p1 j1_p2 ... j1_pD, j2_p1, j2_p2, ... j2_pD, ... , jN_p1, jN_p2, ... jN_pD- Here, j stands for joint, p for parameter, N for number of joints, and D for dimension of parameters. For the Laban Effort factors, each message contains the properties for the five different skeleton joint groups. 

The following OSC messages can be chosen to be sent by the software:

- joint positions as list of 3D vectors in world coordinates: `/mocap/0/joint/pos_world <float j1x> <float j1y> <float j1z> .... <float jNx> <float jNy> <float jNz>` 

- joint rotations as list of Quaternions in world coordinates: `/mocap/0/joint/rot_local <float j1w> <float j1x> <float j1y> <float j1z> .... <float jNw> <float jNx> <float jNy> <float jNz>` 

The following OSC messages are received by the tool:

- select of motion descriptor for playback and OSC: `/synth/motionfeature <string feature_name>`
- select of cluster index for playback and OSC: `/synth/clusterlabel <int cluster_index>`