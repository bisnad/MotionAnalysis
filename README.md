# AI-Toolbox - Motion Analysis

The "Motion Analysis" category of the AI-Toolbox contains a collection of mostly python-based utilities for detecting body key-points in video, for deriving low and medium-level motion descriptors from motion data, for classifying motion data, and for re-organising motion according to the similarity among motion descriptors. 

The following tools are available:

- [Yolo Pose Estimation](PoseEstimation/Yolo)

  A Python script that employs the Yolo 2D pose estimation model on images, movie recordings, or a live camera input

- [MMPose Pose Estimation](PoseEstimation/MMPose)

  A Python script that Employs one of the many 2D or 3D pose estimation models included in MMPose on images, movie recordings, or a live camera input

- [ZED Pose Estimation](PoseEstimation/ZED_C++)

​	Several C++ applications that employ the Stereolabs 3D pose estimation model on recordings or live camera input from a ZED stereovision camera

- [Mocap Analysis](MocapAnalysis)

  A C++ application for deriving motion descriptors in real-time from an incoming motion capture data stream. 

- [Mocap Analysis Python](MocapAnalysisPython)

  A Python script for deriving motion descriptors in real-time from an incoming motion capture data stream. 

- [Mocap Classifier](MocapClassifier)

  A Python script for training a model on a motion classification task.

- [Mocap Classifier Interactive](MocapClassifier_Interactive)

  A  Python for using a trained model in classify in real-time motions in an incoming motion capture data stream.

- [Clustering Interactive](ClusteringInteractive)

  A Python script for grouping motion segments in a motion capture recording according to their similarity with regards to chosen motion descriptors.

- [Clustering Interactive Position 2D](ClusteringInteractive_pos2d)

  The same script as in Clustering Interactive but specific for motion capture recording that consists of 2D joint positions only.

- [Nearest Neighbors](NearestNeighbors)

  A Python script to create a new motion sequence by conducting a nearest neighbor search among motion segments in a motion capture recording  according to their similarity with regards to chosen motion descriptors.
