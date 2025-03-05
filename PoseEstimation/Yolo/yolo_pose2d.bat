@echo OFF
setlocal EnableDelayedExpansion

:: Set the path to your local Anaconda installation
set ANACONDA_PATH=%USERPROFILE%\Anaconda3

:: Set the name of your Conda environment
set CONDA_ENV=ultralytics

:: Activate the Conda environment
call %ANACONDA_PATH%\Scripts\activate.bat %CONDA_ENV%

:: Run Python in the activated environment
python yolo_pose2d.py webcam --pose2d yolov8x-pose.pt

:: Deactivate the environment
call conda deactivate

:: Keep the command window open
cmd /k