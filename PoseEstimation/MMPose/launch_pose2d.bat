@echo OFF
setlocal EnableDelayedExpansion

:: Set the path to your local Anaconda installation
set ANACONDA_PATH=%USERPROFILE%\Anaconda3

:: Set the name of your Conda environment
set CONDA_ENV=openmmlab

:: Activate the Conda environment
call %ANACONDA_PATH%\Scripts\activate.bat %CONDA_ENV%

:: Run Python in the activated environment
python demo/premiere_mmpose.py webcam --pose2d ipr_res50_8xb64-210e_coco-256x256

:: Deactivate the environment
call conda deactivate

:: Keep the command window open
cmd /k