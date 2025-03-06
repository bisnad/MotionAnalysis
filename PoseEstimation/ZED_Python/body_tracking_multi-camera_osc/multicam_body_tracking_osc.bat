@echo OFF
setlocal EnableDelayedExpansion

:: Set the path to your local Anaconda installation
set ANACONDA_PATH=%USERPROFILE%\Anaconda3

:: Set the name of your Conda environment
set CONDA_ENV=zed

:: Activate the Conda environment
call %ANACONDA_PATH%\Scripts\activate.bat %CONDA_ENV%

:: Run Python in the activated environment
python multicam_body_tracking_osc.py data/calibs/calib_file.json

:: Deactivate the environment
call conda deactivate

:: Keep the command window open
cmd /k