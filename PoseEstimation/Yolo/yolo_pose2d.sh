#!/bin/bash

# Set the path to your Anaconda installation
CONDA_PATH="$HOME/opt/anaconda3"

# Set the name of your Conda environment
CONDA_ENV="ultralytics"

# Activate the Conda environment
source "$CONDA_PATH/bin/activate" "$CONDA_ENV"

# Set the path of the directory that contains the python script
SCRIPT_PATH=$(cd "$(dirname "$0")"; pwd -P)

# Change into the directory that contains the python script
cd "$SCRIPT_PATH"

# Run the Python script
python pose_2d.py

# Deactivate the Conda environemnt
conda deactivate