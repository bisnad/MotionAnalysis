#!/bin/bash

# Set the path to your Anaconda installation
CONDA_PATH="$HOME/opt/anaconda3"

# Set the name of your Conda environment
CONDA_ENV="mmpose"

# Activate the Conda environment
source "$CONDA_PATH/bin/activate" "$CONDA_ENV"

# Set the path of the directory that contains the python script
SCRIPT_PATH=$(cd "$(dirname "$0")"; pwd -P)

# Change into the directory that contains the python script
cd "$SCRIPT_PATH"

# Run the Python script
python demo/premiere_mmpose.py webcam --pose2d ipr_res50_8xb64-210e_coco-256x256

# Deactivate the Conda environemnt
conda deactivate