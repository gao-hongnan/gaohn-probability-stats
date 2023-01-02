#!/bin/bash
set -e # tell bash to exit on error

echo "Installing JupyterBook for macOS"
echo "--------------------------------"

# Global working vars
SELF=`basename $0`
CONDA=`which conda`
PY_VER=3.8
ENV_NAME="jupyterbook"

conda create -y -n $ENV_NAME python=$PY_VER
conda init bash
source ~/.bash_profile
conda activate $ENV_NAME

conda install -c conda-forge jupyter-book
pip install myst-nb==0.16.0

echo "JupyterBook installation complete"
echo "---------------------------------"
echo "Activate the environment with: conda activate $ENV_NAME"
echo "Exiting $SELF"