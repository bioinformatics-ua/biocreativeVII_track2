#!/bin/bash

# exit ehen any command fails
set -e

GZ_FILE="BC7T2-NLMChem-corpus_v2.BioC.json.gz"
URL_GZ_FILE="https://ftp.ncbi.nlm.nih.gov/pub/lu/BC7-NLM-Chem-track/$GZ_FILE"

# DOWNLOAD SOME BIG FILES THAT ARE REQUIRED

if [ ! -d "local/datasets/NLM-Chem" ]; then
    mkdir -p "local/datasets/NLM-Chem"
fi

cd "local/datasets/NLM-Chem"

if [ ! -f "$GZ_FILE" ]; then
	wget $URL_GZ_FILE
fi	     

# untar
tar -xf $GZ_FILE

cd -

# PYTHON DEPENDENCIES
PYTHON=python3.6
VENV_NAME="biocreative"

echo "Creating a python environment ($VENV_NAME)"
$PYTHON -m venv $VENV_NAME

PYTHON=$(pwd)/$VENV_NAME/bin/python
PIP=$(pwd)/$VENV_NAME/bin/pip
IPYTHON=$(pwd)/$VENV_NAME/bin/ipython
# update pip

echo "Updating pip"
$PYTHON -m pip install -U pip

echo "Installing python requirements"
$PIP install -r requirements.txt

# ADD to jupyter
$IPYTHON kernel install --name "$VENV_NAME" --user