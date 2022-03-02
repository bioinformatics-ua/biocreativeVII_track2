#!/bin/bash

#
# Exit when any command fails.
#
set -e

#
# Download NLM-Chem, NLM-Chem-Test, CDR, and CHEMDNER datasets.
# Create required directories.
#

NLMCHEM_GZ_FILE="BC7T2-NLMChem-corpus_v2.BioC.json.gz"
URL_NLMCHEM_GZ_FILE="https://ftp.ncbi.nlm.nih.gov/pub/lu/BC7-NLM-Chem-track/$NLMCHEM_GZ_FILE"

NLMCHEMTEST_GZ_FILE="BC7T2-NLMChemTest-corpus_v1.BioC.json.gz"
URL_NLMCHEMTEST_GZ_FILE="https://ftp.ncbi.nlm.nih.gov/pub/lu/BC7-NLM-Chem-track/test/$NLMCHEMTEST_GZ_FILE"

CDR_GZ_FILE="BC7T2-CDR-corpus_v2.BioC.json.gz"
URL_CDR_GZ_FILE="https://ftp.ncbi.nlm.nih.gov/pub/lu/BC7-NLM-Chem-track/$CDR_GZ_FILE"

CHEMDNER_GZ_FILE="BC7T2-CHEMDNER-corpus_v2.BioC.json.gz"
URL_CHEMDNER_GZ_FILE="https://ftp.ncbi.nlm.nih.gov/pub/lu/BC7-NLM-Chem-track/$CHEMDNER_GZ_FILE"

# download a trained model

if [ ! -d "datasets/NLM-Chem" ]; then
    mkdir -p "datasets/NLM-Chem"
fi

if [ ! -d "datasets/NLM-Chem-Test" ]; then
    mkdir -p "datasets/NLM-Chem-Test"
fi

if [ ! -d "datasets/CDR" ]; then
    mkdir -p "datasets/CDR"
fi

if [ ! -d "datasets/CHEMDNER" ]; then
    mkdir -p "datasets/CHEMDNER"
fi

cd "datasets/NLM-Chem"

if [ ! -f "$NLMCHEM_GZ_FILE" ]; then
    echo "Download $NLMCHEM_GZ_FILE"
    wget $URL_NLMCHEM_GZ_FILE
fi

echo "Untar $NLMCHEM_GZ_FILE"
tar -xf $NLMCHEM_GZ_FILE

cd -

cd "datasets/NLM-Chem-Test"

if [ ! -f "$NLMCHEMTEST_GZ_FILE" ]; then
    echo "Download $NLMCHEMTEST_GZ_FILE"
    wget $URL_NLMCHEMTEST_GZ_FILE
fi

echo "Unzip $NLMCHEMTEST_GZ_FILE"
gzip -dkf $NLMCHEMTEST_GZ_FILE

cd -

cd "datasets/CDR"

if [ ! -f "$CDR_GZ_FILE" ]; then
    echo "Download $CDR_GZ_FILE"
    wget $URL_CDR_GZ_FILE
fi

echo "Untar $CDR_GZ_FILE"
tar -xf $CDR_GZ_FILE

cd -

cd "datasets/CHEMDNER"

if [ ! -f "$CHEMDNER_GZ_FILE" ]; then
    echo "Download $CHEMDNER_GZ_FILE"
    wget $URL_CHEMDNER_GZ_FILE
fi

echo "Untar $CHEMDNER_GZ_FILE"
tar -xf $CHEMDNER_GZ_FILE

cd -

## TODO Add the drug prot here

#
# Python dependencies.
#
PYTHON=python3.6
VENV_NAME="bc-venv"

echo "Creating a python environment ($VENV_NAME)"
$PYTHON -m venv $VENV_NAME

PYTHON=$(pwd)/$VENV_NAME/bin/python
PIP=$(pwd)/$VENV_NAME/bin/pip
IPYTHON=$(pwd)/$VENV_NAME/bin/ipython

#
# Update pip.
#
echo "Updating pip"
$PYTHON -m pip install -U pip

echo "Installing python requirements"
$PIP install -r requirements.txt #-U

$PIP install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.4.0/en_core_sci_sm-0.4.0.tar.gz



