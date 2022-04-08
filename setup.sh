#!/bin/bash

#
# Exit when any command fails.
#
set -e

#
# Download the NLMChem, NLMChemTest, CDR, and CHEMDNER datasets.
#

NLMCHEM_GZ_FILE="BC7T2-NLMChem-corpus_v2.BioC.json.gz"
URL_NLMCHEM_GZ_FILE="https://ftp.ncbi.nlm.nih.gov/pub/lu/BC7-NLM-Chem-track/$NLMCHEM_GZ_FILE"

NLMCHEMTEST_GZ_FILE="BC7T2-NLMChemTest-corpus_v1.BioC.json.gz"
URL_NLMCHEMTEST_GZ_FILE="https://ftp.ncbi.nlm.nih.gov/pub/lu/BC7-NLM-Chem-track/test/$NLMCHEMTEST_GZ_FILE"

CDR_GZ_FILE="BC7T2-CDR-corpus_v2.BioC.json.gz"
URL_CDR_GZ_FILE="https://ftp.ncbi.nlm.nih.gov/pub/lu/BC7-NLM-Chem-track/$CDR_GZ_FILE"

CHEMDNER_GZ_FILE="BC7T2-CHEMDNER-corpus_v2.BioC.json.gz"
URL_CHEMDNER_GZ_FILE="https://ftp.ncbi.nlm.nih.gov/pub/lu/BC7-NLM-Chem-track/$CHEMDNER_GZ_FILE"

mkdir -p "datasets/NLMChem"
mkdir -p "datasets/NLMChemTest"
mkdir -p "datasets/CDR"
mkdir -p "datasets/CHEMDNER"

cd "datasets/NLMChem"

echo "Download $NLMCHEM_GZ_FILE"
wget -nc $URL_NLMCHEM_GZ_FILE

echo "Untar $NLMCHEM_GZ_FILE"
tar -xf $NLMCHEM_GZ_FILE

cd -

cd "datasets/NLMChemTest"

echo "Download $NLMCHEMTEST_GZ_FILE"
wget -nc $URL_NLMCHEMTEST_GZ_FILE

echo "Unzip $NLMCHEMTEST_GZ_FILE"
gzip -dkf $NLMCHEMTEST_GZ_FILE

cd -

cd "datasets/CDR"

echo "Download $CDR_GZ_FILE"
wget -nc $URL_CDR_GZ_FILE

echo "Untar $CDR_GZ_FILE"
tar -xf $CDR_GZ_FILE

cd -

cd "datasets/CHEMDNER"

echo "Download $CHEMDNER_GZ_FILE"
wget -nc $URL_CHEMDNER_GZ_FILE

echo "Untar $CHEMDNER_GZ_FILE"
tar -xf $CHEMDNER_GZ_FILE

cd -

#
# Get a PMCID-PMID mapping for the NLMChem and NLMChemTest datasets.
#

echo "Create a PMCID-PMID mapping for the NLMChem and NLMChemTest datasets."

cd "src/scripts"
python3 get_nlmchem_nlmchemtest_pmids.py
mv nlmchem_pmcid_pmid.tsv ../../datasets/NLMChem
mv nlmchemtest_pmcid_pmid.tsv ../../datasets/NLMChemTest
cd -

#
# Download the DrugProt dataset. Only the training and development
# subsets will be used, because we do not known which documents in the
# test subset contain the ground-truth annotations.
#

DRUGPROT_ZIP_FILE="drugprot-training-development-test-background.zip"
URL_DRUGPROT_ZIP_FILE="https://zenodo.org/record/5119892/files/$DRUGPROT_ZIP_FILE"

mkdir -p "datasets/DrugProt"
mkdir -p "datasets/DrugProtFiltered"

cd "datasets/DrugProt"

echo "Download $DRUGPROT_ZIP_FILE"
wget -nc $URL_DRUGPROT_ZIP_FILE

echo "Unzip $DRUGPROT_ZIP_FILE"
unzip -u $DRUGPROT_ZIP_FILE

cd -

echo "Convert the DrugProt training and development subsets to JSON format."

cd "src/scripts"
python3 convert_drugprot_to_json.py ../../datasets/DrugProt/drugprot-gs-training-development/training/
python3 convert_drugprot_to_json.py ../../datasets/DrugProt/drugprot-gs-training-development/development/

cd -

mv datasets/DrugProt/drugprot-gs-training-development/DrugProt-training.json datasets/DrugProt/
mv datasets/DrugProt/drugprot-gs-training-development/DrugProt-development.json datasets/DrugProt/

mv datasets/DrugProt/drugprot-gs-training-development/DrugProtFiltered-training.json datasets/DrugProtFiltered/
mv datasets/DrugProt/drugprot-gs-training-development/DrugProtFiltered-development.json datasets/DrugProtFiltered/

#
# Download additional required data:
#
# (1) ctdbase/
#     The CTD chemical vocabulary file. Required for creating the
#     "NLMChemSyn" dataset (synthetic dataset).
#
# (2) evaluation/
#     The official evaluation script.
#
# (3) mesh/
#     MeSH dictionaries, and pre-trained SapBERT embeddings.
#     Required for the normalization subtask.
#
# (4) model_checkpoint/
#     Pre-trained model checkpoint for the entity recognition task.
#
# (5) tools/
#     External tools (NCBITextLib and Ab3P) required for the
#     normalization subtask.
#
DATA_ZIP_FILE="data.zip"
URL_DATA_ZIP_FILE="https://medstore1.myqnapcloud.com/share.cgi?ssid=41e5a2a9b1854105b69c26f4f8f94f62&fid=41e5a2a9b1854105b69c26f4f8f94f62&filename=$DATA_ZIP_FILE&openfolder=forcedownload&ep="

echo "Download $DATA_ZIP_FILE"
wget -nc -O data.zip $URL_DATA_ZIP_FILE

echo "Unzip $DATA_ZIP_FILE"
unzip -u data.zip

#
# Extract the official evaluation script.
#
unzip -u -d evaluation/BC7T2-evaluation_v3/ evaluation/BC7T2-evaluation_v3.zip

#
# Extract and compile the NCBITextLib and Ab3P tools.
#
unzip -u -d tools/ tools/NCBITextLib.zip
unzip -u -d tools/ tools/Ab3P.zip

cd tools/NCBITextLib/lib/
make
cd -

NCBITEXTLIB="NCBITEXTLIB=\"$(pwd)/tools/NCBITextLib\""

sed -i "1s|.*|$NCBITEXTLIB|" tools/Ab3P/Makefile
sed -i "1s|.*|$NCBITEXTLIB|" tools/Ab3P/lib/Makefile

cd tools/Ab3P/
make
cd -

echo "Setup finished successfully!"
