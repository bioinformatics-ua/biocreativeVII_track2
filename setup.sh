#
# Exit when any command fails.
#
set -e

#
# Download the NLM-Chem, NLM-Chem-Test, CDR, and CHEMDNER datasets.
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

if [ ! -f "$NLMCHEM_GZ_FILE" ]; then
    echo "Download $NLMCHEM_GZ_FILE"
    wget $URL_NLMCHEM_GZ_FILE
fi

echo "Untar $NLMCHEM_GZ_FILE"
tar -xf $NLMCHEM_GZ_FILE

cd -

cd "datasets/NLMChemTest"

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

#
# Get a PMCID-PMID mapping for the NLM-Chem and NLM-Chem-Test datasets.
#

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

mkdir -p "datasets/DrugProt"
mkdir -p "datasets/DrugProtFiltered"

cd "datasets/DrugProt"

DRUGPROT_ZIP_FILE="drugprot-training-development-test-background.zip"
URL_DRUGPROT_ZIP_FILE="https://zenodo.org/record/5119892/files/$DRUGPROT_ZIP_FILE"

wget $URL_DRUGPROT_ZIP_FILE
unzip $DRUGPROT_ZIP_FILE

cd -

cd "src/scripts"
python3 convert_drugprot_to_json.py ../../datasets/DrugProt/drugprot-gs-training-development/training/
python3 convert_drugprot_to_json.py ../../datasets/DrugProt/drugprot-gs-training-development/development/

cd -

mv datasets/DrugProt/drugprot-gs-training-development/DrugProt-training.json datasets/DrugProt/
mv datasets/DrugProt/drugprot-gs-training-development/DrugProt-development.json datasets/DrugProt/

mv datasets/DrugProt/drugprot-gs-training-development/DrugProtFiltered-training.json datasets/DrugProtFiltered/
mv datasets/DrugProt/drugprot-gs-training-development/DrugProtFiltered-development.json datasets/DrugProtFiltered/

#
# TODO: download the CTD chemical vocabulary file.
#

#
# TODO: download trained model weights.
#
