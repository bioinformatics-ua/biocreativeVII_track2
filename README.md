# BioCreative VII Track 2

#### NLM-Chem track: full-text chemical identification and indexing in PubMed articles

[This repository](https://github.com/bioinformatics-ua/biocreativeVII_track2) presents our system for the
[NLM-Chem track](https://biocreative.bioinformatics.udel.edu/tasks/biocreative-vii/track-2/).


### Installation

First, make sure you have [Anaconda](https://www.anaconda.com/products/individual) installed.
Then, create the `biocreative` conda environment with the Python 3.6.9 version and install the dependencies:

```
$ conda create --name biocreative python=3.6.9
$ conda activate biocreative
$ pip install -r requirements.txt
```

Alternatively, if you have Python 3.6 installed on your system you can create a [Python virtual environment](https://docs.python.org/3/library/venv.html).

```
$ python3.6 -m venv biocreative
$ source biocreative/bin/activate
$ python -m pip install --upgrade pip
$ pip install -r requirements.txt
```

Finally, execute the `setup.sh` file in order to download and prepare the required data files: (1) the [NLM-Chem, CDR, and CHEMDNER](https://ftp.ncbi.nlm.nih.gov/pub/lu/BC7-NLM-Chem-track/) datasets; (2) the [DrugProt](https://doi.org/10.5281/zenodo.5119892) dataset; (3) the [CTD chemical vocabulary](http://ctdbase.org/downloads/#allchems), and (4) pre-trained model weights for straightforward inference.

```
$ ./setup.sh
```

### Installation using docker

Additionally, we also provide a docker image [bioinformaticsua/biocreative:1.1.0](https://hub.docker.com/r/bioinformaticsua/biocreative) with all the dependencies installed and ready to run, for instance, consider executing:

```
docker run -it bioinformaticsua/biocreative:1.1.0
```

### How to run

By default the pipeline will perform the annotation, normalization and indexing of a given PMC documents or collection of documents following the BioC format, as presented below:

- Given a PMC identifier
	```
	$ python src/main.py PMC8524328
	```
	The pipeline will try to download the article with the identifier PMC8524328 and store it under the `datasets` folder, then it will proceed to the annotation, normalization and indexing, outputing the resulting files under the `outputs` folder.


- Given a Bio.json file that may contain more that one article:
	```
	$ python src/main.py datasets/NLMChem/BC7T2-NLMChem-corpus-train.BioC.json
	```
- Given a folder that contains BioC.json files:
	```
	$ python src/main.py datasets/NLMChem
	```
	Here the pipeline will individually run each of the BioC files found under the given directory.
    
Note: It is advisible the availability of a GPU for speeding up the annotation procedure.

### Documentation

By default the pipeline will use all the configurations under the `src/settings.yaml` file. However it is also possible to hot changing some of the configurations by using the command line interface.

TODO COMPLETE


### Team
  * Tiago Almeida<sup id="a1">[1](#f1)</sup>
  * Rui Antunes<sup id="a1">[1](#f1)</sup>
  * João F. Silva<sup id="a1">[1](#f1)</sup>
  * João R. Almeida<sup id="a1">[1](#f1)</sup> <sup id="a2">[2](#f2)</sup>
  * Sérgio Matos<sup id="a1">[1](#f1)</sup>

1. <small id="f1"> University of Aveiro, Department of Electronics, Telecommunications and Informatics (DETI / IEETA), Aveiro, Portugal </small> [↩](#a1)
2. <small id="f2"> University of A Coruña, Department of Information and Communications Technologies, A Coruña, Spain </small> [↩](#a2)


### Cite

Please cite our publication, if you use this code in your work:

```bib
to do
```
