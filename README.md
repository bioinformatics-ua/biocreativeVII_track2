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

Additionally, we also provide a GPU-ready docker image [bioinformaticsua/biocreative:1.1.0](https://hub.docker.com/r/bioinformaticsua/biocreative) with all the dependencies installed and ready to run, for instance, consider executing:

```
docker run -it --rm bioinformaticsua/biocreative:1.1.0
```

Note that by using the flag "--rm" the container and its data will be wiped by the docker

### How to run 

By default the pipeline will perform the annotation, normalization and indexing of given PMC documents or collection of documents following the BioC format, as presented below:

- Given a PMC identifier
	```
	$ python src/main.py PMC8524328
	```
	The pipeline will try to download the article with the identifier PMC8524328 and store it under the `datasets` folder, then it will proceed to the annotation, normalization and indexing, outputting the resulting files under the `outputs` folder.


- Given a BioC.json file that may contain more than one article:
	```
	$ python src/main.py datasets/NLMChem/BC7T2-NLMChem-corpus-train.BioC.json
	```
- Given a folder that contains BioC.json files:
	```
	$ python src/main.py datasets/NLMChem
	```
	Here the pipeline will individually run each of the BioC files found under the given directory.

Furthermore, it is also possible to run each module separately or combined by specifying the flags `-a`  or `--annotator`, `-n`  or `--normalizer`, `-i`  or `--indexer`, which enable the utilization of its respective module. By default when none is specified the program will use all of the modules. For instance, it is possible to perform only annotation by specifying the flag `-a`.  

```
$ python src/main.py PMC8524328 -a
```
    
Additionally, it is also possible to combine the modules and for instance do the annotation following the normalization step, for that the flags `-a` and `-n` must be specified as so:

```
$ python src/main.py PMC8524328 -a -n
```

The same is also valid if we want to just run the normalizer module followed by the indexing module, the difference is that now the input file must be in BioC format with annotation, which is the type of file that is produced by the annotator module.

```
$ python src/main.py outputs/annotator/BaseCorpus_BioC_PMC8524328.json -n -i
```

Note: It is advisable the use a GPU for speeding up the annotation procedure.

### How to train the annotator (pre-train and finetune)

To train a new annotator model go to the `src/annotator` folder and execute the command `cli_train_script.py`. The script has several parameters and configuration settings use `--help` to see all of the options. For instance, to pretrain a model with the exact same configuration of the `model_checkpoint/avid-dew-5.cfg` run:

```
$ python cli_train_script.py -epoch 20 -gaussian_noise 0.15 -use_crf_mask -use_fulltext -base_lr 0.00015 -batch_size 32 -rnd_seed 5 -train_datasets CDR CHEMDNER DrugProt -train_w_test -wandb "[Extension-Revision] Biocreative Track2 NER - pretrain CCD(train, dev, test)"
```

Assume that the model resulting from the above pretrain has the name `pretrain-avid-dew-5.cfg`, then for finetuning the `avid-dew-5.cfg` we run the same script but instead of randomly initializing a new model we load the previous pretrain model specified under the flag `-from_model`

```
$ python cli_train_script.py -from_model pretrain-avid-dew-5.cfg -epoch 20 -gaussian_noise 0.15 -random_augmentation noise -use_crf_mask -use_fulltext -base_lr 0.0001 -batch_size 32 -rnd_seed 1 -train_datasets NLMCHEM -train_w_test -wandb "[Extension-Revision] Biocreative Track2 NER - pretrain CCD(train-dev-test) ft (train-dev-test)"
```

Additionally, it is recommended the use of a GPU and due to a memory leak presented in the `tf.Dataset.shuffle` method, it would be beneficial to run with [TCMalloc](https://google.github.io/tcmalloc/overview.html).

### Documentation

#### Configuration file

By default, the pipeline will use all the configurations under the `src/settings.yaml` file. However, it is also possible to hot changing some of the configurations by using the command-line interface. For instance, if we want to change the default entity-level majority voting method to tag-level it can be done by changing it in the `src/settings.yaml` file or by directly passing it as an optional parameter like so:

```
$ python src/main.py PMC8524328 --annotator.majority_voting_mode tag-level
```
    
Furthermore, all of the model parameters under the `src/settings.yaml` file can be "hot changed" in the CLI by following the pattern `--{module name}.{property} {value}`.

#### Directory organization

All of the developed code resides under the `src` directory, which it is also subdivided into the three modules (annotator, normalizer and indexer) and a folder with some useful scripts. Then the folder `datasets` contains the BioC.json files of all of the datasets used for training and evaluation. Furthermore, new BioC files that are automatically downloaded by giving their id are stored under `datasets/PMC`. The `outputs` folder stores all of the outputs produced by the main pipeline and it is also divided according to the three modules. The `model_checkpoint` folder stores the already trained models that are used for annotation. The `mesh` folder contains the mesh-related files for the normalization process and the resulting embeddings representation for each mesh term. Finally, the `evaluation` folder contains the official NLMChem evaluation scripts.


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
