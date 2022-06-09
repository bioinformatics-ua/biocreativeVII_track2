# BioCreative VII — Track 2 (NLM-Chem)


### Full-text chemical identification and indexing in PubMed articles

[This repository](https://github.com/bioinformatics-ua/biocreativeVII_track2) presents our system for the [NLM-Chem track](https://biocreative.bioinformatics.udel.edu/tasks/biocreative-vii/track-2/).


## Standard installation

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

Finally, execute the `setup.sh` file in order to download and prepare the required data files: (1) the [NLM-Chem, CDR, and CHEMDNER](https://ftp.ncbi.nlm.nih.gov/pub/lu/BC7-NLM-Chem-track/) datasets; (2) the [DrugProt](https://doi.org/10.5281/zenodo.5119892) dataset; (3) the [CTD chemical vocabulary](http://ctdbase.org/downloads/#allchems); (4) the [official evaluation script](https://ftp.ncbi.nlm.nih.gov/pub/lu/BC7-NLM-Chem-track/); (5) [MeSH](https://www.nlm.nih.gov/mesh/meshhome.html)-related files; (6) external tools, [NCBITextLib](https://github.com/ncbi-nlp/NCBITextLib) and [Ab3P](https://github.com/ncbi-nlp/Ab3P), required for the entity normalization subtask; and (7) pre-trained models for the entity recognition and normalization subtasks.

```
$ ./setup.sh
```


## Alternative installation using Docker

As another option, we also provide a GPU-ready Docker image [bioinformaticsua/biocreative:1.1.0](https://hub.docker.com/r/bioinformaticsua/biocreative) with all the dependencies installed and ready to run. For instance, consider executing:

```
$ docker run -it --runtime=nvidia --rm bioinformaticsua/biocreative:1.1.0
```

Note that by using the flag `--rm` the container and its data will be wiped out by Docker.


## Usage


### How to run the pipeline (inference only)

By default, the pipeline performs chemical identification (annotation and normalization) and chemical indexing of (1) a given [PMC document](https://www.ncbi.nlm.nih.gov/research/bionlp/APIs/BioC-PMC/), or (2) a set of documents, in the `BioC.json` format, as presented below. We note that, in this context, _annotation_ refers to named entity recognition (NER).

- Given a PMC identifier:

    ```
    $ python src/main.py PMC8524328
    ```

    The pipeline downloads the full-text article with the identifier `PMC8524328` and saves it in the `datasets/` directory. Then, it performs the annotation, normalization, and indexing subtasks, outputting the prediction files in the `outputs/` directory.


- Given a `BioC.json` file that may contain multiple articles:

    ```
    $ python src/main.py datasets/NLMChem/BC7T2-NLMChem-corpus-train.BioC.json
    ```

- Given a directory containing `BioC.json` files:

    ```
    $ python src/main.py datasets/NLMChem/
    ```

    In this case the pipeline performs the prediction for each of the `BioC.json` files in the given directory.

Furthermore, it is also possible to run each module (Annotator, Normalizer, or Indexer) separately or combined. This can be specified by using the flags `--annotator`, `--normalizer`, and `--indexer`, or their abbreviated forms `-a`, `-n`, `-i`, which enables the respective modules. By default, when no individual module is specified, the program will use the three modules. For instance, it is possible to perform only annotation (entity recognition) by specifying the flag `-a` as follows:

```
$ python src/main.py PMC8524328 -a
```
    
Another example, for performing annotation followed by the normalization step the flags `-a` and `-n` must be specified:

```
$ python src/main.py PMC8524328 -a -n
```

The same is also valid if we intend to perform only normalization and indexing:

```
$ python src/main.py outputs/annotator/BaseCorpus_BioC_PMC8524328.json -n -i
```

Note that the input file is in the `BioC.json` format and contains entity annotations that were predicted by the Annotator module. We advise the use of a GPU for speeding up the annotation procedure (see below our [System specifications](#system-specifications)).


### How to train (pre-train and finetune) the annotator

To train a new annotator NER model go to the `src/annotator/` directory and run the script `cli_train_script.py`. The script has several parameters and configuration settings (use `--help` to see all of the options). For instance, you can pre-train a model with the exact same configuration of the `model_checkpoint/avid-dew-5.cfg`. Firstly, run:

```
$ python cli_train_script.py -epoch 20 -gaussian_noise 0.15 -use_crf_mask -use_fulltext -base_lr 0.00015 -batch_size 32 -rnd_seed 5 -train_datasets CDR CHEMDNER DrugProt -train_w_test -wandb "[Extension-Revision] Biocreative Track2 NER - pretrain CCD(train, dev, test)"
```

Then, assume that the model resulting from the above pre-training has the name `pretrain-avid-dew-5.cfg`. For finetuning the `avid-dew-5.cfg` model we run the same script but instead of randomly initializing a new model, we load the previous pre-trained model specified in the flag `-from_model`:

```
$ python cli_train_script.py -from_model pretrain-avid-dew-5.cfg -epoch 20 -gaussian_noise 0.15 -random_augmentation noise -use_crf_mask -use_fulltext -base_lr 0.0001 -batch_size 32 -rnd_seed 1 -train_datasets NLMCHEM -train_w_test -wandb "[Extension-Revision] Biocreative Track2 NER - pretrain CCD(train-dev-test) ft (train-dev-test)"
```

We strongly recommend the use of a GPU to train a NER model (see below our [System specifications](#system-specifications)). Also, due to a memory leak present in the `tf.data.Dataset.shuffle` method, it is beneficial to run the script with the [TCMalloc](https://google.github.io/tcmalloc/overview.html) implementation. These is a brief example of use:

```
$ sudo apt-get install libtcmalloc-minimal4
$ LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4 python example.py
```


## Documentation


### Configuration file

By default, the pipeline uses the configurations in the `src/settings.yaml` file. However, it is possible to override them using the command-line interface (CLI). For instance, if we want to change the default NER ensemble method from entity-level to tag-level this can be done (1) by modifying the `src/settings.yaml` file, or (2) directly passing it as an optional parameter:

```
$ python src/main.py PMC8524328 --annotator.majority_voting_mode tag-level
```
    
Furthermore, all the model parameters in the `src/settings.yaml` file can be overridden in the CLI by following the pattern `--{module_name}.{property} {value}`.


### Directory structure

```
.
├── ctdbase/
│       The CTD chemical vocabulary we used for NER data augmentation.
│
├── datasets/
│       The BioC.json files of all the datasets used for training and
│       evaluation. Furthermore, new BioC.json PMC files that are automatically
│       downloaded are saved in the datasets/PMC/ directory.
│
├── evaluation/
│       The official NLM-Chem evaluation script.
│
├── mesh/
│       MeSH-related files for the normalization step and the resulting
│       embeddings representation for each MeSH term.
│
├── model_checkpoint/
│       Already trained (ready for inference) models used for annotation.
│
├── outputs/
│       The prediction files produced by the main pipeline, divided according to
│       the three modules.
│
├── src/
│       The developed code containing sub-directories for the three modules
│       (annotator, normalizer, indexer) and another one with utility scripts.
│
└── tools/
        External tools required for the normalization subtask.
```


## System specifications

For computational reference, our experiments were performed on a server machine with the following characteristics:

- Operating system: Ubuntu 18.04
- CPU: Intel Xeon E5-2630 v4 (40) @ 3.1GHz
- GPU: NVIDIA Tesla K80
- 128 GB RAM


## Team
  * Tiago Almeida<sup id="a1">[1](#f1)</sup>
  * Rui Antunes<sup id="a1">[1](#f1)</sup>
  * João F. Silva<sup id="a1">[1](#f1)</sup>
  * João R. Almeida<sup id="a1">[1](#f1)</sup><sup>, </sup><sup id="a2">[2](#f2)</sup>
  * Sérgio Matos<sup id="a1">[1](#f1)</sup>

1. <small id="f1"> University of Aveiro, Department of Electronics, Telecommunications and Informatics (DETI), Institute of Electronics and Informatics Engineering of Aveiro (IEETA), Aveiro, Portugal </small> [↩](#a1)
2. <small id="f2"> University of A Coruña, Department of Information and Communications Technologies, A Coruña, Spain </small> [↩](#a2)


## Reference

Please cite our paper (unpublished, accepted), if you use this code in your work:

```
@article{almeida2022a,
  author    = {Almeida, Tiago and Antunes, Rui and Silva, Jo{\~a}o F. and Almeida, Jo{\~a}o R. and Matos, S{\'e}rgio},
  journal   = {Database},
  publisher = {{Oxford University Press}},
  title     = {Chemical identification and indexing in {{PubMed}} full-text articles using deep learning and heuristics},
  year      = {2022},
}
```
