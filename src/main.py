import argparse
import yaml

from utils import Utils

from annotator import Annotator
from annotator.corpora import BaseCorpus
from utils import download_from_PMC
from normalizer import Normalizer
from indexer import Indexer

import traceback
import glob
import os

def read_settings(settings_file):    
    with open(settings_file) as f:
        return yaml.safe_load(f)

def cli_settings_override(args, settings):
    """
    Override the specific settings with the cli args
    
    Not implemented...
    """
    return settings
    
def print_current_configuration(settings, tab=""):
    if tab=="":
        print()
        print("Settings:")
        print_current_configuration(settings, tab="\t")
        print()
    else:
        for k,v in settings.items():
            if isinstance(v, dict):
                print(f"{tab}{k}:")
                print_current_configuration(v, tab=tab+"\t")
            else:
                print(tab,k,"=",v)

    
def load_corpus(corpus_path, 
                ignore_non_contiguous_entities, 
                ignore_normalization_identifiers,
                solve_overlapping_passages):
    
    if os.path.isdir(corpus_path):
        # load corpus
        corpus = {f"{os.path.splitext(os.path.basename(file))[0]}":file for file in glob.glob(os.path.join(corpus_path, "*.json"))}
    elif os.path.splitext(corpus_path)[1]==".json":
        corpus = {f"{os.path.splitext(os.path.basename(corpus_path))[0]}":corpus_path}
    elif corpus_path.startswith("PMC"):
        
        try:
            corpus_path = download_from_PMC(corpus_path)
        except:
            traceback.print_exc()
            print()
            print("The download of the PMC didn't returned a valid json object, please check the above stack trace for more information")
            exit()
            
        # call the same method but now with the downloaded .json as file
        return load_corpus(corpus_path,
                           ignore_non_contiguous_entities, 
                           ignore_normalization_identifiers,
                           solve_overlapping_passages)
    else:
        raise ValueError(f"found {corpus_path} as the path. However only folders, json and PMCID are supported")
    
    base_corpus = BaseCorpus(corpus,
                         ignore_non_contiguous_entities=ignore_non_contiguous_entities,
                         ignore_normalization_identifiers=ignore_normalization_identifiers,
                         solve_overlapping_passages=solve_overlapping_passages)
    
    return base_corpus

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('source_directory',
                        type=str,
                        help='')
    
    configs = parser.add_argument_group('Global settings', 'This settings are related with the location of the files and directories.')
    configs.add_argument('-s', '--settings', dest='settings', \
                        type=str, default="src/settings.yaml", \
                        help='The system settings file (default: settings.yaml)')    
    configs.add_argument('-a', '--annotator', default=False, action='store_true', \
                         help='Flag to annotate the files (default: False)')
    configs.add_argument('-n', '--normalizer', default=False, action='store_true', \
                            help='Flag to normalize the detected concepts (default: False)')
    configs.add_argument('-i', '--indexer', default=False, action='store_true', \
                            help='Flag to index the detected concepts (default: False)')
    
    args = parser.parse_args()
    
    if not args.annotator and \
       not args.normalizer and \
       not args.indexer:
        # by default lets assume that we want to run the full pipeline!
        args.annotator, args.normalizer, args.indexer = True, True, True
    
    if (args.annotator, args.normalizer, args.indexer) in {(True, False, True)}:
        print("It is not possible to run the indexer after the annotator module in this pipeline. Any other configuration is valid. ")
        exit()
    
    # read the default settings
    settings = read_settings(args.settings)
    settings = cli_settings_override(args, settings)
    print_current_configuration(settings)
    
    # load to baseCorpus
    next_module_input = load_corpus(args.source_directory, **settings["ReadCollectionParams"])
    
    pipeline = [class_name(**settings[class_name.__name__]) for class_name, init in ((Annotator, args.annotator), (Normalizer, args.normalizer), (Indexer, args.indexer)) if init]
    
    for module in pipeline:
        next_module_input = module.transform(next_module_input)
        
        
