import os
os.environ["POLUS_JIT"]="false"
os.environ["TOKENIZERS_PARALLELISM"]="false"

import argparse
import yaml

from utils import Utils

from annotator.base import Annotator
from annotator.corpora import BaseCorpus
from normalizer.base import Normalizer
from indexer import Indexer
from core import load_corpus

import traceback
import glob

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
        
        
