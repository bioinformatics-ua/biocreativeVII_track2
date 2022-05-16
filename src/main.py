import os
os.environ["POLUS_JIT"]="false"
os.environ["TOKENIZERS_PARALLELISM"]="false"

import argparse
import yaml

from annotator.base import Annotator
from annotator.corpora import BaseCorpus
from normalizer.base import Normalizer
from indexer.base import Indexer
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
    if args.annotator_model_checkpoint is not None:
        settings["Annotator"]["model_checkpoint"] = args.annotator_model_checkpoint
    if args.annotator_write_path is not None:
        settings["Annotator"]["write_path"] = args.annotator_write_path
    
    if args.normalizer_skip_rule_based is not None:
        settings["Normalizer"]["skip_rule_based"] = args.normalizer_skip_rule_based
    if args.normalizer_write_path is not None:
        settings["Normalizer"]["write_path"] = args.normalizer_write_path
    
    if args.indexer_write_path is not None:
        settings["Indexer"]["write_path"] = args.indexer_write_path
    if args.indexer_min_occur_captions is not None:
        settings["Indexer"]["min_occur_captions"] = args.indexer_min_occur_captions
    if args.indexer_min_occur_abstract is not None:
        settings["Indexer"]["min_occur_abstract"] = args.indexer_min_occur_abstract
    if args.indexer_min_occur_title is not None:
        settings["Indexer"]["min_occur_title"] = args.indexer_min_occur_title
    if args.indexer_min_occur_concl is not None:
        settings["Indexer"]["min_occur_concl"] = args.indexer_min_occur_concl
    if args.indexer_method is not None:
        settings["Indexer"]["method"] = args.indexer_method
    
    
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
    
    annotator_configs = parser.add_argument_group('Annotator settings', 'This settings are related to the indexer module.')
    annotator_configs.add_argument('--annotator.model_checkpoint', dest='annotator_model_checkpoint', \
                                 type=str, default=None, \
                                 help='The annotator model cfg path')
    annotator_configs.add_argument('--annotator.write_path', dest='annotator_write_path', \
                                 type=str, default=None, \
                                 help='Path where to write the model')
    
    normalizer_configs = parser.add_argument_group('Normalizer settings', 'This settings are related to the normalizer module.')
    normalizer_configs.add_argument('--normalizer.skip_rule_based', dest='normalizer_skip_rule_based', \
                                 action='store_true', default=None, \
                                 help='The annotator model cfg path')
    normalizer_configs.add_argument('--normalizer.write_path', dest='normalizer_write_path', \
                                 type=str, default=None, \
                                 help='Path where to write the model')
    
    
    indexer_configs = parser.add_argument_group('Indexer settings', 'This settings are related to the indexer module.')
    indexer_configs.add_argument('--indexer.write_path', dest='indexer_write_path', \
                                 type=str, default=None, \
                                 help='The indexer outputs path')
    indexer_configs.add_argument('--indexer.min_occur_captions', dest='indexer_min_occur_captions', \
                                 type=float, default=None, \
                                 help='The indexer min occur captions')
    indexer_configs.add_argument('--indexer.min_occur_abstract', dest='indexer_min_occur_abstract', \
                                 type=float, default=None, \
                                 help='The indexer min occur abstract')
    indexer_configs.add_argument('--indexer.min_occur_title', dest='indexer_min_occur_title', \
                                 type=float, default=None, \
                                 help='The indexer min occur title')
    indexer_configs.add_argument('--indexer.min_occur_concl', dest='indexer_min_occur_concl', \
                                 type=float, default=None, \
                                 help='The indexer min occur conclusion')
    indexer_configs.add_argument('--indexer.method', dest='indexer_method', \
                                 type=int, default=None, \
                                 help='The indexer method')
    
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
        
        
