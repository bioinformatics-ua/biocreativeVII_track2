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
    
    for var, value in vars(args).items():
        
        if value is None:
            continue
            
        if var.startswith("annotator_"):
            settings["Annotator"][var[10:]] = value
        elif var.startswith("normalizer_"):
            settings["Normalizer"][var[11:]] = value
        elif var.startswith("indexer_"):
            settings["Indexer"][var[8:]] = value
    
    return settings
    
def print_current_configuration(args, settings, tab=""):
    if tab=="":
        print()
        print("Settings:")
        print_current_configuration(args, settings, tab="\t")
        print()
    else:
        for k,v in settings.items():
            if k=="Annotator" and not args.annotator \
               or k=="Normalizer" and not args.normalizer \
               or k=="Indexer" and not args.indexer:
                # only print the configurations of the modules that will be used
                continue
            if isinstance(v, dict):
                print(f"{tab}{k}:")
                print_current_configuration(args, v, tab=tab+"\t")
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
    annotator_configs.add_argument('--annotator.write_output', dest='annotator_write_output', \
                                 default=None, \
                                 help='Flag that signals if this module would write the produced annotated files')
    annotator_configs.add_argument('--annotator.write_path', dest='annotator_write_path', \
                                 type=str, default=None, \
                                 help='Path where to write the model')
    annotator_configs.add_argument('--annotator.write_add_checkpoint_name', dest='annotator_write_add_checkpoint_name', \
                                 default=None, \
                                 help='Flag that signals if the model_checkpoint name will be appended to the output file name. (boolean parameter)')
    annotator_configs.add_argument('--annotator.cache_context_embeddings', dest='annotator_cache_context_embeddings', \
                                 default=None, \
                                 help='Flag that signals if the contextualized embeddings will be stored in disk. (boolean parameter)')
    annotator_configs.add_argument('--annotator.batch_size', dest='annotator_batch_size', \
                                 default=None, \
                                 help='Number of samples that are fed for the neural bio tagger')
    
    normalizer_configs = parser.add_argument_group('Normalizer settings', 'This settings are related to the normalizer module.')
    normalizer_configs.add_argument('--normalizer.skip_rule_based', dest='normalizer_skip_rule_based', \
                                 default=None, \
                                 help='Flag that signals the use of the rule based normalization')
    normalizer_configs.add_argument('--normalizer.write_path', dest='normalizer_write_path', \
                                 default=None, \
                                 help='Path where to write the normalized corpus')
    normalizer_configs.add_argument('--normalizer.write_output', dest='normalizer_write_output', \
                                 default=None, \
                                 help='Flag that signals if this module would write the produced annotated files')
    normalizer_configs.add_argument('--normalizer.ab3p_path', dest='normalizer_ab3p_path', \
                                 default=None, \
                                 help='Path to the ab3p tool, used for the rule-based normalization')
    normalizer_configs.add_argument('--normalizer.dictionary_dataset_augmentation', dest='normalizer_dictionary_dataset_augmentation', \
                                 default=None, \
                                 help='Flag to do dataset augmentation, used for the rule-based normalization. (boolean parameter)')
    normalizer_configs.add_argument('--normalizer.ab3p_abbreviation_expansion', dest='normalizer_ab3p_abbreviation_expansion', \
                                 default=None, \
                                 help='Flag to perform the ab3p abbreviation. (boolean parameter)')
    
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
    print_current_configuration(args, settings)
    
    # load to baseCorpus
    next_module_input = load_corpus(args.source_directory, **settings["ReadCollectionParams"])
    
    pipeline = [class_name(**settings[class_name.__name__]) for class_name, init in ((Annotator, args.annotator), (Normalizer, args.normalizer), (Indexer, args.indexer)) if init]
    
    for module in pipeline:
        next_module_input = module.transform(next_module_input)
        
        
