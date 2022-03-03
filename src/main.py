import argparse
import configparser

from Utils import Utils

from annotator import Annotator
from annotator.corpora import BaseCorpus

from normalizer import Normalizer
from indexer import Indexer
from core import ConfigParserDict

def readSettings(settings_file):
    configuration = ConfigParserDict()
    
    with open(settings_file) as f:
        # this will raise if file not exists
        return configuration.read_file(f)
    
def load_corpus(corpus_folder, 
                ignore_non_contiguous_entities, 
                ignore_normalization_identifiers,
                solve_overlapping_passages):
    
    # read files from folder
    print(ignore_non_contiguous_entities, 
                ignore_normalization_identifiers,
                solve_overlapping_passages)
    
    base_corpus = BaseCorpus({"test":corpus_folder},
                         ignore_non_contiguous_entities=ignore_non_contiguous_entities,
                         ignore_normalization_identifiers=ignore_normalization_identifiers,
                         solve_overlapping_passages=solve_overlapping_passages)
    
    return base_corpus
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('source_directory',
                        type=str,
                        help='The system settings file (default: settings.ini)')
    
    configs = parser.add_argument_group('Global settings', 'This settings are related with the location of the files and directories.')
    configs.add_argument('-s', '--settings', dest='settings', \
                        type=str, default="src/settings.ini", \
                        help='The system settings file (default: settings.ini)')    
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
    settings = readSettings(args.settings)
    pipeline = [class_name(**settings[class_name.__name__]) for class_name, init in ((Annotator, args.annotator), (Normalizer, args.normalizer), (Indexer, args.indexer)) if init]
    
    next_module_input = load_corpus(args.source_directory, **settings["Corpus.read.settings"])
    
    # load to baseCorpus
    
    
    for module in pipeline:
        next_module_input = module.transform(next_module_input)

        
    