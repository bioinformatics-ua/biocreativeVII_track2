import configparser
import os 
import traceback
import glob
from utils import download_from_PMC
from annotator.corpora import BaseCorpus

class IModule():
    """
    Module interface describing the 
    """
    def __init__(self):
        super().__init__()
        if self.__class__.__name__ == "IModule":
            raise Exception("This is an interface that cannot be instantiated")
            
    def transform(self, inputs):
        raise NotImplementedError(f"Method transform of {self.__class__.__name__} was not implemented")


class ConfigParserDict(configparser.ConfigParser):
    """
    ini to dict from: https://stackoverflow.com/questions/3220670/read-all-the-contents-in-ini-file-into-dictionary-with-python
    
    changed by: Tiago Almeida
    """
    def read_file(self, f, source=None):
        """
        overload of the read_file method to convert the ini to python dict
        """
        super().read_file(f, source)
        d = dict(self._sections)
        for k in d:
            d[k] = dict(self._defaults, **d[k])
            d[k].pop('__name__', None)
        return d
    
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

