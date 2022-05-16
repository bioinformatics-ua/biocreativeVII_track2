import sys
sys.path.extend(['../../tools/BC7T2-evaluation_v3/', '../../src'])

from main import read_settings
from core import load_corpus
from indexer.base import Indexer

from evaluate import get_annotations_from_JSON, get_annotations_from_path, evaluation_config, do_strict_eval

import optuna
import subprocess
import argparse
import collections
import json

def build_searching_func(normalizer_path, gs_path):
    
    print("Read settings")
    settings = read_settings("../settings.yaml")
    settings["Indexer"]["write_output"] = False
    
    print("Load normalize predictions")
    input_corpus = load_corpus(normalizer_path, **settings["ReadCollectionParams"])
    
    evaluation_type = "identifier"
    evaluation_method = "strict"
    annotation_type = "MeSH_Indexing_Chemical"
    
    print("Build eval config")
    eval_config = evaluation_config(annotation_type, evaluation_type)

    reference_annotations, reference_passages = get_annotations_from_path(gs_path, eval_config)
    
    print("Build done")
    def get_indexer_evaluation(trial):
        
        #min_occur_captions = trial.suggest_float("min_occur_captions",0, 1) # X
        
        #min_occur_abstract = trial.suggest_float("min_occur_abstract",0, 1-min_occur_captions) if (1-min_occur_captions)>0 else 0
       
        #min_occur_title = trial.suggest_float("min_occur_title",0, 1-min_occur_abstract-min_occur_captions) if (1-min_occur_abstract-min_occur_captions)>0 else 0
        
        #min_occur_concl = 1-min_occur_abstract-min_occur_captions-min_occur_title
        settings["Indexer"]["min_occur_captions"] = trial.suggest_float("min_occur_captions",0, 1)
        settings["Indexer"]["min_occur_abstract"] = trial.suggest_float("min_occur_abstract",0, 1)
        settings["Indexer"]["min_occur_title"] = trial.suggest_float("min_occur_title",0, 1)
        settings["Indexer"]["min_occur_concl"] = trial.suggest_float("min_occur_concl",0, 1)

        #print(min_occur_captions + min_occur_abstract + min_occur_title + min_occur_concl)
        
        indexer = Indexer(**settings["Indexer"])
        output_corpus = indexer.transform(input_corpus)
        
        ##
        ## evaluation starts here
        ##

        
        for group, collection in output_corpus:

            predicted_annotations, predicted_passages = get_annotations_from_JSON(json.loads(collection.pretty_json()), "blank", eval_config)
            
            eval_result = do_strict_eval(reference_annotations, predicted_annotations)
        
        print("---------")
        print(eval_result)
        
        return eval_result.f_score
        
    return get_indexer_evaluation
    
    

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("normalizer_path", type=str)
    parser.add_argument("gs_path", type=str)
    args = parser.parse_args()
    
    func = build_searching_func(args.normalizer_path, args.gs_path)
    
    sampler = optuna.samplers.TPESampler(n_startup_trials = 50)
    study = optuna.create_study(sampler=sampler, 
                                direction="maximize",
                                study_name=f'tpe_index_search_v11',
                                load_if_exists=True,
                                storage='sqlite:///optuna.db')

    s = study.optimize(func, n_trials=500)
