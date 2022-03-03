import argparse
import json

import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np

import glob
import os
import gc

# import trainer
# import metrics MacroF1Score, Accuracy, MacroF1ScoreBI, EntityF1
from polus.callbacks import ConsoleLogCallback, TimerCallback, LossSmoothCallback, ValidationDataCallback, SaveModelCallback, EarlyStop, WandBLogCallback
from polus.utils import set_random_seed
from polus.data import CachedDataLoader, build_bert_embeddings
from polus.schedulers import warmup_scheduler
from polus.ner.metrics import MacroF1Score, Accuracy
from polus.training import ClassifierTrainer
from polus.models import load_model

import modelsv2

from utils import get_temp_file, write_collections_to_file
from data import short_checkpoint_names, bertseq_left_generator, bertseq_center_generator, tokseq_generator, sentence_generator, passage_generator, document_generator, selector_generator, bertseq_left128_generator, SequenceDecoder
from losses import sum_cross_entropy, weighted_cross_entropy, sample_weighted_cross_entropy
from corpora import NLMChemCorpus, NLMChemTestCorpus, CDRCorpus, CHEMDNERCorpus, DrugProtFilteredCorpus, BC5CDRCorpus, CRAFTCorpus, BioNLP11IDCorpus, BioNLP13CGCorpus, BioNLP13PCCorpus
from preprocessing import Tokenizer, PUBMEDBERT_FULL, SAPBERT
from metrics import EntityF1, EntityReconstructedF1

from transformers.optimization_tf import WarmUp, AdamWeightDecay

def convert_to_tensor(sample):
    return {k:tf.convert_to_tensor(v) for k,v in sample.items()}

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Ensemble script')
    parser.add_argument("--predicts", nargs='+', type=str, help="Numpy file with the predictions", required=True)
    
    args = parser.parse_args()
    
    print(f"Running ensemble of {len(args.predicts)} models")
    
    samples = {}
    name = []
    for predicts_path in args.predicts:
        
        print(" - Loading", predicts_path)
        name.append(predicts_path.split("/")[-1].split("run")[0][:-1])
        docs_pred = np.load(predicts_path, allow_pickle=True, fix_imports=True)
        
        for i, _sample in enumerate(map(convert_to_tensor, docs_pred)):
            
            if i in samples:
                _sample["tags_int_pred"] = tf.one_hot(_sample["tags_int_pred"], depth=4) + samples[i]["tags_int_pred"]
            else:
                _sample["tags_int_pred"] = tf.one_hot(_sample["tags_int_pred"], depth=4)
            
            samples[i] = _sample
            
        del docs_pred
        gc.collect()
    
    sequence_decoder = SequenceDecoder([NLMChemCorpus(), NLMChemTestCorpus()])
    
    number_of_draws = 0
    
    for i in range(len(samples)):
        #print(samples[i]["tags_int_pred"])
        values , _ = tf.math.top_k(samples[i]["tags_int_pred"], 2)
        number_of_draws += tf.reduce_sum(tf.cast(values[:,:,0] == values[:,:,1], tf.int32)).numpy()
        
        samples[i]["tags_int_pred"] = tf.argmax(samples[i]["tags_int_pred"], axis=-1, output_type=tf.int32)
        sequence_decoder.samples_from_batch(samples[i])
    
    print("number of draws", number_of_draws)
    
    name = "|".join(name)
    
    collections = sequence_decoder.get_collections()
    
    _name = f"runs/{name}"
    
    print(f"Writting to {_name}")
    write_collections_to_file(collections, name = _name)

    
    #_f1 = sequence_decoder.evaluate_ner()['f1']
    #print(f"Ensemble of the previous models achieved a F1 score of {_f1}")
    

  