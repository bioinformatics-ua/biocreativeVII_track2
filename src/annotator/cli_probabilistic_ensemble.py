import argparse
import json

import tensorflow as tf
import tensorflow_addons as tfa

import glob
import os

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

from utils import get_temp_file
from data import short_checkpoint_names, bertseq_left_generator, bertseq_center_generator, tokseq_generator, sentence_generator, passage_generator, document_generator, selector_generator, bertseq_left128_generator, SequenceDecoder
from losses import sum_cross_entropy, weighted_cross_entropy, sample_weighted_cross_entropy
from corpora import NLMChemCorpus, CDRCorpus, CHEMDNERCorpus, DrugProtFilteredCorpus, BC5CDRCorpus, CRAFTCorpus, BioNLP11IDCorpus, BioNLP13CGCorpus, BioNLP13PCCorpus
from preprocessing import Tokenizer, PUBMEDBERT_FULL, SAPBERT
from metrics import EntityF1, EntityReconstructedF1

from transformers.optimization_tf import WarmUp, AdamWeightDecay


def dataloader_checkpoint(cfg):
    check_point_name = short_checkpoint_names[cfg["embeddings"]["checkpoint"]]
    _index = cfg["embeddings"]["bert_layer_index"]
    
    return f"index{_index}_{check_point_name}_heavy_batch_map_function_{corpus}_{group}_document_passage_tokseq_bertseq_center.index"

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Ensemble script')
    parser.add_argument("--models", nargs='+', type=str, help="the path for the trained models", required=True)
    
    args = parser.parse_args()
    
    print(f"Running emsemble of {len(args.models)} models")
    
    PATH_CACHE = "/backup/cache_biocreative_extension_track2/"
    
    # needed for the evaluation metrics
    corpus = NLMChemCorpus()
    group = "test"
    
    dataloader = {}
    
    sequence_decoder = SequenceDecoder([corpus])
    _results = {}
    
    samples = {}
    
    for model_path in args.models:
        
        print(" - Running", model_path)
        
        model = load_model(model_path, external_module=modelsv2)
        cfg = model.savable_config
        
        data_path = dataloader_checkpoint(cfg)
        
        if data_path not in dataloader:
            dataloader[data_path] = CachedDataLoader.from_cached_index(os.path.join(PATH_CACHE, "dataloaders", data_path))
        
        
        
        for i,sample in enumerate(dataloader[data_path].batch(64)):
 
            _sample = {
                "corpus": sample["corpus"],
                "group": sample["group"],
                "identifier": sample["identifier"],
                "spans": sample["spans"][:,cfg["model"]["low"]:cfg["model"]["high"]],
                "is_prediction": sample["is_prediction"][:,cfg["model"]["low"]:cfg["model"]["high"]],
                "tags_int": tf.cast(sample["tags_int"][:,cfg["model"]["low"]:cfg["model"]["high"]], tf.int32),
            }
            _sample["tags_int_pred"] = model.inference(sample["embeddings"], sample["attention_mask"])
            sequence_decoder.samples_from_batch(_sample)
            
            if i in samples:
                _sample["tags_int_pred"] = tf.one_hot(_sample["tags_int_pred"], depth=4) + samples[i]["tags_int_pred"]
            else:
                _sample["tags_int_pred"] = tf.one_hot(_sample["tags_int_pred"], depth=4)
            
            samples[i] = _sample

        _results[model_path] = sequence_decoder.evaluate_ner()['f1']
        
        sequence_decoder.clear_state()
        
        # clear the session
        tf.keras.backend.clear_session()
    
    print()
    for _m,_m_f1 in _results.items():
        print(f"Model {_m} got {_m_f1} F1")
    
    for i in range(len(samples)):
        #print(samples[i]["tags_int_pred"])
        samples[i]["tags_int_pred"] = tf.argmax(samples[i]["tags_int_pred"], axis=-1, output_type=tf.int32)
        sequence_decoder.samples_from_batch(samples[i])
    
    _f1 = sequence_decoder.evaluate_ner()['f1']
    print(f"Ensemble of the previous models achieved a F1 score of {_f1}")
    

  