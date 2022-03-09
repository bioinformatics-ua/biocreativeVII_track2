import argparse
import json

import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np

import glob
import os

# import trainer
# import metrics MacroF1Score, Accuracy, MacroF1ScoreBI, EntityF1
from polus.utils import set_random_seed, complex_json_serializer
from polus.data import CachedDataLoader, build_bert_embeddings, DataLoader
from polus.models import load_model

import modelsv2

from utils import get_temp_file, write_collections_to_file
from data import short_checkpoint_names, bertseq_left_generator, bertseq_center_generator, tokseq_generator, sentence_generator, passage_generator, document_generator, selector_generator, bertseq_left128_generator, tokseqconcat_generator, random_augmentation, ShufflerAugmenter, NoiseAugmenter, SequenceDecoder
from corpora import NLMChemCorpus, NLMChemTestCorpus, CDRCorpus, CHEMDNERCorpus, DrugProtFilteredCorpus, BC5CDRCorpus, CRAFTCorpus, BioNLP11IDCorpus, BioNLP13CGCorpus, BioNLP13PCCorpus, BaseCorpus
from preprocessing import Tokenizer, PUBMEDBERT_FULL, SAPBERT
from metrics import EntityF1, EntityF1DocAgreement
from config import ROOT

def convert_to_numpy(sample):
    return {k:v.numpy() for k,v in sample.items()}

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Inference script')
    parser.add_argument("model_path", type=str, help="the path for the pretrained model")
    parser.add_argument("-o", type=str, default=None, help="output file")
    parser.add_argument("-use_NLMChem", action='store_true', help="Flag to use NLMCHem Test")
    args = parser.parse_args()
    
    BERT_CHECKPOINT = PUBMEDBERT_FULL
    PATH_CACHE = "/backup/cache_biocreative_extension_track2/"
    
    model = load_model(args.model_path, external_module=modelsv2)

    cfg = model.savable_config
    
    if args.use_NLMChem:
        corpus = NLMChemCorpus()
        collection = corpus["test"]
    else:
        corpus = NLMChemTestCorpus()
        collection = corpus["test"]
    
    # pre embeddings computation
    
    get_bert_embeddings = build_bert_embeddings(**cfg)
            
    def heavy_batch_map_function(data):

        output = get_bert_embeddings(input_ids = data["input_ids"], 
                                     token_type_ids = data["token_type_ids"],
                                     attention_mask = data["attention_mask"])
        
        data["embeddings"] = output["last_hidden_state"]

        return data
    
    check_point_name = short_checkpoint_names[cfg["embeddings"]["checkpoint"]]
    _index = cfg["embeddings"]["bert_layer_index"]
    
    ## Building the DataLoader
    gen = bertseq_center_generator(
                tokseqconcat_generator(
                    document_generator(collection),
                    tokenizer=Tokenizer(model_name=BERT_CHECKPOINT)
                )
            )     

    dataloader = CachedDataLoader(gen, 
                                  cache_chunk_size = 512,
                                  show_progress = True,
                                  cache_folder = os.path.join(PATH_CACHE, "dataloaders"),
                                  cache_additional_identifier = f"index{_index}_{check_point_name}",
                                  accelerated_map_f=heavy_batch_map_function)
    
    BATCH = 128
    
    test_ds_NLMCHEM = dataloader.batch(BATCH)\
                                .prefetch(tf.data.AUTOTUNE)
    
    sequence_decoder = SequenceDecoder([corpus])
    
    samples = []
    
    for step, sample in enumerate(test_ds_NLMCHEM):
        print(f"{step*BATCH}", end = "\r")
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
        
        
        samples.append(convert_to_numpy(_sample))
    
    # save the predicts to be used during ensemble
    np.save(f'{args.o}{corpus}-test-docs_samples.npy', samples, allow_pickle=True, fix_imports=True)
    
    collections = sequence_decoder.get_collections()
    
    if args.o is None:
        write_collections_to_file(collections)
    else:
        write_collections_to_file(collections, name = args.o)
    