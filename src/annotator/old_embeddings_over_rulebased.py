import argparse
import json

import tensorflow as tf
import tensorflow_addons as tfa

from corpora import NLMChemCorpus, BaseCorpus
from preprocessing import Tokenizer, SAPBERT
from transformers import TFBertModel, BertTokenizerFast
from evaluation_extra import precision_recall_f1
from collections import defaultdict
import os
import pickle

from utils import write_collections_to_file

def evaluation(collection, normalized_entities):

    tp_total = 0
    fp_total = 0
    fn_total = 0
    p_list = []
    for doc_id, doc in collection:

        gs = {mesh for e in doc.nes() for mesh in e.identifiers}
        
        preds = normalized_entities[doc_id]
        
        n_true_entities = len(gs)
        n_pred_entities = len(preds)
        
        #print(gs, preds, gs&preds)
        #print(n_pred_entities)
        
        tp = len(gs&preds)

        fp = n_pred_entities - tp
        fn = n_true_entities - tp
        
        tp_total += tp
        fp_total += fp
        fn_total += fn

    return precision_recall_f1(tp_total, fp_total, fn_total, return_nan=True)

def mesh_from_file(*files, collections=None):

    index_mesh = {}
    for file in files:
        with open(file) as f:
            for data in json.load(f):
                index_mesh[f'MESH:{data["DescriptorUI"]}'] = data
    
    if collections is not None:
        #diff_mesh = set(collections_mesh.keys())-set(index_mesh.keys())
        #print(len(diff_mesh), sum([collections_mesh[mesh] for mesh in diff_mesh]))
        for collection in (collections + [merge_collections(*collections)]):

            normalized_entities = perfect_normalizer(collection, index_mesh)

            print(evaluation(collection, normalized_entities))
            
    return index_mesh

#@tf.function(input_signature=[tf.TensorSpec(shape=(None, 25, 768), dtype=tf.float32), tf.TensorSpec(shape=(None, 25, 768), dtype=tf.float32), tf.TensorSpec(shape=(None, 25, 768), dtype=tf.float32)])
tf.function
def get_cls(**kwargs):
    return bert_model(kwargs)["last_hidden_state"][:,0,:] # NONE, 512, 768


def get_mesh_embedding(mesh):
    encoding = tokenizer.encode_plus(mesh, 
                                  max_length = 25, 
                                  padding="max_length",
                                  truncation=True,
                                  return_token_type_ids = True,
                                  return_attention_mask = True,
                                  return_tensors = "tf")

    return get_cls(**encoding)

def normalize_vector(x):
    return x/tf.norm(x, ord=2)

def build_normalized_index(index_mesh_info, normalize_vectors=True):

    emb_matrix = []
    normalized_mesh_index = []

    _max = len(index_mesh_info)
    k = 0
    for _id, mesh in index_mesh_info.items():
        if not k%1000:
            print(f"{k}/{_max}", end="\r")
        k+=1
        emb_vector = get_mesh_embedding(mesh["DescriptorName"])

        if normalize_vectors:
            emb_vector = normalize_vector(emb_vector)

        emb_matrix.append(emb_vector)
        normalized_mesh_index.append(_id)

    emb_matrix = tf.concat(emb_matrix, axis=0)

    return normalized_mesh_index, emb_matrix
    
def mesh_by_similarity(mesh, emb_matrix, normalize=True, top_k=10):
    mesh_vector = get_mesh_embedding(mesh)
    
    if normalize:
        mesh_vector = normalize_vector(mesh_vector)
    
    distances = tf.squeeze(emb_matrix @ tf.transpose(mesh_vector))

    return tf.math.top_k(
        distances, k=top_k, sorted=True, name=None
    )

def get_reminder_distances(collection, normalized_emb_matrix, normalize=True):
    distances = defaultdict(dict)
    j = 0
    for doc_id, doc in collection:
        print(j," - doc_id ",doc_id,end="\r")
        j+=1
        for passage in doc:
            for i,entity in enumerate(passage.entities()):
                if entity.identifiers==["-"]:
                
                    values, indices = mesh_by_similarity(entity.text, normalized_emb_matrix, top_k=5, normalize=normalize)
                    values = values.numpy().tolist()
                    indices = indices.numpy().tolist()
                    assert(values[0]>values[1])

                    distances[doc_id][f"{i}_{entity.text}"] = {"values":values, "indices": indices}
    
    return distances


CACHE_RESULT = {}
def normalizer_by_threshold_v2(collection, normalized_emb_matrix, normalized_mesh_index, threshold=0.9, diff_threshold = 0.1):
    

    j=0
    for doc_id, doc in collection:
        print(j," - doc_id ",doc_id,end="\r")
        j+=1
        for passage in doc:
            for i,entity in enumerate(passage.entities()):
                
                # [-, mesh, mesh] 
                if entity.identifiers==["-"]:
                    
                    if entity.text not in CACHE_RESULT:
                        
                        values, indices = mesh_by_similarity(entity.text, normalized_emb_matrix, top_k=5, normalize=True)
                        values = values.numpy().tolist()
                        indices = indices.numpy().tolist()
                        
                        diff_to_second = values[0] - values[1]

                        if diff_to_second > diff_threshold:
                            entity.set_identifiers([normalized_mesh_index[indices[0]]])
                            CACHE_RESULT[entity.text] = [normalized_mesh_index[indices[0]]]
                            #normalized_entities[doc_id].add(normalized_mesh_index[indices[0]])
                            continue

                        _temp_mesh = []
                        for i in range(len(values)):
                            if values[i]>threshold:
                                #normalized_entities[doc_id].add(normalized_mesh_index[indices[i]])
                                _temp_mesh.append(normalized_mesh_index[indices[i]])
                            else:
                                entity.set_identifiers(_temp_mesh)
                                CACHE_RESULT[entity.text] = _temp_mesh
                                break
                    
                    else:
                        entity.set_identifiers(CACHE_RESULT[entity.text])
                    
                    #if len(normalized_entities[doc_id])==0:
                        #entity.set_identifiers(["-"])
                        #normalized_entities[doc_id].add("-")
                    
                #else:
                #    for mesh in entity.identifiers:
                #        normalized_entities[doc_id].add(mesh)

    return collection


def get_normalization(run):
    normalized_entities = defaultdict(set)
    j=0
    for doc_id, doc in run:
        print(j," - doc_id ",doc_id,end="\r")
        j+=1
        for passage in doc:
            for i,entity in enumerate(passage.entities()):
                for mesh in entity.identifiers:
                    normalized_entities[doc_id].add(mesh)
                        
    return normalized_entities

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Embedding normalizer')
    parser.add_argument("base_run", type=str, help="A json file with the base run")
    parser.add_argument("mesh_file", type=str, nargs='+', help="A json file with the configuration of the model")
    
    args = parser.parse_args()
    
    print("Load corpus")
    base_run = BaseCorpus({"test":args.base_run},
                         ignore_non_contiguous_entities=False,
                         ignore_normalization_identifiers=False,
                         solve_overlapping_passages=False)
    
    #base_run_normalization = get_normalization(base_run["test"])

    print("Load Tokenizer")
    tokenizer = BertTokenizerFast.from_pretrained(SAPBERT)
    
    print("Load BERT")
    bert_model = TFBertModel.from_pretrained(SAPBERT, 
                                             output_attentions = False,
                                             output_hidden_states = False,
                                             return_dict=True,
                                             from_pt=True)
    
    print("Building mesh dictionary from the files:",  args.mesh_file)
    index_mesh = mesh_from_file(*args.mesh_file)
    
    
    shorter_names = [os.path.basename(path).split(".")[0] for path in args.mesh_file]
    
    cache_name = "_".join(shorter_names)
    
    if os.path.exists(f"{cache_name}.matrix"):
        print("load norm matrix")
        with open(f"{cache_name}.matrix", "rb") as f:
            normalized_emb_matrix = pickle.load(f)
            
        with open(f"{cache_name}.mindex", "rb") as f:
            normalized_mesh_index = pickle.load(f)
    else:
        print("Build norm matrix")
        normalized_mesh_index, normalized_emb_matrix = build_normalized_index(index_mesh, normalize_vectors=True)
        
        with open(f"{cache_name}.matrix", "wb") as f:
            pickle.dump(normalized_emb_matrix, f)
            
        with open(f"{cache_name}.mindex", "wb") as f:
            pickle.dump(normalized_mesh_index, f)

    augmented_collection = normalizer_by_threshold_v2(base_run["test"], normalized_emb_matrix, normalized_mesh_index, threshold=0.98, diff_threshold=0.06)
    
    #write_collections_to_file(base_run)
    with open(f"emb_runs/embedding_plus_rules_{os.path.basename(args.base_run)}", "w") as f:
        f.write(base_run["test"].pretty_json())