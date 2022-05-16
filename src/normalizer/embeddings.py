import tensorflow as tf
import pickle
import os
from transformers import TFBertModel, BertTokenizerFast

def build_mesh_embedding_encoder():
    
    SAPBERT = 'cambridgeltl/SapBERT-from-PubMedBERT-fulltext'
    
    print("Load Tokenizer")
    tokenizer = BertTokenizerFast.from_pretrained(SAPBERT)
    
    print("Load BERT")
    bert_model = TFBertModel.from_pretrained(SAPBERT, 
                                             output_attentions = False,
                                             output_hidden_states = False,
                                             return_dict=True,
                                             from_pt=True)
    
    
    @tf.function
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
    
    return get_mesh_embedding

def normalize_vector(x):
    return x/tf.norm(x, ord=2)

def build_normalized_index(index_mesh_info, get_mesh_embedding, normalize_vectors=True):

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

def load_index(index_base_name):

    with open(os.path.join(f"{index_base_name}.matrix"), "rb") as f:
        emb_matrix = pickle.load(f)

    with open(os.path.join(f"{index_base_name}.mindex"), "rb") as f:
        normalized_mesh_index = pickle.load(f)
            
    return normalized_mesh_index, emb_matrix

def mesh_by_similarity(mesh, emb_matrix, get_mesh_embedding, normalize=True, top_k=10):
    mesh_vector = get_mesh_embedding(mesh)
    
    if normalize:
        mesh_vector = normalize_vector(mesh_vector)
    
    distances = tf.squeeze(emb_matrix @ tf.transpose(mesh_vector))

    return tf.math.top_k(
        distances, k=top_k, sorted=True, name=None
    )


def normalizer_by_threshold_v2(collection, normalized_emb_matrix, normalized_mesh_index, get_mesh_embedding, CACHE_RESULT, threshold=0.9, diff_threshold = 0.1):
    
    j=0
    for doc_id, doc in collection:
        print(j," - doc_id ",doc_id,end="\r")
        j+=1
        for passage in doc:
            for i,entity in enumerate(passage.entities()):
                
                # [-, mesh, mesh] 
                if entity.identifiers==["-"]:
                    
                    if entity.text not in CACHE_RESULT:
                        
                        values, indices = mesh_by_similarity(entity.text, normalized_emb_matrix, get_mesh_embedding, top_k=5, normalize=True)
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
