from core import IModule, load_corpus

from normalizer.utils import dictionaryLoader, mapWithoutAb3P, mapWithoutAb3P_AugmentedDictionary, mapWithAb3P, mapWithAb3P_AugmentedDictionary
from normalizer.embeddings import load_index, normalizer_by_threshold_v2, build_mesh_embedding_encoder
from annotator.elements import merge_collections
from annotator.utils import write_collections_to_file
from annotator.corpora import BaseCorpus

from collections import defaultdict

import json
import os

class Normalizer(IModule):
    
    def __init__(self, 
                 write_output,
                 write_path,
                 ab3p_path,
                 dictionary_dataset_augmentation,
                 ab3p_abbreviation_expansion,
                 mesh_dictionaries,
                 corpus_for_expansion,
                 embedding_index = None):
        super().__init__()
        self.write_output = write_output
        self.write_path = write_path
        self.ab3p_path = ab3p_path
        self.dictionary_dataset_augmentation = dictionary_dataset_augmentation
        self.ab3p_abbreviation_expansion = ab3p_abbreviation_expansion
        self.ab3p_dict_level = "Corpus"
        self.mesh_dictionaries = dictionaryLoader(mesh_dictionaries)
        self.corpus_for_expansion = corpus_for_expansion
        self.embedding_index = embedding_index
        
        if self.embedding_index is not None:
            # init embeddings normalizer
            self.normalized_mesh_index, self.normalized_emb_matrix = load_index(self.embedding_index)
            self.get_mesh_embedding = build_mesh_embedding_encoder()

    def normalize_collection(self, collection, train_collection):
        meshDictionary = self.mesh_dictionaries
        
        if self.ab3p_abbreviation_expansion:
            if self.dictionary_dataset_augmentation:
                _, _, meshDictionary, abbreviationMap = mapWithAb3P_AugmentedDictionary(self.ab3p_path, train_collection, meshDictionary, self.ab3p_dict_level, test=False)
            else:
                _, _, abbreviationMap = mapWithAb3P(self.ab3p_path, train_collection, meshDictionary, self.ab3p_dict_level)
                
            outputs, mappedDocuments, _ = mapWithAb3P(self.ab3p_path, collection, meshDictionary, self.ab3p_dict_level, abbreviationMap)
        else:
            if self.dictionary_dataset_augmentation:
                _, _, meshDictionary = mapWithoutAb3P_AugmentedDictionary(train_collection, meshDictionary, test=False)
                
            outputs, mappedDocuments = mapWithoutAb3P(collection, meshDictionary)
        
        return outputs, mappedDocuments
    
    def normalize_collection_wembeddings(self, collection, CACHE_RESULT):
        
         return normalizer_by_threshold_v2(collection, 
                                           self.normalized_emb_matrix, 
                                           self.normalized_mesh_index, 
                                           self.get_mesh_embedding,
                                           CACHE_RESULT,
                                           threshold=0.98, 
                                           diff_threshold=0.06)

    
    def transform(self, corpus):

        train_collections = [load_corpus(corpus_path, True, False, False)[os.path.splitext(os.path.basename(corpus_path))[0]] for corpus_path in self.corpus_for_expansion]
        train_collection = merge_collections(*train_collections)
        
        collections = defaultdict(dict)
        embeddings_cache = {}
        
        for group, collection in corpus:
            output_collection, mappedDocuments = self.normalize_collection(collection, train_collection)
            
            if self.embedding_index is not None:
                output_collection = self.normalize_collection_wembeddings(output_collection, CACHE_RESULT=embeddings_cache)
                
            collections[output_collection.corpus][output_collection.group] = output_collection
        
        if self.write_output:
            _suffix = "" if self.embedding_index is None else "_wEmbeddings"
            write_collections_to_file(collections, name = self.write_path, suffix=_suffix)
        
        return BaseCorpus.from_dict(collections)[0]

