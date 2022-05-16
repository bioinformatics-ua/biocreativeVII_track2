from core import IModule

from polus.models import load_model
from polus.data import DataLoader, CachedDataLoader, build_bert_embeddings, DataLoader

from annotator import modelsv2
from annotator.data import short_checkpoint_names, bertseq_center_generator, document_generator, tokseqconcat_generator, SequenceDecoder
from annotator.preprocessing import PUBMEDBERT_FULL, Tokenizer
from annotator.utils import write_collections_to_file
from annotator.corpora import BaseCorpus, Collection
from annotator.elements import span_overlaps_span

from collections import defaultdict
from copy import deepcopy

import subprocess
import os
import tensorflow as tf
import gc
import numpy as np

def convert_to_numpy(sample):
    return {k:v.numpy() for k,v in sample.items()}

class Annotator(IModule):
    
    def __init__(self, 
                 model_checkpoint,
                 write_output,
                 write_tags_output,
                 write_path,
                 write_add_checkpoint_name,
                 cache_context_embeddings,
                 majority_voting_mode,
                 batch_size,
                 ):
        super().__init__()
        self.write_output = write_output
        self.write_tags_output = write_tags_output
        self.write_add_checkpoint_name = write_add_checkpoint_name
        
        self.write_path = write_path
        self.cache_context_embeddings = cache_context_embeddings
        self.batch_size = batch_size
        
        self.model_checkpoint = model_checkpoint
        
        self.majority_voting_mode = majority_voting_mode
    
    def build_dataloaders(self, base_corpus):
        print("Building dataloaders")
        
        PATH_CACHE = "/backup/cache_biocreative_extension_track2/"
        
        get_bert_embeddings = build_bert_embeddings(**self.cfg)
            
        def heavy_batch_map_function(data):

            output = get_bert_embeddings(input_ids = data["input_ids"], 
                                         token_type_ids = data["token_type_ids"],
                                         attention_mask = data["attention_mask"])

            data["embeddings"] = output["last_hidden_state"]

            return data

        check_point_name = short_checkpoint_names[self.cfg["embeddings"]["checkpoint"]]
        _index = self.cfg["embeddings"]["bert_layer_index"]
        
        dataloaders = {}

        for group, corpus in base_corpus:

            ## Building the DataLoader
            gen = bertseq_center_generator(
                        tokseqconcat_generator(
                            document_generator(corpus),
                            tokenizer=Tokenizer(model_name=PUBMEDBERT_FULL)
                        )
                    )     
            if self.cache_context_embeddings:
                dataloader = CachedDataLoader(gen, 
                                              cache_chunk_size = 512,
                                              show_progress = True,
                                              #cache_folder = os.path.join(PATH_CACHE, "dataloaders"),
                                              cache_additional_identifier = f"index{_index}_{check_point_name}",
                                              accelerated_map_f=heavy_batch_map_function)
            else:
                dataloader = DataLoader(gen,
                                        show_progress = True,
                                        accelerated_map_f=heavy_batch_map_function)
            dataloaders[group] = dataloader
            
        return dataloaders
    
    def single_inference(self, base_corpus, model_checkpoint):
        
        # load the neural model
        if self.write_add_checkpoint_name:
            self.suffix = f"_{os.path.splitext(os.path.basename(model_checkpoint))[0]}"
            
        self.model = load_model(model_checkpoint, external_module=modelsv2)
        self.cfg = self.model.savable_config
        npy_files_names = []
        
        sequence_decoder = SequenceDecoder([base_corpus])
        
        dataloaders = self.build_dataloaders(base_corpus)
        
        for group, dataloader in dataloaders.items():
            n_samples = dataloader.get_n_samples()
            
            test_ds = dataloader.batch(self.batch_size)\
                                .prefetch(tf.data.AUTOTUNE)
            
            samples = []

            for step, sample in enumerate(test_ds):
                print(f"{step*self.batch_size}", end = "\r")
                _sample = {
                    "corpus": sample["corpus"],
                    "group": sample["group"],
                    "identifier": sample["identifier"],
                    "spans": sample["spans"][:,self.cfg["model"]["low"]:self.cfg["model"]["high"]],
                    "is_prediction": sample["is_prediction"][:,self.cfg["model"]["low"]:self.cfg["model"]["high"]],
                    "tags_int": tf.cast(sample["tags_int"][:,self.cfg["model"]["low"]:self.cfg["model"]["high"]], tf.int32),
                }

                _sample["tags_int_pred"] = self.model.inference(sample["embeddings"], sample["attention_mask"])

                sequence_decoder.samples_from_batch(_sample)
                
                if self.write_tags_output:
                    samples.append(convert_to_numpy(_sample))

            # save the predicts to be used during ensemble
            if len(samples)>0:
                npy_name = f'{self.write_path}/{group}{self.suffix}-docs_samples.npy'
                np.save(npy_name, samples, allow_pickle=True, fix_imports=True)
                npy_files_names.append(npy_name)
        
        
        collections = sequence_decoder.get_collections()
        ## writhe the collections to disk
        if self.write_output:
            files_names = write_collections_to_file(collections, name = self.write_path, suffix=self.suffix)
            
        return BaseCorpus.from_dict(collections)[0]#, files_names, npy_files_names
    
    def majority_voting_entity_level(self, input_collections, THRESHOLD=0.5):

        n_collections = len(input_collections)
        first_collection = input_collections[0]
        doc_ids = first_collection.ids()

        for c in input_collections:
            assert doc_ids == c.ids(), 'The provided Collections do not share exactly the same Documents Identifiers.'

        new_collection = deepcopy(first_collection)
        new_collection.clear_entities()
        for di in doc_ids:
            hash_to_ne = dict()
            for c in input_collections:
                for ne in c[di].nes():
                    ne_hash = hash(ne)
                    if ne_hash not in hash_to_ne:
                        hash_to_ne[ne_hash] = {'ne': ne, 'count': 0}
                    hash_to_ne[ne_hash]['count'] += 1

            ne_count_list = [v for k, v in hash_to_ne.items()]
            #
            # Sort the Normalized Entities according to the following three
            # criteria:
            # (1) Start offset. Entities that appear first in the text have
            #     higher priority.
            # (2) Entity span length. Larger spans have higher priority.
            # (3) Counts. The number of collections in which the entity exists.
            #
            sorted_ne_count_list = sorted(ne_count_list, key=lambda x: x['ne'].start)
            sorted_ne_count_list = sorted(sorted_ne_count_list, key=lambda x: x['ne'].n_characters, reverse=True)
            sorted_ne_count_list = sorted(sorted_ne_count_list, key=lambda x: x['count'], reverse=True)

            accepted_entities = list()
            for ne_count in sorted_ne_count_list:
                ne = ne_count['ne']
                count = ne_count['count']
                ratio = count / n_collections
                if ratio >= THRESHOLD:
                    #
                    # Make sure the entity does not overlap with existing
                    # entities.
                    #
                    overlap = False
                    for ae in accepted_entities:
                        if span_overlaps_span(ne.span, ae.span):
                            overlap = True
                            break
                    if not overlap:
                        accepted_entities.append(ne)

            #
            # Add the accepted entities to the respective passage.
            #
            for ae in accepted_entities:
                added = False
                for p in new_collection[di]:
                    if span_overlaps_span(ae.span, p.span):
                        p.add_entity(ae)
                        added = True
                assert added, 'Entity {} does not overlap with any passage in document {}.'.format(ae, repr(di))
        return new_collection

    def majority_voting(self, base_corpus_ner):
        
        if self.write_add_checkpoint_name:
            models_name = [os.path.splitext(os.path.basename(model_name))[0] for model_name in self.model_checkpoint]
            self.suffix = "_"+ "_".join(models_name)
        
        print("\tPerforming majority voting ensemble")
        if self.majority_voting_mode == "entity-level":
            print("\t\tMode: entity-level")
            inputs_collections = defaultdict(list)
            collections = defaultdict(dict)
            
            for base_corpus in base_corpus_ner:
                for group, corpus in base_corpus:
                    inputs_collections[group].append(corpus)
                    
            for group, base_corpus_model_predictions in inputs_collections.items():
                new_collection = self.majority_voting_entity_level(base_corpus_model_predictions)
                collections[new_collection.corpus][new_collection.group] = new_collection
            
            if self.write_output:
                write_collections_to_file(collections, name = self.write_path, suffix=self.suffix)
                    
            return BaseCorpus.from_dict(collections)[0]
                
            
        elif self.majority_voting_mode == "tag-level":
            print("Mode: tag-level")
            
            
        else:
            raise ValueError("Majority voting method can only be entity-level or tag-level")
    
    def transform(self, base_corpus):
        
        base_corpus_ner = []
        
        for model_checkpoint in self.model_checkpoint:
            output_base_corpus = self.single_inference(base_corpus, model_checkpoint)
            base_corpus_ner.append(output_base_corpus)
            # reset tf session and call GC
            tf.keras.backend.clear_session()
            gc.collect()
            
        if len(self.model_checkpoint)>1:
            # run ensemble
            output_base_corpus = self.majority_voting(base_corpus_ner)
        
        return output_base_corpus
