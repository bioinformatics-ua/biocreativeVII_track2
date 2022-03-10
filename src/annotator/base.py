from core import IModule

from polus.models import load_model
from polus.data import DataLoader, CachedDataLoader, build_bert_embeddings, DataLoader

from annotator import modelsv2
from annotator.data import short_checkpoint_names, bertseq_center_generator, document_generator, tokseqconcat_generator, SequenceDecoder
from annotator.preprocessing import PUBMEDBERT_FULL, Tokenizer
from annotator.utils import write_collections_to_file
from annotator.corpora import BaseCorpus

import tensorflow as tf
import numpy as np

def convert_to_numpy(sample):
    return {k:v.numpy() for k,v in sample.items()}

class Annotator(IModule):
    
    def __init__(self, 
                 model_checkpoint,
                 write_output,
                 write_tags_output,
                 write_path,
                 cache_context_embeddings,
                 batch_size,
                 ):
        super().__init__()
        self.write_output = write_output
        self.write_tags_output = write_tags_output
        self.write_path = write_path
        self.cache_context_embeddings = cache_context_embeddings
        self.batch_size = batch_size
        
        # load the neural model
        self.model = load_model(model_checkpoint, external_module=modelsv2)
        self.cfg = self.model.savable_config
    
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
    
    def transform(self, base_corpus):
        
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
                np.save(f'{self.write_path}/{group}-docs_samples.npy', samples, allow_pickle=True, fix_imports=True)
        
        collections = sequence_decoder.get_collections()
        ## writhe the collections to disk
        if self.write_output:
            write_collections_to_file(collections, name = self.write_path)
            
        return BaseCorpus.from_dict(collections)[0]
