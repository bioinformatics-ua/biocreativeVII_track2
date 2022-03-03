import tensorflow as tf
import tensorflow_addons as tfa

import os
import json
import h5py
import types

#import for refering to this file, used in the load_model method
import models
from polus.layers import CRF
from polus.core import BaseLogger, get_jit_compile
from polus.models import from_config, split_bert_model, split_bert_model_from_checkpoint, SavableModel

from functools import wraps
from polus.utils import merge_dicts

from transformers import TFBertModel


class NERPreTrainBertModel_768(SavableModel):
        def __init__(self, 
                     checkpoint,
                     bert_layer_index,
                     *args,
                     dropout_p = 0.4,
                     hidden_space = 128,
                     sequence_length = 256,
                     output_classes = 4,
                     low= 128, 
                     high = 384,
                     activation = "mish",
                     gaussian_noise = None,
                     mask_impossible_transitions = None,
                     **kwargs):
            
            super().__init__(*args, **kwargs)
            
            # build layers
            if gaussian_noise is not None:
                self.gaussian_noise = tf.keras.layers.GaussianNoise(gaussian_noise)
            else:
                self.gaussian_noise = None

            
            self.post_model = split_bert_model_from_checkpoint(checkpoint, 
                                                              bert_layer_index, 
                                                              init_models=False,
                                                              return_pre_bert_model=False,
                                                              return_post_bert_model=True)
            
            self.dropout_layer = tf.keras.layers.Dropout(dropout_p)
            
            self.low = low
            self.high = high
            
            self.mlp_1 = tf.keras.layers.Dense(hidden_space, input_shape=(sequence_length, 768), activation=activation)
            self.mlp_2 = tf.keras.layers.Dense(output_classes)
            self.crf_layer = CRF(output_classes, mask_impossible_transitions=mask_impossible_transitions)
            self.loss = self.crf_layer.loss
            self.loss_sample_weights = self.crf_layer.loss_sample_weights

        # override of inference method
        @tf.function(input_signature=[tf.TensorSpec(shape=(None, None, 768), dtype=tf.float32), 
                                      tf.TensorSpec(shape=(None, None), dtype=tf.int32)],
                     jit_compile=get_jit_compile())
        def inference(self, embeddings, attention_mask):
            print("CUSTOM INFERENCE WAS TRACED")
            return tf.argmax(self(embeddings=embeddings, attention_mask=attention_mask, training=False), axis=-1, output_type=tf.dtypes.int32)
        
        def call(self, embeddings, attention_mask, training=False):
            
            if self.gaussian_noise is not None:
                embeddings = self.gaussian_noise(embeddings, training=training)
            
            embeddings = self.post_model(hidden_states = embeddings, 
                                         attention_mask = attention_mask, 
                                         training=training)["last_hidden_state"][:,self.low:self.high,:]
            embeddings = self.dropout_layer(embeddings, training=training)
            h = self.mlp_1(embeddings, training=training)
            logits = self.mlp_2(h, training=training)
            crf_logits = self.crf_layer(logits, training=training)
            return crf_logits
            
        
@from_config
def BERT_MLP_DROPOUT_CRF_V2(checkpoint,
                            bert_layer_index=-1,
                            bert_embedding=768,
                            sequence_length=256, 
                            output_classes = 4,
                            hidden_space=128,
                            low=0,
                            high=512,
                            activation=tf.keras.activations.swish, 
                            dropout_p=0.1,
                            gaussian_noise=None,
                            mask_impossible_transitions=None,
                            **kwargs):
    
    if len(kwargs)>0:
        print("[WARNING] The following arguments were given to the model function, but are not been used", kwargs)
    
    if bert_embedding == 768:
        model = NERPreTrainBertModel_768(checkpoint,
                                         bert_layer_index,
                                         sequence_length=sequence_length,
                                         output_classes=output_classes,
                                         hidden_space=hidden_space,
                                         low=low,
                                         high=high,
                                         activation=activation,
                                         dropout_p=dropout_p,
                                         gaussian_noise=gaussian_noise,
                                         mask_impossible_transitions=mask_impossible_transitions,
                                    )
    else:
        raise ValueError(f"There is no valid implementation for that embeddings size, found {bert_embedding}")
        
    if kwargs:
        print(f"The following kwargs args were not used during model initialization")
    
    return model