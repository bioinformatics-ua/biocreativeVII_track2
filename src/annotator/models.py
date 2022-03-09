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

def load_model(file_name, change_config={}):
    
    with open(file_name,"r") as f:
        cfg = json.load(f)
    
    cfg["model"] = merge_dicts(cfg["model"], change_config)
    
    model = getattr(models, cfg['func_name'])(**cfg)
        
    # load weights
    with h5py.File(file_name.split(".")[0]+".h5", 'r') as f:
        weight = []
        for i in range(len(f.keys())):
            weight.append(f['weight'+str(i)][:])
        model.set_weights(weight)
    
    return model

class NERBertModel(tf.keras.Model, BaseLogger):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # This class also extends BaseLogger, but Keras last subclass do not call super 
        # so it must be manually called
        BaseLogger.__init__(self)
    
    @tf.function(input_signature=[tf.TensorSpec(shape=(None, None, 768), dtype=tf.float32)])
    def inference(self, x):
        self.logger.debug("Inference function was traced")
        return tf.argmax(self(x), axis=-1, output_type=tf.dtypes.int32)
    
class SequentialNERBertModel(tf.keras.Sequential, NERBertModel):
    """
    This class is just to be compatible with the Sequential Model from keras and implements
    the same inference method from the Classifier Model class
    """
    def __init__(self, layers, **kwargs):
        super().__init__(layers, **kwargs)

@from_config
def baselineNER(sequence_length=256, output_classes = 3, **kwargs):
    
    model = SequentialNERBertModel([
        tf.keras.layers.Dense(output_classes, input_shape=(sequence_length, 768))
    ])
    
    return model

@from_config
def baselineMLP_NER(sequence_length=256, output_classes = 3, activation=tf.keras.activations.swish, **kwargs):
    
    activation = resolve_activation(activation)
    
    model = SequentialNERBertModel([
        tf.keras.layers.Dense(128, input_shape=(sequence_length, 768), activation=activation),
        tf.keras.layers.Dense(output_classes)
    ])
    
    return model

@from_config
def baselineContext(sequence_length=256, output_classes = 3, context_window = 3, **kwargs):
    
    model = SequentialNERBertModel([
        tf.keras.layers.Conv1D(output_classes, context_window, padding="SAME",input_shape=(sequence_length, 768)),
    ])
    
    return model

@from_config
def baselineContext_ML(sequence_length=256, output_classes = 3, context_window = 3, activation=tf.keras.activations.swish, **kwargs):
    
    activation = resolve_activation(activation)
    
    model = SequentialNERBertModel([
        tf.keras.layers.Conv1D(128, context_window, padding="SAME",input_shape=(sequence_length, 768), activation=activation),
        tf.keras.layers.Conv1D(output_classes, context_window, padding="SAME"),
    ])
    
    return model

@from_config
def baselineContext_ML_sofmax(sequence_length=256, output_classes = 3, context_window = 3, activation=tf.keras.activations.swish, **kwargs):
    
    activation = resolve_activation(activation)
    
    model = SequentialNERBertModel([
        tf.keras.layers.Conv1D(128, context_window, padding="SAME",input_shape=(sequence_length, 768), activation=activation),
        tf.keras.layers.Conv1D(output_classes, context_window, padding="SAME", activation="softmax"),
        tf.keras.layers.Conv1D(output_classes, context_window, padding="SAME"),
    ])
    
    return model

@from_config
def baselineContext_MLP(sequence_length=256, output_classes = 3, context_window = 3, activation=tf.keras.activations.swish, **kwargs):
    
    activation = resolve_activation(activation)
    
    model = SequentialNERBertModel([
        tf.keras.layers.Conv1D(128, context_window, padding="SAME",input_shape=(sequence_length, 768), activation=activation),
        tf.keras.layers.Dense(output_classes),
    ])
    
    return model

@from_config
def baselineNER_CRF(sequence_length=256, output_classes = 3, **kwargs):
    
    crf_layer = CRF(output_classes)
    
    model = SequentialNERBertModel([
        tf.keras.layers.Dense(output_classes, input_shape=(sequence_length, 768)),
        crf_layer
    ])
    
    model.loss = crf_layer.loss
    
    return model

@from_config
def baselineNER_MLP_CRF(sequence_length=256, 
                        output_classes = 3, 
                        hidden_space = 128,
                        activation=tf.keras.activations.swish, **kwargs):
    
    activation = resolve_activation(activation)
    
    crf_layer = CRF(output_classes)
    
    model = SequentialNERBertModel([
        tf.keras.layers.Dense(hidden_space, input_shape=(sequence_length, 768), activation=activation),
        tf.keras.layers.Dense(output_classes),
        crf_layer
    ])
    
    model.loss = crf_layer.loss
    model.loss_sample_weights = crf_layer.loss_sample_weights
    
    return model

@from_config
def baselineNER_MLP_Dropout_CRF(sequence_length=256, 
                        output_classes = 3, 
                        hidden_space = 128,
                        droupout_p = 0.1,
                        activation=tf.keras.activations.swish, **kwargs):
    
    activation = resolve_activation(activation)
    
    crf_layer = CRF(output_classes)
    
    model = SequentialNERBertModel([
        tf.keras.layers.Dropout(droupout_p, input_shape=(sequence_length, 768)),
        tf.keras.layers.Dense(hidden_space, activation=activation),
        tf.keras.layers.Dense(output_classes),
        crf_layer
    ])
    
    model.loss = crf_layer.loss
    model.loss_sample_weights = crf_layer.loss_sample_weights
    
    return model

@from_config
def baselineContext_ML_CRF(sequence_length=256, output_classes = 3, context_window = 3, activation=tf.keras.activations.swish, **kwargs):
    
    activation = resolve_activation(activation)
    
    crf_layer = CRF(output_classes)
    
    model = SequentialNERBertModel([
        tf.keras.layers.Conv1D(128, context_window, padding="SAME",input_shape=(sequence_length, 768), activation=activation),
        tf.keras.layers.Conv1D(output_classes, context_window, padding="SAME"),
        crf_layer
    ])
    
    model.loss = crf_layer.loss
    
    return model


@from_config
def baselineNER_BILSTM(sequence_length=256, output_classes = 3, **kwargs):
    
    model = SequentialNERBertModel([
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(4, return_sequences=True), input_shape=(sequence_length, 768)),
        tf.keras.layers.Dense(output_classes),
    ])
    
    
    return model


@from_config
def baselineNER_BILSTM_CRF(sequence_length=256, output_classes = 3, **kwargs):
    
    crf_layer = CRF(output_classes)
    
    model = SequentialNERBertModel([
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(4, return_sequences=True, activation=tf.keras.activations.swish ), input_shape=(sequence_length, 768)),
        tf.keras.layers.Dense(output_classes),
        crf_layer
    ])
    
    model.loss = crf_layer.loss
    
    return model

@from_config
def baselineNER_BILSTM_MUL(sequence_length=256, output_classes = 3, **kwargs):
    
    model = SequentialNERBertModel([
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(output_classes, return_sequences=True), merge_mode="mul", input_shape=(sequence_length, 768)),
    ])
    
    
    return model


@from_config
def baselineNER_BILSTM_MUL_CRF(sequence_length=256, output_classes = 3, **kwargs):
    
    crf_layer = CRF(output_classes)
    
    model = SequentialNERBertModel([
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(output_classes, return_sequences=True), merge_mode="mul", input_shape=(sequence_length, 768)),
        crf_layer
    ])
    
    model.loss = crf_layer.loss
    
    return model

@from_config
def BERT_MLP_CRF(checkpoint, 
                 sequence_length=256, 
                 output_classes = 4,
                 hidden_space=128,
                 low=0,
                 high=512,
                 activation=tf.keras.activations.swish, 
                 **kwargs):
    
    activation = resolve_activation(activation)
    
    #layers
    hidden_states = tf.keras.Input((512, 768))
    attention_mask = tf.keras.Input((512))
    
    bert_model = TFBertModel.from_pretrained(checkpoint,
                                             output_attentions = False,
                                             output_hidden_states = True,
                                             return_dict=True,
                                             from_pt=True)
    
    bert_last_layer = bert_model.layers[0].encoder.layer[-1]
    
    def global_attention_mask(x):
        # This codes mimics the transformer BERT implementation: https://huggingface.co/transformers/_modules/transformers/modeling_tf_bert.html
        
        extended_attention_mask = x[:, tf.newaxis, tf.newaxis, :]

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = tf.cast(extended_attention_mask, tf.float32)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        
        return extended_attention_mask
    
    global_attention_mask_layer = tf.keras.layers.Lambda(global_attention_mask)
    
    def cut_embeddings(x):
        return x[:,low:high,:]
    
    cut_layer = tf.keras.layers.Lambda(cut_embeddings)
    
    mlp_1 = tf.keras.layers.Dense(hidden_space, input_shape=(sequence_length, 768), activation=activation)
    mlp_2 = tf.keras.layers.Dense(output_classes)
    crf_layer = CRF(output_classes)
    
    #build
    extended_attention_mask = global_attention_mask_layer(attention_mask)
    embeddings = bert_last_layer(hidden_states = hidden_states, attention_mask = extended_attention_mask, head_mask=None, output_attentions=None)[0]
    embeddings = cut_layer(embeddings)
    h = mlp_1(embeddings)
    logits = mlp_2(h)
    crf_logits = crf_layer(logits)
    
    class NERPreTrainBertModel(NERBertModel):
        
        def __init__(self, **kwargs):
            super().__init__( **kwargs)
        
        # override of inference method
        @tf.function(input_signature=[tf.TensorSpec(shape=(None, None, 768), dtype=tf.float32), tf.TensorSpec(shape=(None, None), dtype=tf.int32)])
        def inference(self, embeddings, attention_mask):
            print("CUSTOM INFERENCE WAS TRACED")
            return tf.argmax(self([embeddings, attention_mask]), axis=-1, output_type=tf.dtypes.int32)
    
    model = NERPreTrainBertModel(inputs=[hidden_states, attention_mask], outputs=[crf_logits])
    
    model.loss = crf_layer.loss
    model.loss_sample_weights = crf_layer.loss_sample_weights
    
    return model

@from_config
def BERT_MLP_DROPOUT_CRF(checkpoint, 
                 sequence_length=256, 
                 output_classes = 4,
                 hidden_space=128,
                 low=0,
                 high=512,
                 activation=tf.keras.activations.swish, 
                 dropout_p=0.1,
                 **kwargs):
    
    #layers
    hidden_states = tf.keras.Input((512, 768))
    attention_mask = tf.keras.Input((512))
    
    bert_model = TFBertModel.from_pretrained(checkpoint,
                                             output_attentions = False,
                                             output_hidden_states = True,
                                             return_dict=True,
                                             from_pt=True)
    
    bert_last_layer = bert_model.layers[0].encoder.layer[-1]
    
    def global_attention_mask(x):
        # This codes mimics the transformer BERT implementation: https://huggingface.co/transformers/_modules/transformers/modeling_tf_bert.html
        
        extended_attention_mask = x[:, tf.newaxis, tf.newaxis, :]

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = tf.cast(extended_attention_mask, tf.float32)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        
        return extended_attention_mask
    
    global_attention_mask_layer = tf.keras.layers.Lambda(global_attention_mask)
    
    def cut_embeddings(x):
        return x[:,low:high,:]
    
    cut_layer = tf.keras.layers.Lambda(cut_embeddings)
    dropout_layer = tf.keras.layers.Dropout(dropout_p)
    
    mlp_1 = tf.keras.layers.Dense(hidden_space, input_shape=(sequence_length, 768), activation=activation)
    mlp_2 = tf.keras.layers.Dense(output_classes)
    crf_layer = CRF(output_classes)
    
    #build
    extended_attention_mask = global_attention_mask_layer(attention_mask)
    embeddings = bert_last_layer(hidden_states = hidden_states, attention_mask = extended_attention_mask, head_mask=None, output_attentions=None)[0]
    embeddings = cut_layer(embeddings)
    embeddings = dropout_layer(embeddings)
    h = mlp_1(embeddings)
    logits = mlp_2(h)
    crf_logits = crf_layer(logits)
    
    class NERPreTrainBertModel(NERBertModel):
        
        def __init__(self, **kwargs):
            super().__init__( **kwargs)
        
        # override of inference method
        @tf.function(input_signature=[tf.TensorSpec(shape=(None, None, 768), dtype=tf.float32), tf.TensorSpec(shape=(None, None), dtype=tf.int32)])
        def inference(self, embeddings, attention_mask):
            print("CUSTOM INFERENCE WAS TRACED")
            return tf.argmax(self([embeddings, attention_mask]), axis=-1, output_type=tf.dtypes.int32)
    
    model = NERPreTrainBertModel(inputs=[hidden_states, attention_mask], outputs=[crf_logits])
    
    model.loss = crf_layer.loss
    model.loss_sample_weights = crf_layer.loss_sample_weights
    
    return model


@from_config
def BERT_MLP_DROPOUT_CRF_V2(checkpoint,
                            bert_layer_index=-1,
                            sequence_length=256, 
                            output_classes = 4,
                            hidden_space=128,
                            low=0,
                            high=512,
                            activation=tf.keras.activations.swish, 
                            dropout_p=0.1,
                            mask_impossible_transitions=None,
                            **kwargs):
    
    #layers
    hidden_states = tf.keras.Input((512, 768), dtype=tf.float32)
    attention_mask = tf.keras.Input((512), dtype=tf.int32)
    
    post_model = split_bert_model_from_checkpoint(checkpoint, 
                                                  bert_layer_index, 
                                                  init_models=False,
                                                  return_pre_bert_model=False,
                                                  return_post_bert_model=True)
    
    def cut_embeddings(x):
        return x[:,low:high,:]
    
    cut_layer = tf.keras.layers.Lambda(cut_embeddings)
    dropout_layer = tf.keras.layers.Dropout(dropout_p)
    
    mlp_1 = tf.keras.layers.Dense(hidden_space, input_shape=(sequence_length, 768), activation=activation)
    mlp_2 = tf.keras.layers.Dense(output_classes)
    crf_layer = CRF(output_classes, mask_impossible_transitions=mask_impossible_transitions)
    
    #build
    embeddings = post_model(hidden_states = hidden_states, attention_mask = attention_mask)["last_hidden_state"]
    embeddings = cut_layer(embeddings)
    embeddings = dropout_layer(embeddings)
    h = mlp_1(embeddings)
    logits = mlp_2(h)
    crf_logits = crf_layer(logits)
    
    class NERPreTrainBertModel(NERBertModel):
        
        def __init__(self, **kwargs):
            super().__init__( **kwargs)
        
        # override of inference method
        @tf.function(input_signature=[tf.TensorSpec(shape=(None, None, 768), dtype=tf.float32), 
                                      tf.TensorSpec(shape=(None, None), dtype=tf.int32)],
                     jit_compile=get_jit_compile())
        def inference(self, embeddings, attention_mask):
            print("CUSTOM INFERENCE WAS TRACED")
            return tf.argmax(self([embeddings, attention_mask], training=False), axis=-1, output_type=tf.dtypes.int32)
    
    model = NERPreTrainBertModel(inputs=[hidden_states, attention_mask], outputs=[crf_logits])
    
    model.loss = crf_layer.loss
    model.loss_sample_weights = crf_layer.loss_sample_weights
    
    return model

@from_config
def LinearCLS_BERT(**kwargs):
    
    class SequentialCLSBertModel(SequentialNERBertModel):
        def __init__(self, layers, **kwargs):
            super().__init__(layers, **kwargs)
        
        @tf.function(input_signature=[tf.TensorSpec(shape=(None, 768), dtype=tf.float32)])
        def inference(self, x):
            print("CUSTOM INFERENCE WAS TRACED")
            return self(x)
        
    return SequentialCLSBertModel([
        tf.keras.layers.Dense(768, input_shape=(768,))
    ])