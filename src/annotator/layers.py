#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow_addons.text import crf_log_likelihood, crf_decode


class CRF(tf.keras.layers.Layer):
    """
    #
    # Code from:
    # https://github.com/tensorflow/addons/issues/1769
    # 
    # Update by Tiago Almeida for recent versions of TF and some simplifications
    """
    def __init__(self,
                 output_dim,
                 sparse_target=True,
                 **kwargs):
        """    
        Args:
            output_dim (int): the number of labels to tag each temporal input.
            sparse_target (bool): whether the the ground-truth label represented in one-hot.
        Input shape:
            (batch_size, sentence length, output_dim)
        Output shape:
            (batch_size, sentence length, output_dim)
        """
        super().__init__(**kwargs)
        self.output_dim = int(output_dim) 
        self.input_spec = tf.keras.layers.InputSpec(min_ndim=3)
        self.sequence_lengths = None
        self.transitions = None

    def build(self, input_shape):
        assert len(input_shape) == 3
        f_shape = tf.TensorShape(input_shape)
        input_spec = tf.keras.layers.InputSpec(min_ndim=3, axes={-1: f_shape[-1]})

        if f_shape[-1] is None:
            raise ValueError('The last dimension of the inputs to `CRF` '
                             'should be defined. Found `None`.')
        if f_shape[-1] != self.output_dim:
            raise ValueError('The last dimension of the input shape must be equal to output'
                             ' shape. Use a linear layer if needed.')
        self.input_spec = input_spec
        self.transitions = self.add_weight(name='transitions',
                                           shape=[self.output_dim, self.output_dim],
                                           initializer='glorot_uniform',
                                           trainable=True)
        super().build(input_shape)

    def call(self, inputs, sequence_lengths=None, training=None, **kwargs):
        sequences = tf.convert_to_tensor(inputs, dtype=self.dtype)
        if sequence_lengths is not None:
            assert len(sequence_lengths.shape) == 2
            assert tf.convert_to_tensor(sequence_lengths).dtype == 'int32'
            seq_len_shape = tf.convert_to_tensor(sequence_lengths).get_shape().as_list()
            assert seq_len_shape[1] == 1
            self.sequence_lengths = K.flatten(sequence_lengths)
        else:
            self.sequence_lengths = tf.ones(tf.shape(inputs)[0], dtype=tf.int32) * (
                tf.shape(inputs)[1]
            )

        viterbi_sequence, _ = crf_decode(sequences,
                                         self.transitions,
                                         self.sequence_lengths)
        
        output = tf.one_hot(viterbi_sequence, self.output_dim )
        
        return tf.keras.backend.in_train_phase(sequences, output)

    @property
    def loss(self):

        def crf_loss(y_true, y_pred):

            log_likelihood, _ = crf_log_likelihood(
                y_pred,
                tf.argmax(y_true, axis=-1, output_type=tf.dtypes.int32),
                self.sequence_lengths,
                transition_params=self.transitions,
            )
            
            return tf.reduce_mean(-log_likelihood)
        return crf_loss
    
    @property
    def loss_sample_weights(self):

        def crf_loss(y_true, y_pred):

            log_likelihood, _ = crf_log_likelihood(
                y_pred,
                tf.argmax(y_true, axis=-1, output_type=tf.dtypes.int32),
                self.sequence_lengths,
                transition_params=self.transitions,
            )
            
            per_sample = tf.reduce_sum(2* y_true * [0,0,1,0], axis=[-2,-1]) + tf.reduce_sum(y_true * [0,0,0,1], axis=[-2,-1])
            sample_weight = tf.math.log(per_sample+1)
            
            loss_per_sample = -log_likelihood * sample_weight
            
            return tf.reduce_mean(loss_per_sample)
        return crf_loss

    def compute_output_shape(self, input_shape):
        tf.TensorShape(input_shape).assert_has_rank(3)
        return input_shape[:2] + (self.output_dim,)

    def get_config(self):
        config = {
            'output_dim': self.output_dim,
            'sparse_target': self.sparse_target,
            'supports_masking': self.supports_masking,
            'transitions': K.eval(self.transitions)
        }
        base_config = super(CRF, self).get_config()
        return dict(base_config, **config)