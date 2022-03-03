#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from bio import reconstruct_bio
from data import SequenceDecoder, document_level_agreement
from polus.core import BaseLogger
from collections import defaultdict
import tensorflow as tf


class IMetric(BaseLogger):
    def __init__(self):
        super().__init__()
        if self.__class__.__name__ == "IMetric":
            raise Exception("This is an interface that cannot be instantiated")
            
        self.name = self.__class__.__name__
    
    def samples_from_batch(self, samples):
        raise Exception("samples_from_batch was called, but is not implemented")
    
    def reset(self):
        raise Exception("clear was called, but is not implemented")
    
    def _evaluate(self):
        raise Exception("_evaluate was internally called, but is not implemented")
        
    def evaluate(self):
        measure = self._evaluate()
        if isinstance(measure, tf.Tensor):
            measure = measure.numpy()
            
        self.reset()
        
        return measure

    
class EntityF1(IMetric):
    def __init__(self, corpora):
        super().__init__()
        self.sequence_decoder = SequenceDecoder(corpora)
        self.reset()
        
    def samples_from_batch(self, samples):
        self.sequence_decoder.samples_from_batch(samples)
    
    def reset(self):
        self.sequence_decoder.clear_state()
    
    def _evaluate(self):
        return self.sequence_decoder.evaluate_ner()['f1']
    
class EntityF1DocAgreement(IMetric):
    def __init__(self, corpora):
        super().__init__()
        self.sequence_decoder = SequenceDecoder(corpora, doc_majoraty_voting=document_level_agreement(mode="all"))
        self.reset()
        
    def samples_from_batch(self, samples):
        self.sequence_decoder.samples_from_batch(samples)
    
    def reset(self):
        self.sequence_decoder.clear_state()
    
    def _evaluate(self):
        return self.sequence_decoder.evaluate_ner()['f1']
    
    
class EntityReconstructedF1(IMetric):
    def __init__(self, corpora):
        super().__init__()
        self.sequence_decoder = SequenceDecoder(corpora, reconstruct_bio)
        self.reset()
        
    def samples_from_batch(self, samples):
        self.sequence_decoder.samples_from_batch(samples)
    
    def reset(self):
        self.sequence_decoder.clear_state()
    
    def _evaluate(self):
        return self.sequence_decoder.evaluate_ner()['f1']