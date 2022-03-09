#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from copy import deepcopy
import glob
import inspect
import os

import tensorflow as tf
from transformers import TFBertModel
# from transformers import BatchEncoding
# from tokenizers import Encoding

from annotator.bio import decode_bio
from annotator.bio import get_bio
from annotator.elements import get_entities_within_span, EntitySet, Entity
from annotator.evaluation_extra import eval_list_of_entity_sets
from polus.core import BaseLogger, find_dtype_and_shapes
from annotator.preprocessing import sentence_splitting
from annotator.preprocessing import Tokenizer

import random


from collections import defaultdict
import re

TAG2INT = {
    'PAD': 0,
    'O': 1,
    'B-Chemical': 2,
    'I-Chemical': 3
}
INT2TAG = {i: t for t, i in TAG2INT.items()}


#
# We need to be able to identify the document (corpus, group,
# identifier) because later we need to have access to the original
# text. So we can faithfully recreate the set with the predicted
# entities.
#
def document_generator(collection):
    
    def generator():
        for identifier, document in collection:
            text = document.text()
            entities = document.entities(sort=True)
            passages_texts = document.passages_texts()
            passages_spans = document.passages_spans()
            passages_entities = document.passages_entities(sort=True)
            
            yield {
                'corpus': collection.corpus,
                'group': collection.group,
                'identifier': identifier,
                'text': text,
                'entities': entities,
                'passages_texts': passages_texts,
                'passages_spans': passages_spans,
                'passages_entities': passages_entities,
            }
    
    generator.__name__ = f"{collection.corpus}_{collection.group}_document"
    
    return generator


#
# We want to only consider passages, because annotations beyond a
# passage are ignored in the official evaluation.
#
# To make sure the annotations are inside one passage and do not overlap
# multiple passages.
#
def passage_generator(document_generator):
    def generator():
        for document in document_generator():
            corpus = document['corpus']
            group = document['group']
            identifier = document['identifier']
            
            passages_texts = document['passages_texts']
            passages_spans = document['passages_spans']
            passages_entities = document['passages_entities']
            
            for text, span, entities in zip(passages_texts,
                                            passages_spans,
                                            passages_entities):
                
                yield {
                    'corpus': corpus,
                    'group': group,
                    'identifier': identifier,
                    'passage_text': text,
                    'passage_span': span,
                    'passage_entities': entities,
                }
    
    generator.__name__ = f"{document_generator.__name__}_passage"
    
    return generator


#
# Iterate over the sentences in the passages.
#
def sentence_generator(passage_generator):
    def generator():
        for passage in passage_generator():
            corpus = passage['corpus']
            group = passage['group']
            identifier = passage['identifier']
            
            passage_text = passage['passage_text']
            passage_span = passage['passage_span']
            passage_entities = passage['passage_entities']
            
            offset = passage_span[0]
            sentences, sentences_spans, _ = sentence_splitting(passage_text,
                                                               offset)
            
            for sentence, sentence_span in zip(sentences, sentences_spans):
                
                sentence_entities = get_entities_within_span(passage_entities,
                                                             sentence_span)
                
                yield {
                        'corpus': corpus,
                        'group': group,
                        'identifier': identifier,
                        'sentence_text': sentence,
                        'sentence_span': sentence_span,
                        'sentence_entities': sentence_entities,
                }
    
    generator.__name__ = f"{passage_generator.__name__}_sentence"
    
    return generator


#
# Tokenized sequence generator.
# It can be derived from a passage or sentence generator.
#
def tokseq_generator(passage_or_sentence_generator, tokenizer=Tokenizer()):
    def generator():
        for sequence in passage_or_sentence_generator():
            corpus = sequence['corpus']
            group = sequence['group']
            identifier = sequence['identifier']
            
            if 'passage_text' in sequence:
                text = sequence['passage_text']
                span = sequence['passage_span']
                entities = sequence['passage_entities']
            else:
                text = sequence['sentence_text']
                span = sequence['sentence_span']
                entities = sequence['sentence_entities']
            
            offset = span[0]
            
            tokens, input_ids, token_type_ids, attention_mask, spans = \
                tokenizer.tokenize(text, offset)
            
            tags = get_bio(spans, entities)
            tags_int = [TAG2INT[t] for t in tags]
            
            yield {
                'corpus': corpus,
                'group': group,
                'identifier': identifier,
                'text': text,
                'span': span,
                'entities': entities,
                'tokens': tokens,
                'input_ids' : input_ids,
                'token_type_ids' : token_type_ids,
                'attention_mask' : attention_mask,
                'spans': spans,
                'tags': tags,
                'tags_int': tags_int,
            }
    
    generator.__name__ = f"{passage_or_sentence_generator.__name__}_tokseq"
    
    return generator


#
# Tokenized sequence (concatenated) generator.
# All passages are concatenated: this means that the full-text is used.
# Each token has an associated passage index (number).
# It can be derived from the document generator.
#
def tokseqconcat_generator(document_generator, tokenizer=Tokenizer()):
    def generator():
        for document in document_generator():
            corpus = document['corpus']
            group = document['group']
            identifier = document['identifier']
            
            # document_text = document['text']
            # document_entities = document['entities']
            
            passages_texts = document['passages_texts']
            passages_spans = document['passages_spans']
            passages_entities = document['passages_entities']
            
            tokens_ = []
            input_ids_ = []
            token_type_ids_ = []
            attention_mask_ = []
            spans_ = []
            tags_ = []
            tags_int_ = []
            passage_number_ = []
            
            passage_number = 0
            for text, span, entities in zip(passages_texts,
                                            passages_spans,
                                            passages_entities):
                
                offset = span[0]
                
                tokens, input_ids, token_type_ids, attention_mask, spans = \
                    tokenizer.tokenize(text, offset)
                
                tags = get_bio(spans, entities)
                tags_int = [TAG2INT[t] for t in tags]
                
                tokens_.extend(tokens)
                input_ids_.extend(input_ids)
                token_type_ids_.extend(token_type_ids)
                attention_mask_.extend(attention_mask)
                spans_.extend(spans)
                
                tags_.extend(tags)
                tags_int_.extend(tags_int)
                
                n = len(tokens)
                passage_number_.extend([passage_number] * n)
                
                passage_number += 1
            
            yield {
                'corpus': corpus,
                'group': group,
                'identifier': identifier,
                # 'text': document_text,
                # 'entities': document_entities,
                'tokens': tokens_,
                'input_ids' : input_ids_,
                'token_type_ids' : token_type_ids_,
                'attention_mask' : attention_mask_,
                'spans': spans_,
                'tags': tags_,
                'tags_int': tags_int_,
                'passage_number': passage_number_,
            }
    
    generator.__name__ = f"{document_generator.__name__}_tokseqconcat"
    
    return generator


class IAugmenter(BaseLogger):

    def __init__(self, **kwargs):
        super().__init__( **kwargs)

    def augment(self, sample):
        raise NotImplementedError("method choose should be implemented")    
        
class ShufflerAugmenter(IAugmenter):

    def __init__(self, 
                 k_selections = 3,
                 prob_change_entity = 0.3,
                 prob_change_non_entity = 0.3,
                 **kwargs):
        super().__init__( **kwargs)
        
        self.k_selections = k_selections
        self.prob_change_entity=prob_change_entity
        self.prob_change_non_entity=prob_change_non_entity
        self.prob_non_change = 1 - self.prob_change_entity+self.prob_change_non_entity
        
        assert self.prob_non_change >= 0
        

    def augment(self, sample):
        
        rnd_action = random.choices([0,1,2], [self.prob_non_change, self.prob_change_entity, self.prob_change_non_entity], k=1)[0]
            
        if rnd_action==1:
            # add a random noise to a sub-token that is an entity
            sub_token_idxs = tf.reshape(tf.concat([tf.where(sample["tags_int"]==2),tf.where(sample["tags_int"]==3)], axis=0),-1).numpy().tolist()    
        elif rnd_action==2:
            # add a random noise to a sub-token that isn't an entity
            sub_token_idxs = tf.reshape(tf.where(sample["tags_int"]==1),-1).numpy().tolist()
        else:
            sub_token_idxs = []
            
        sample["embeddings"] = self._random_embeddings_change(sample["embeddings"], sub_token_idxs)
            
        return sample
    
    def _choose_indexes(self, valid_indexes):
        return random.choices(valid_indexes, k=min(self.k_selections, len(valid_indexes)))
    
    def _random_embeddings_change(self, embeddings, valid_indexes):

        if len(valid_indexes)>0:
            #choose one
            sub_token_idxs = self._choose_indexes(valid_indexes)
            
            #convert to numpy bc tf.Tensor does not support item assigment
            embeddings = embeddings.numpy()
            
            for idx in sub_token_idxs:
                embeddings[idx] = self._modify_embedding(embeddings[idx])

        return embeddings
    
    def _modify_embedding(self, embedding):
        return tf.random.shuffle(embedding)
    
class NoiseAugmenter(ShufflerAugmenter):
    # override
    def _modify_embedding(self, embedding):
        # mannually computed statistics
        return tf.random.normal(embedding.shape, mean=0, stddev=0.76, dtype=tf.dtypes.float32, seed=None, name=None)
        
def random_augmentation(data_generator,
                        augmenter: IAugmenter ,
                        samples_per_step=None,
                        shuffle=True,
                        ):

    if samples_per_step is None:
        samples_per_step = data_generator.get_n_samples()
    
    def generator():
        
        i = 0
        
        dl = data_generator.repeat()
        if shuffle:
            dl = dl.shuffle(min(samples_per_step, 30000))
            
        dl_iter = iter(dl)
        
        while i<samples_per_step:
            
            sample = augmenter.augment(next(dl_iter))
            
            yield {
                'corpus': sample["corpus"],
                'group': sample["group"],
                'identifier': sample["identifier"],
                'tokens': sample["tokens"],
                'spans': sample["spans"],
                'tags': sample["tags"],
                'tags_int': sample["tags_int"],
                'passage_number': sample["passage_number"],
                'is_prediction': sample["is_prediction"],
                'embeddings': sample["embeddings"],
                'attention_mask': sample["attention_mask"],
            }
            
            i+=1
    
    generator.__name__ = f"random_augmentation_{data_generator.__name__}"
    
    return generator
    


def prepare_sequence_for_bertseq_left128(sequence, start, end, padding):
    #
    # Guarantee that the input sequence:
    #   (1) has a maximum length of 126 values.
    #
    # Steps to obtain the output sequence:
    #   (1) the "start" value is added at the beginning of the sequence.
    #   (2) the "end" value is added at the end of the sequence.
    #   (3) the sequence is padded at the right to have a length of 512.
    #
    assert len(sequence) <= 126
    
    sequence = [start] + sequence + [end]
    
    remaining = 512 - len(sequence)
    
    sequence = sequence + remaining * [padding]
    
    return sequence


def bertseq_left128_generator(tokseq_generator):
    #
    # We are only considering the first 128 tokens (0:128) in the
    # BERT sequence.
    #
    def generator():
        prepseq = prepare_sequence_for_bertseq_left128
        
        for tokseq in tokseq_generator():
            corpus = tokseq['corpus']
            group = tokseq['group']
            identifier = tokseq['identifier']
            
            tokens = tokseq['tokens']
            input_ids = tokseq['input_ids']
            token_type_ids = tokseq['token_type_ids']
            attention_mask = tokseq['attention_mask']
            spans = tokseq['spans']
            tags = tokseq['tags']
            tags_int = tokseq['tags_int']
            
            n = len(tokens)
            
            if n == 0:
                #
                # Ignore empty sequences.
                #
                continue
            elif n <= 126:
                #
                # Sentences with up to 126 tokens fit in a single
                # BERT sequence. That's nice.
                #
                tokens = prepseq(tokens, start='[CLS]', end='[SEP]', padding='[PAD]')
                input_ids = prepseq(input_ids, start=2, end=3, padding=0)
                token_type_ids = prepseq(token_type_ids, start=0, end=0, padding=0)
                attention_mask = prepseq(attention_mask, start=1, end=1, padding=0)
                spans = prepseq(spans, start=(-1, -1), end=(-1, -1), padding=(-1, -1))
                tags = prepseq(tags, start='O', end='O', padding='PAD')
                tags_int = prepseq(tags_int, start=1, end=1, padding=0)
                
                is_prediction = prepseq([1] * n, start=0, end=0, padding=0)
                
                yield {
                    'corpus': corpus,
                    'group': group,
                    'identifier': identifier,
                    'tokens': tokens,
                    'input_ids' : input_ids,
                    'token_type_ids' : token_type_ids,
                    'attention_mask' : attention_mask,
                    'spans': spans,
                    'tags': tags,
                    'tags_int': tags_int,
                    'is_prediction': is_prediction,
                }
                
            elif n > 126:
                #
                # Not so nice, but we must take care of this anyway.
                #
                rest = (n - 126) % 86
                if rest == 0:
                    remaining = 0
                else:
                    remaining = 86 - rest
                
                new_n = n + remaining
                
                #
                # Split into sequences of 126 tokens by shifting 86
                # tokens at a time.
                # Note: at the end it will be 512 tokens because we will
                #       add (i) [CLS] and [SEP] special tokens, and
                #       (ii) padding as needed.
                #
                step = 86
                for i in range(0, new_n - 126 + step, step):
                    start = i
                    end = i + 126
                    
                    sequence_tokens = tokens[start:end]
                    sequence_input_ids = input_ids[start:end]
                    sequence_token_type_ids = token_type_ids[start:end]
                    sequence_attention_mask = attention_mask[start:end]
                    sequence_spans = spans[start:end]
                    sequence_tags = tags[start:end]
                    sequence_tags_int = tags_int[start:end]
                    
                    if start == 0:
                        #
                        # If it is the first split then the first 106
                        # tokens (20 + 86) are considered for
                        # prediction.
                        #
                        is_prediction = [1] * 106 + [0] * 20
                    elif end >= n:
                        remaining_no_context = end - n
                        remaining_for_prediction = 126 - 20 - remaining_no_context
                        is_prediction = [0] * 20 + [1] * remaining_for_prediction + [0] * remaining_no_context
                    else:
                        #
                        # If it is one of the middle splits then the
                        # first 20 tokens are not for prediction (only
                        # context), the next 86 tokens are for
                        # prediction, and the last 20 tokens are not
                        # for prediction (only context).
                        #
                        is_prediction = [0] * 20 + [1] * 86 + [0] * 20
                    
                    sequence_tokens = prepseq(sequence_tokens, start='[CLS]', end='[SEP]', padding='[PAD]')
                    sequence_input_ids = prepseq(sequence_input_ids, start=2, end=3, padding=0)
                    sequence_token_type_ids = prepseq(sequence_token_type_ids, start=0, end=0, padding=0)
                    sequence_attention_mask = prepseq(sequence_attention_mask, start=1, end=1, padding=0)
                    sequence_spans = prepseq(sequence_spans, start=(-1, -1), end=(-1, -1), padding=(-1, -1))
                    sequence_tags = prepseq(sequence_tags, start='O', end='O', padding='PAD')
                    sequence_tags_int = prepseq(sequence_tags_int, start=1, end=1, padding=0)
                    
                    is_prediction = prepseq(is_prediction, start=0, end=0, padding=0)
                    
                    yield {
                        'corpus': corpus,
                        'group': group,
                        'identifier': identifier,
                        'tokens': sequence_tokens,
                        'input_ids' : sequence_input_ids,
                        'token_type_ids' : sequence_token_type_ids,
                        'attention_mask' : sequence_attention_mask,
                        'spans': sequence_spans,
                        'tags': sequence_tags,
                        'tags_int': sequence_tags_int,
                        'is_prediction': is_prediction,
                    }
    
    generator.__name__ = f"{tokseq_generator.__name__}_bertseq_left128"
    
    return generator


def selector_generator(bertseq_generator, start=0, end=128):
    def generator():
        for bertseq in bertseq_generator():
            yield {
                'corpus': bertseq['corpus'],
                'group': bertseq['group'],
                'identifier': bertseq['identifier'],
                'tokens': bertseq['tokens'][start:end],
                'input_ids' : bertseq['input_ids'][start:end],
                'token_type_ids' : bertseq['token_type_ids'][start:end],
                'attention_mask' : bertseq['attention_mask'][start:end],
                'spans': bertseq['spans'][start:end],
                'tags': bertseq['tags'][start:end],
                'tags_int': bertseq['tags_int'][start:end],
                'is_prediction': bertseq['is_prediction'][start:end],
            }
    
    generator.__name__ = f"{bertseq_generator.__name__}_selector"
    
    return generator


def prepare_sequence_for_bertseq_left(sequence, start, end, padding):
    #
    # Guarantee that the input sequence:
    #   (1) has a maximum length of 510 values.
    #
    # Steps to obtain the output sequence:
    #   (1) the "start" value is added at the beginning of the sequence.
    #   (2) the "end" value is added at the end of the sequence.
    #   (3) the sequence is padded at the right to have a length of 512.
    #
    assert len(sequence) <= 510
    
    sequence = [start] + sequence + [end]
    
    remaining = 512 - len(sequence)
    
    sequence = sequence + remaining * [padding]
    
    return sequence


def bertseq_left_generator(tokseq_generator):
    def generator():
        prepseq = prepare_sequence_for_bertseq_left
        
        for tokseq in tokseq_generator():
            corpus = tokseq['corpus']
            group = tokseq['group']
            identifier = tokseq['identifier']
            
            tokens = tokseq['tokens']
            input_ids = tokseq['input_ids']
            token_type_ids = tokseq['token_type_ids']
            attention_mask = tokseq['attention_mask']
            spans = tokseq['spans']
            tags = tokseq['tags']
            tags_int = tokseq['tags_int']
            
            n = len(tokens)
            
            if n == 0:
                #
                # Ignore empty sequences.
                #
                continue
            elif n <= 510:
                #
                # Sentences with up to 510 tokens fit in a single
                # BERT sequence. That's nice.
                #
                tokens = prepseq(tokens, start='[CLS]', end='[SEP]', padding='[PAD]')
                input_ids = prepseq(input_ids, start=2, end=3, padding=0)
                token_type_ids = prepseq(token_type_ids, start=0, end=0, padding=0)
                attention_mask = prepseq(attention_mask, start=1, end=1, padding=0)
                spans = prepseq(spans, start=(-1, -1), end=(-1, -1), padding=(-1, -1))
                tags = prepseq(tags, start='O', end='O', padding='PAD')
                tags_int = prepseq(tags_int, start=1, end=1, padding=0)
                
                is_prediction = prepseq([1] * n, start=0, end=0, padding=0)
                
                yield {
                    'corpus': corpus,
                    'group': group,
                    'identifier': identifier,
                    'tokens': tokens,
                    'input_ids' : input_ids,
                    'token_type_ids' : token_type_ids,
                    'attention_mask' : attention_mask,
                    'spans': spans,
                    'tags': tags,
                    'tags_int': tags_int,
                    'is_prediction': is_prediction,
                }
                
            elif n > 510:
                #
                # Not so nice, but we must take care of this anyway.
                #
                rest = n % 255
                if rest == 0:
                    remaining = 0
                else:
                    remaining = 255 - rest
                
                new_n = n + remaining
                
                #
                # Split into sequences of 510 tokens by shifting 255
                # tokens at a time.
                # Note: at the end it will be 512 tokens because we will
                #       add [CLS] and [SEP] special tokens. And in the
                #       last split we make sure padding is added as
                #       needed.
                #
                step = 255
                for i in range(0, new_n - step, step):
                    start = i
                    end = i + 510
                    
                    sequence_tokens = tokens[start:end]
                    sequence_input_ids = input_ids[start:end]
                    sequence_token_type_ids = token_type_ids[start:end]
                    sequence_attention_mask = attention_mask[start:end]
                    sequence_spans = spans[start:end]
                    sequence_tags = tags[start:end]
                    sequence_tags_int = tags_int[start:end]
                    
                    if start == 0:
                        #
                        # If it is the first split then the first 383
                        # tokens (128 + 255) are considered for
                        # prediction.
                        #
                        is_prediction = [1] * 383 + [0] * 127
                    elif end >= n:
                        remaining_no_context = end - n
                        remaining_for_prediction = 510 - 128 - remaining_no_context
                        is_prediction = [0] * 128 + [1] * remaining_for_prediction + [0] * remaining_no_context
                    else:
                        #
                        # If it is one of the middle splits then the
                        # first 128 tokens are not for prediction (only
                        # context), the next 255 tokens are for
                        # prediction, and the last 127 tokens are not
                        # for prediction (only context).
                        #
                        is_prediction = [0] * 128 + [1] * 255 + [0] * 127
                    
                    sequence_tokens = prepseq(sequence_tokens, start='[CLS]', end='[SEP]', padding='[PAD]')
                    sequence_input_ids = prepseq(sequence_input_ids, start=2, end=3, padding=0)
                    sequence_token_type_ids = prepseq(sequence_token_type_ids, start=0, end=0, padding=0)
                    sequence_attention_mask = prepseq(sequence_attention_mask, start=1, end=1, padding=0)
                    sequence_spans = prepseq(sequence_spans, start=(-1, -1), end=(-1, -1), padding=(-1, -1))
                    sequence_tags = prepseq(sequence_tags, start='O', end='O', padding='PAD')
                    sequence_tags_int = prepseq(sequence_tags_int, start=1, end=1, padding=0)
                    
                    is_prediction = prepseq(is_prediction, start=0, end=0, padding=0)
                    
                    yield {
                        'corpus': corpus,
                        'group': group,
                        'identifier': identifier,
                        'tokens': sequence_tokens,
                        'input_ids' : sequence_input_ids,
                        'token_type_ids' : sequence_token_type_ids,
                        'attention_mask' : sequence_attention_mask,
                        'spans': sequence_spans,
                        'tags': sequence_tags,
                        'tags_int': sequence_tags_int,
                        'is_prediction': is_prediction,
                    }
    
    generator.__name__ = f"{tokseq_generator.__name__}_bertseq_left"
    
    return generator


def pad_sequence_for_bertseq_center(sequence, value):
    #
    # Guarantee that the output sequence:
    #   (1) has 128 padding values at left.
    #   (2) has (at least) 128 padding values at right.
    #   (3) it is divisible by 256.
    #   (4) has at least 512 values.
    #
    n_mid = len(sequence)
    
    n_left = 128
    rest = len(sequence) % 128
    if rest == 0:
        remaining = 0
    else:
        remaining = 128 - rest
    n_right = remaining + 128
    
    if (n_left + n_mid + n_right) % 256 != 0:
        n_right += 128
    
    while (n_left + n_mid + n_right) < 512:
        n_right += 256
    
    left = n_left * [value]
    right = n_right * [value]
    
    return left + sequence + right


def bertseq_center_generator(tokseq_generator):
    #
    # Split a document into sequences of length equal to 512 tokens.
    # Each sequence is represented as:
    #     [CLS] token1 token2 ... tokenN [SEP].
    #
    # Only the middle 256 tokens (indexes 128 to 383) are used for
    # inference.
    # The first one (index 0) is [CLS].
    # The last one (index 511) is [SEP].
    # The first 127 tokens after [CLS] (indexes 1 to 127) and the last
    # 127 tokens before [SEP] (indexes 384 to 510) are used as
    # additional context. If they are not available they are set to
    # padding (its input_id is 0, and attention_mask is set to 0).
    #
    def generator():
        padseq = pad_sequence_for_bertseq_center
        
        for tokseq in tokseq_generator():
            corpus = tokseq['corpus']
            group = tokseq['group']
            identifier = tokseq['identifier']
            
            #
            # The tokseq (tokenized sequence) will be split into smaller
            # sequences for BERT (containing 512 tokens).
            #
            # Only the middle 256 tokens are used for inference.
            #
            tokens = tokseq['tokens']
            input_ids = tokseq['input_ids']
            token_type_ids = tokseq['token_type_ids']
            attention_mask = tokseq['attention_mask']
            spans = tokseq['spans']
            tags = tokseq['tags']
            tags_int = tokseq['tags_int']
            
            if 'passage_number' in tokseq:
                passage_number = tokseq['passage_number']
            
            n = len(tokens)
            is_prediction = [1] * n
            
            #
            # We ignore empty sequences because we do not want the BERT
            # model to become expert at predicting padding.
            # Just kidding. That's useless.
            #
            if n == 0:
                continue
            
            tokens = padseq(sequence=tokens, value='[PAD]')
            input_ids = padseq(sequence=input_ids, value=0)
            token_type_ids = padseq(sequence=token_type_ids, value=0)
            attention_mask = padseq(sequence=attention_mask, value=0)
            spans = padseq(sequence=spans, value=(-1, -1))
            tags = padseq(sequence=tags, value='PAD')
            tags_int = padseq(sequence=tags_int, value=0)
            
            if 'passage_number' in tokseq:
                passage_number = padseq(sequence=passage_number, value=-1)
            
            is_prediction = padseq(sequence=is_prediction, value=0)
            
            #
            # Split into sequences of 512 tokens by shifting 256 tokens
            # at a time.
            #
            step = 256
            new_n = len(tokens)
            for i in range(0, new_n - step, step):
                start = i
                end = i + 512
                
                sequence_tokens = tokens[start:end]
                sequence_input_ids = input_ids[start:end]
                sequence_token_type_ids = token_type_ids[start:end]
                sequence_attention_mask = attention_mask[start:end]
                sequence_spans = spans[start:end]
                sequence_tags = tags[start:end]
                sequence_tags_int = tags_int[start:end]
                
                if 'passage_number' in tokseq:
                    sequence_passage_number = passage_number[start:end]
                
                sequence_is_prediction = is_prediction[start:end]
                sequence_is_prediction[0:128] = [0] * 128
                sequence_is_prediction[384:512] = [0] * 128
                
                #
                # [CLS] at start.
                #
                sequence_tokens[0] = '[CLS]'
                sequence_input_ids[0] = 2
                sequence_attention_mask[0] = 1
                sequence_tags[0] = 'O'
                sequence_tags_int[0] = 1
                #
                # [SEP] at end.
                #
                sequence_tokens[-1] = '[SEP]'
                sequence_input_ids[-1] = 3
                sequence_attention_mask[-1] = 1
                sequence_tags[-1] = 'O'
                sequence_tags_int[-1] = 1
                
                d = {
                    'corpus': corpus,
                    'group': group,
                    'identifier': identifier,
                    'tokens': sequence_tokens,
                    'input_ids' : sequence_input_ids,
                    'token_type_ids' : sequence_token_type_ids,
                    'attention_mask' : sequence_attention_mask,
                    'spans': sequence_spans,
                    'tags': sequence_tags,
                    'tags_int': sequence_tags_int,
                }
                
                if 'passage_number' in tokseq:
                    d['passage_number'] = sequence_passage_number
                
                d['is_prediction'] = sequence_is_prediction
                
                yield d
    
    generator.__name__ = f"{tokseq_generator.__name__}_bertseq_center"
    
    return generator


def entity_set_to_dict(es):
    entities = defaultdict(set)
    for entity in es:
        entities[entity.text.lower()].add(entity.span)
        
    return entities
        

def document_level_agreement(mode="all"):
    
    assert mode in ["voting", "all"]
    
    def doc_lvl_agreement(text, es):
        
        entities = entity_set_to_dict(es)

        #lower_text = text.lower()

        items = sorted(list(entities.items()), key=lambda x: -len(x[0]))

        document_level_entities = []

        for entity_text, occurences in items:

            c_in_entities = []
            c_out_entities = []

            for match in re.finditer(r"(?:^|\W)("+re.escape(entity_text)+r")(?:\W|$)", text, flags=re.IGNORECASE):
                _entity_text = match.groups()[0]
                assert len(entity_text)==len(_entity_text)
                span = match.span(1)

                if span in occurences:
                    c_in_entities.append(Entity(_entity_text, span, "Chemical"))
                else:
                    c_out_entities.append(Entity(_entity_text, span, "Chemical"))

                if not len(text) == len(text[:span[0]] + " "*len(_entity_text)+ text[span[1]:]):
                    print("DEBUG")
                    print(entity_text)
                    print(span)
                    print(match.span())
                    print(text[span[0]-40:span[1]+40])

                    assert 1==0

                text = text[:span[0]] + " "*len(_entity_text)+ text[span[1]:]

            if mode == "voting":
                if len(c_in_entities) > len(c_out_entities):
                # use all the text matched entities
                    document_level_entities.extend(c_in_entities+c_out_entities)
            elif mode == "all":
                document_level_entities.extend(c_in_entities+c_out_entities)

                #print(f"{entity_text} was accepted with {c_in}/{c_out} votes")
            #else:
                # entity is rejected
                #print(f"{entity_text} was rejected with {c_in}/{c_out} votes")  

        return EntitySet(document_level_entities)
    
    return doc_lvl_agreement
    

class SequenceDecoder(BaseLogger):
    
    def __init__(self, corpora, reconstructor_bio=None, doc_majoraty_voting=None):
        
        super().__init__()
        self.corpora = {str(corpus): corpus for corpus in corpora}
        
        # auxiliar var that holds a dictionary that can be built in a batch-wise fashion
        self.documents_dict = {}
        self.reconstructor_bio = reconstructor_bio
        self.doc_majoraty_voting = doc_majoraty_voting
        
        #
        # Load into memory the gold standard (true) texts and entity
        # sets.
        #
        self.documents = dict()
        for corpus_str, corpus in self.corpora.items():
            self.documents[corpus_str] = dict()
            for group, collection in corpus:
                self.documents[corpus_str][group] = dict()
                for i, d in collection:
                    self.documents[corpus_str][group][i] = dict()
                    self.documents[corpus_str][group][i]['text'] = d.text()
                    # self.documents[corpus_str][group][i]['nes'] = d.nes()
                    self.documents[corpus_str][group][i]['es'] = d.get_entity_set()
    
    def clear_state(self):
        self.documents_dict = {}
    
    def samples_from_batch(self, samples):
         #
        # Unbatching in Python.
        #
        _samples = []
        
        if isinstance(samples, dict):
            samples = [samples]
            
        for sample in samples:
            key = list(sample.keys())[0]
            batch_size = sample[key].shape[0]
            for i in range(batch_size):
                _samples.append({k: v[i] for k, v in sample.items()})
        samples = _samples
        
        for data in samples:
            corpus = data['corpus'].numpy().decode()
            group = data['group'].numpy().decode()
            identifier = data['identifier'].numpy().decode()
            spans = data['spans'].numpy().tolist()
            is_prediction = data['is_prediction'].numpy().tolist()
            
            #
            # Convert predicted integer values tags to predicted tags.
            #
            tags_pred = [INT2TAG[i] for i in data['tags_int_pred'].numpy().tolist()]
            
            # THIS IS FOR DEBUG, REMOVE 
            tags_true = [INT2TAG[i] for i in data['tags_int'].numpy().tolist()]
            
            _zip = [spans, tags_pred, tags_true, is_prediction]
            
            #if "tokens" in data:
            #    tokens = [ x.decode() for x in data['tokens'].numpy().tolist()]
            #    _zip.append(tokens)
                
            if "passage_number" in data:
                passage_numbers = data['passage_number'].numpy().tolist()
                _zip.append(passage_numbers)
            
            if corpus not in self.documents_dict:
                self.documents_dict[corpus] = dict()
            if group not in self.documents_dict[corpus]:
                self.documents_dict[corpus][group] = dict()
            if identifier not in self.documents_dict[corpus][group]:
                self.documents_dict[corpus][group][identifier] = {
                    'spans': list(), 'tags': list(), 'tags_true':list(), 'passage_number':list()}
            
            #
            # Select only the values that were marked for prediction.
            #
            filtered_spans = list()
            filtered_tags = list()
            filtered_tags_true = list()
            #filtered_tokens = list()
            filtered_passage_numbers = list()
            for s, t, t_true, p, *passage_numbers in zip(*_zip):
                if p == 1:
                    filtered_spans.append(s)
                    filtered_tags.append(t)
                    filtered_tags_true.append(t_true)
                    if len(passage_numbers)>0:
                        filtered_passage_numbers.append(passage_numbers[0])
                        
            
            self.documents_dict[corpus][group][identifier]['spans'].extend(filtered_spans)
            self.documents_dict[corpus][group][identifier]['tags'].extend(filtered_tags)
            self.documents_dict[corpus][group][identifier]['tags_true'].extend(filtered_tags_true)
            #self.documents_dict[corpus][group][identifier]['tokens'].extend(filtered_tokens)
            self.documents_dict[corpus][group][identifier]['passage_number'].extend(filtered_passage_numbers)
            
    def decode(self):

        counts = {
            'tags': 0,
            'inside_tag_at_start': 0,
            'inside_tag_after_other_tag': 0,
            'inside_tag_with_different_entity_type': 0,
            'inside_tag_at_passage_start': 0,
        }
        
        #
        # The spans and the respective tags are assumed to be already
        # ordered.
        #
        for corpus in self.documents_dict:
            for group in self.documents_dict[corpus]:
                for identifier in self.documents_dict[corpus][group]:
                    #
                    # Get the original text from the gold standard (true)
                    # document.
                    #
                    text = self.documents[corpus][group][identifier]['text']
                    
                    spans = self.documents_dict[corpus][group][identifier]['spans']
                    tags = self.documents_dict[corpus][group][identifier]['tags']
                    passage_number = self.documents_dict[corpus][group][identifier]['passage_number']
                    passage_number = passage_number if len(passage_number)>0 else None
                    #
                    # Given the predicted tags, the respective spans,
                    # and the original (true) text get the predicted
                    # entity set.
                    #
                    if self.reconstructor_bio is not None:
                        #print("RUN RECONSTRUCT BIO")
                        raise ValueError("Tokens were removed from samples_from_batch, so reconstructor will not work, please change this if you want to use it.")
                        tags = self.reconstructor_bio(tags, spans, self.documents_dict[corpus][group][identifier]['tokens'])
                        
                    es, c = decode_bio(tags, spans, text, passage_number, allow_errors=True)
                    
                    if self.doc_majoraty_voting is not None:
                        es = self.doc_majoraty_voting(text, es)
                    
                    self.documents_dict[corpus][group][identifier]['es'] = es
                    
                    ## majority voting of the whole documnet
                    
                    # set strings das entidade {entity_text: {spans}}
                    # find and count string on text 
                    
                    counts['tags'] += c['tags']
                    counts['inside_tag_at_start'] += c['inside_tag_at_start']
                    counts['inside_tag_after_other_tag'] += c['inside_tag_after_other_tag']
                    counts['inside_tag_with_different_entity_type'] += c['inside_tag_with_different_entity_type']
                    counts['inside_tag_at_passage_start'] += c['inside_tag_at_passage_start']
        
        s = 'Statistics about the BIO decoding process: tags={}, inside_tag_at_start={}, inside_tag_after_other_tag={}, inside_tag_with_different_entity_type={}, inside_tag_at_passage_start={}.'.format(
            counts['tags'], counts['inside_tag_at_start'], counts['inside_tag_after_other_tag'], counts['inside_tag_with_different_entity_type'], counts['inside_tag_at_passage_start'])
        
        self.logger.info(s)
    
    def decode_from_samples(self, samples):
        #
        # To keep track of the number of BIO decoding errors.
        #
        
        # here we assume a gigant batch that contains all the dataset
        self.samples_from_batch(samples)
        
        self.decode()
            
    def evaluate_ner_from_sample(self, samples):
        
        self.decode_from_samples(samples)
        
        return self._evaluate_ner()
    
    def evaluate_ner(self):
        
        self.decode()
        
        return self._evaluate_ner()
    
    def _evaluate_ner(self):
        
        true_list = list()
        pred_list = list()
        
        for corpus in self.documents_dict:
            for group in self.documents_dict[corpus]:
                for identifier in self.documents_dict[corpus][group]:
                    
                    true_es = self.documents[corpus][group][identifier]['es']
                    pred_es = self.documents_dict[corpus][group][identifier]['es']
                    
                    true_list.append(true_es)
                    pred_list.append(pred_es)
        
        results = eval_list_of_entity_sets(true_list, pred_list)
        
        # clear the document state
        self.clear_state()
        
        return results
    
    def _get_collections(self):
                
        collections = dict()
        for corpus in self.documents_dict:
            collections[corpus] = dict()
            for group in self.documents_dict[corpus]:
                #
                # This is important: make a deepcopy to not modify the
                #                    original corpora.
                #
                collections[corpus][group] = deepcopy(self.corpora[corpus][group])
                #
                # Sanity measure: remove all the entities from the
                #                 collection.
                #
                collections[corpus][group].clear_entities()
        
        for corpus in self.documents_dict:
            for group in self.documents_dict[corpus]:
                for identifier in self.documents_dict[corpus][group]:
                    nes = self.documents_dict[corpus][group][identifier]['es'].to_normalized_entity_set()
                    entities = nes.get()
                    collections[corpus][group][identifier].set_entities(entities)
        
        # clear the document state
        self.clear_state()
        
        return collections
    
    def get_collections_from_samples(self, samples):
        r"""
        Return a dictionary with Collection objects.
        The first-level key is the corpus name.
        The second-level key is the group name.
        
        Each collection contains the predicted entities derived from
        the predicted samples (that have the predicted BIO tags).
        """
        self.decode_from_samples(samples)
        
        return self._get_collections()
    
    def get_collections(self):
        r"""
        Return a dictionary with Collection objects.
        The first-level key is the corpus name.
        The second-level key is the group name.
        
        Each collection contains the predicted entities derived from
        the predicted samples (that have the predicted BIO tags).
        """
        self.decode()
        
        return self._get_collections()
        


def get_bert_map_function(checkpoint, bert_layer_index=-1, **kwargs): # selecting only 256
    
    bert_model = TFBertModel.from_pretrained(checkpoint,
                                             output_attentions = False,
                                             output_hidden_states = True,
                                             return_dict=True,
                                             from_pt=True)
    
    @tf.function
    def embeddings(**kwargs):
        return bert_model(kwargs)["hidden_states"][bert_layer_index] # NONE, 512, 768

    def run_bert(data):

        data["embeddings"] = embeddings(input_ids=data["input_ids"], 
                                         token_type_ids=data["token_type_ids"],
                                         attention_mask=data["attention_mask"])

        return data
    
    return run_bert

def get_mapps_for_model(model):
    
    cfg = model.savable_config
    
    mapps = {}
    
    if "embeddings" in cfg:
        if cfg["embeddings"]["type"]=="bert":
            mapps["pre_tf_transformation"] = get_bert_map_function(**cfg["embeddings"])
            
            if "low" in cfg["model"]:
                def training(data):
                    return data["embeddings"][cfg["model"]["low"]:cfg["model"]["high"],:], tf.one_hot(data["tags_int"][cfg["model"]["low"]:cfg["model"]["high"]], 
                                                          cfg["model"]["output_classes"])
            else:
                def training(data):
                    return data["embeddings"], tf.one_hot(data["tags_int"], 
                                                          cfg["model"]["output_classes"])
            
            mapps["training"] = training
            if "low" in cfg["model"]:
                def testing(data):
                    data["embeddings"] = data["embeddings"][cfg["model"]["low"]:cfg["model"]["high"],:]
                    data["spans"] = data["spans"][cfg["model"]["low"]:cfg["model"]["high"]]
                    data["is_prediction"] = data["is_prediction"][cfg["model"]["low"]:cfg["model"]["high"]]
                    data["tags_int"] = tf.cast(data["tags_int"][cfg["model"]["low"]:cfg["model"]["high"]], tf.int32)
                    return data
            else:
                def testing(data):
                    data["tags_int"] = tf.cast(data["tags_int"], tf.int32)

                    return data
                
            mapps["testing"] = testing
            
    return mapps

short_checkpoint_names = {
    "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext":"pubmedBertFull",
    'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract':"pubmedBertAbstract",
    'cambridgeltl/SapBERT-from-PubMedBERT-fulltext':"SapBert"
}

