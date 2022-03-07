#!/usr/bin/env python3
# -*- coding: utf-8 -*-


#
# References:
# https://allenai.github.io/scispacy/
# https://stackoverflow.com/questions/62072566/how-to-speed-up-spacys-nlp-call
# https://stackoverflow.com/questions/58516766/sentence-split-using-spacy-sentenizer
#


import spacy
from transformers import BertTokenizerFast

from annotator.elements import add_offset_to_spans


nlp = spacy.load('en_core_sci_sm', disable=['ner'])


def sentence_splitting(text, offset=0):
    text_length = len(text)

    doc = nlp(text)

    sentences = list()
    sentences_spans = list()

    for sent in doc.sents:
        sentences.append(sent.text)
        sentences_spans.append((sent.start_char + offset, sent.end_char + offset))

    return sentences, sentences_spans, text_length


def global_sentence_splitting(texts, offset=0, n_process=-1):
    docs = nlp.pipe(texts, n_process=n_process)

    global_sentences = []
    global_sentences_spans = []
    global_text_lengths = []

    for doc in docs:
        global_sentences.append(list())
        global_sentences_spans.append(list())
        global_text_lengths.append(len(doc.text))

        for sent in doc.sents:
            global_sentences[-1].append(sent.text)
            global_sentences_spans[-1].append((sent.start_char + offset, sent.end_char + offset))

    return global_sentences, global_sentences_spans, global_text_lengths


def text_reconstruction_from_sentences(sentences, sentences_spans, text_length):
    #
    # Sentences spans are assumed to not overlap.
    #
    text = [' '] * text_length

    for sentence, (start, end) in zip(sentences, sentences_spans):
        text[start:end] = sentence

    text = ''.join(text)
    return text


def global_text_reconstruction_from_sentences(global_sentences,
                                              global_sentences_spans,
                                              global_text_lengths):
    texts = list()

    for sentences, sentences_spans, text_length in zip(global_sentences,
                                                       global_sentences_spans,
                                                       global_text_lengths):

        text = text_reconstruction_from_sentences(sentences,
                                                  sentences_spans,
                                                  text_length)

        texts.append(text)

    return texts


PUBMEDBERT_FULL = 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext'
PUBMEDBERT = 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract'
SAPBERT = 'cambridgeltl/SapBERT-from-PubMedBERT-fulltext'


class Tokenizer:

    def __init__(self, model_name=PUBMEDBERT_FULL):
        self.tokenizer = BertTokenizerFast.from_pretrained(model_name)

    def tokenize(self, text, offset=0):
        #
        # Ignore special tokens '[CLS]' and '[SEP]'.
        #
        tokenized_batch = self.tokenizer(text, add_special_tokens=False)
        tokenized_text = tokenized_batch[0]

        tokens = tokenized_text.tokens
        input_ids = tokenized_text.ids
        token_type_ids = tokenized_text.type_ids
        attention_mask = tokenized_text.attention_mask
        spans = tokenized_text.offsets

        #
        # Add offset to spans.
        #
        spans = add_offset_to_spans(spans, offset)

        return (tokens, input_ids, token_type_ids, attention_mask, spans)
