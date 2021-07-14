#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import json

from config import NLM_CHEM_GROUPS
from elements import Collection
from elements import Document
from elements import Entity
from elements import IndexingIdentifier
from elements import Passage


ANNOTATION_TYPES = ['Chemical', 'MeSH_Indexing_Chemical']


def get_collection_from_json(d):
    #
    c = Collection()
    #
    for document in d['documents']:
        d = Document(identifier=document['id'])
        for passage in document['passages']:
            annotations = passage['annotations']
            typ = passage['infons']['type']
            section_type = passage['infons'].get('section_type', '')
            start = passage['offset']
            text = passage['text']
            #
            # Number of passages need to match. Therefore passages
            # without any text neither annotations, despite useless,
            # should not be discarded.
            #
            end = start + len(text)
            p = Passage(text, (start, end), typ, section_type)
            #
            # If the passage has no text then it has no annotations.
            #
            if len(text) == 0:
                assert len(annotations) == 0
            #
            # Add the annotations (entities and indexing identifiers).
            #
            for ann in annotations:
                ann_type = ann['infons']['type']
                assert ann_type in ANNOTATION_TYPES
                if ann_type == 'Chemical':
                    #
                    # Assert that there is one and only one (contiguous)
                    # span for the annotation.
                    #
                    ann_locations = ann['locations']
                    assert len(ann_locations) == 1
                    ann_location = ann_locations[0]
                    ann_start = ann_location['offset']
                    ann_end = ann_location['offset'] + ann_location['length']
                    ann_text = ann['text']
                    ann_identifiers = ann['infons']['identifier']
                    ann_identifiers_set = set(i for i in ann_identifiers.split(','))
                    #
                    # If the annotation text does not match the text
                    # from the Passage text, then automatically try to
                    # find the correct span by shifting one or two
                    # characters (to left or right). Note: this may
                    # be misleading.
                    #
                    # This was implemented because at the time of
                    # writing (2021-07-14) there is one error (incorrect
                    # annotation span) in the NLM-Chem train subset.
                    #
                    if text[ann_start-start:ann_end-start] != ann_text:
                        ann_start_old = ann_start
                        ann_end_old = ann_end
                        found = False
                        offset_trials = [1, -1, 2, -2]
                        for i in offset_trials:
                            ann_start = ann_start_old + i
                            ann_end = ann_end_old + i
                            if text[ann_start-start:ann_end-end] == ann_text:
                                found = True
                                break
                        if found:
                            print('Warning: in document {}, the annotation {}, with original span {}, did not match the passage text, but the correct span {} was automatically found.'.format(repr(d.identifier), repr(ann_text), (ann_start_old, ann_end_old), (ann_start, ann_end)))
                        else:
                            print('Error: in document {}, the annotation {}, with original span {}, did not match the passage text, and the correct span could not be found.'.format(repr(d.identifier), repr(ann_text), (ann_start_old, ann_end_old)))
                            exit()
                    #
                    e = Entity(ann_text, (ann_start, ann_end), ann_type, ann_identifiers_set)
                    p.add_entity(e)
                elif ann_type == 'MeSH_Indexing_Chemical':
                    ann_identifier = ann['infons']['identifier']
                    ii = IndexingIdentifier(ann_identifier, ann_type)
                    p.add_indexing_identifier(ii)
            d.add_passage(p)
        c.add(d.identifier, d)
    #
    return c


def get_collection_from_json_file(filepath):
    #
    with open(filepath, mode='r', encoding='utf-8') as f:
        d = json.load(f)
    #
    c = get_collection_from_json(d)
    return c



class NLMChemCorpus:
    #
    def __init__(self):
        #
        self.collections = dict()
        for g, fp in NLM_CHEM_GROUPS.items():
            self.collections[g] = get_collection_from_json_file(fp)
    #
    def __str__(self):
        return 'NLMChemCorpus'
