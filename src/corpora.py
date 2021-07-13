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


def get_collection_from_json_file(filepath):
    #
    with open(filepath, mode='r', encoding='utf-8') as f:
        s = json.load(f)
    #
    c = Collection()
    #
    #todo
    #
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
