#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import json
import re

from config import CDR_GROUPS
from config import CHEMDNER_GROUPS
from config import NLM_CHEM_GROUPS
from elements import Collection
from elements import Document
from elements import Entity
from elements import IndexingIdentifier
from elements import Passage

from Utils import BaseLogger

ANNOTATION_TYPES = ['Chemical', 'MeSH_Indexing_Chemical']

RE_MESH = re.compile(r'MESH:\w+')


def is_mesh_identifier(s):
    #
    # Naive check for a MeSH identifier.
    #
    if RE_MESH.fullmatch(s):
        return True
    else:
        return False


def find_mesh_identifiers(s):
    #
    # CDR corpus examples:
    #   "MESH:D014527|D012492"
    #   "MESH:D002188|D000431|D009538"
    #   "-"
    #
    # NLM-Chem corpus examples:
    #   "MESH:D010710,MESH:D000077330"
    #   "MESH:D014325,MESH:D012965,MESH:D011136"
    #   "MESH:D007659,MESH:D000447,-"
    #   "-"
    #
    assert isinstance(s, str)
    #
    if '|' in s:
        assert ',' not in s
        split_char = '|'
    else:
        split_char = ','
    #
    identifiers = set()
    for i in s.split(split_char):
        if i in ['', '-']:
            pass
        else:
            if not i.startswith('MESH:'):
                i = 'MESH:' + i
            assert is_mesh_identifier(i), 'Invalid MeSH identifier: {}.'.format(repr(i))
            identifiers.add(i)
    #
    return identifiers


def get_collection_from_json(d, ignore_non_contiguous_entities=False, ignore_normalization_identifiers=False, solve_overlapping_passages=False):
    assert isinstance(d, dict)
    assert isinstance(ignore_non_contiguous_entities, bool)
    assert isinstance(ignore_normalization_identifiers, bool)
    assert isinstance(solve_overlapping_passages, bool)
    #
    c = Collection()
    #
    for document in d['documents']:
        doc = Document(identifier=document['id'])
        for passage in document['passages']:
            annotations = passage['annotations']
            typ = passage['infons']['type']
            section_type = passage['infons'].get('section_type', '')
            start = passage['offset']
            #
            # At the time of writing (2021-07-21) the CHEMDNER dataset
            # contains overlapping passages (both the title and the
            # abstract have offset 0). This is unexpected and should be
            # automatically corrected (during loading, on-the-fly).
            #
            if solve_overlapping_passages:
                start_old = start
                doc_end = doc.span()[1]
                if (doc_end > 0) and (start < doc_end):
                    start = doc_end + 2
                start_diff = start - start_old
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
                    #
                    # The CDR dataset contains non-contiguous named
                    # entities (they are rare and overlap with longer
                    # named entities). These non-contiguous named
                    # entities are discarded.
                    #
                    if ignore_non_contiguous_entities:
                        if len(ann_locations) > 1:
                            continue
                    #
                    assert len(ann_locations) == 1
                    ann_location = ann_locations[0]
                    ann_start = ann_location['offset']
                    ann_end = ann_location['offset'] + ann_location['length']
                    if solve_overlapping_passages:
                        ann_start += start_diff
                        ann_end += start_diff
                    ann_text = ann['text']
                    #
                    # The CHEMDNER dataset does not contain MeSH
                    # identifiers for the entities.
                    #
                    if ignore_normalization_identifiers:
                        ann_identifiers = ''
                    else:
                        ann_identifiers = ann['infons']['identifier']
                    #
                    ann_identifiers_set = find_mesh_identifiers(ann_identifiers)
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
                            print('Warning: in document {}, the annotation {}, with original span {}, did not match the passage text, but the correct span {} was automatically found.'.format(repr(doc.identifier), repr(ann_text), (ann_start_old, ann_end_old), (ann_start, ann_end)))
                        else:
                            print('Error: in document {}, the annotation {}, with original span {}, did not match the passage text, and the correct span could not be found.'.format(repr(doc.identifier), repr(ann_text), (ann_start_old, ann_end_old)))
                            exit()
                    #
                    e = Entity(ann_text, (ann_start, ann_end), ann_type, ann_identifiers_set)
                    p.add_entity(e)
                elif ann_type == 'MeSH_Indexing_Chemical':
                    ann_identifier = ann['infons']['identifier']
                    assert is_mesh_identifier(ann_identifier), 'Invalid MeSH identifier: {}.'.format(repr(i))
                    ii = IndexingIdentifier(ann_identifier, ann_type)
                    p.add_indexing_identifier(ii)
            doc.add_passage(p)
        c.add(doc.identifier, doc)
    #
    return c


def read_json(filepath):
    with open(filepath, mode='r', encoding='utf-8') as f:
        d = json.load(f)
    return d

class BaseCorpus(BaseLogger):
    
    def __init__(self,
                 groups,
                 ignore_non_contiguous_entities,
                 ignore_normalization_identifiers,
                 solve_overlapping_passages):
        super().__init__()
        self.collections = dict()
        self.groups = list()
        self.n_documents = 0
        self.n_documents_per_group = dict()
        for g, fp in groups.items():
            self.groups.append(g)
            self.collections[g] = get_collection_from_json(d=read_json(fp),
                                                           ignore_non_contiguous_entities=ignore_non_contiguous_entities,
                                                           ignore_normalization_identifiers=ignore_normalization_identifiers,
                                                           solve_overlapping_passages=solve_overlapping_passages)
            self.n_documents += self.collections[g].n_documents
            self.n_documents_per_group[g] = self.collections[g].n_documents
            
        # Change made by Tiago Almeida 29/07/2021
        for group, collection in self.collections.items():
            collection.add_metadata(self.__class__.__name__, group)
    #
    def __str__(self):
        return self.__class__.__name__
    
    def __getitem__(self, g):
        return self.collections[g]

class NLMChemCorpus(BaseCorpus):
    #
    def __init__(self):
        super().__init__(NLM_CHEM_GROUPS, 
                         ignore_non_contiguous_entities=False,
                         ignore_normalization_identifiers=False,
                         solve_overlapping_passages=False)

class CDRCorpus(BaseCorpus):
    #
    def __init__(self):
        super().__init__(CDR_GROUPS, 
                         ignore_non_contiguous_entities=True,
                         ignore_normalization_identifiers=False,
                         solve_overlapping_passages=False)


class CHEMDNERCorpus(BaseCorpus):
    #
    def __init__(self):
        super().__init__(CHEMDNER_GROUPS, 
                         ignore_non_contiguous_entities=False,
                         ignore_normalization_identifiers=True,
                         solve_overlapping_passages=True)

