#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import sys
sys.path.extend(['.', '..'])

from annotator.corpora import get_collection_from_json
from annotator.corpora import read_json
from annotator.elements import merge_collections


#
# Input JSON filepaths.
#
filepaths = [
    '../../outputs/normalizer/BaseCorpus_BC7T2-NLMChem-corpus-train.BioC_wEmbeddings.json',
    '../../outputs/normalizer/BaseCorpus_BC7T2-NLMChem-corpus-dev.BioC_wEmbeddings.json',
    '../../outputs/normalizer/BaseCorpus_BC7T2-NLMChem-corpus-test.BioC_wEmbeddings.json',
]

#
# Output JSON filepath.
#
out_fp = 'output.json'


collections = list()

for fp in filepaths:
    c = get_collection_from_json(d=read_json(fp),
                                ignore_non_contiguous_entities=False,
                                ignore_normalization_identifiers=False,
                                solve_overlapping_passages=False)
    collections.append(c)

merged = merge_collections(*collections)

s = merged.pretty_json()

with open(out_fp, mode='w', encoding='utf-8') as f:
    _ = f.write(s)

print('\nFile {} written with success!'.format(repr(out_fp)))
