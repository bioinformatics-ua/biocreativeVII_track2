#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import sys
sys.path.extend(['.', '..'])

import os

from config import NLM_CHEM_TEST_ANN_PMCID_PMID


args = sys.argv[1:]
n = len(args)

if n != 1:
    print('Usage:')
    print('    $ python3 filter_run_for_chemical_identification_evaluation_on_final_test.py run.json')
    exit()


fp = args[0]
if not os.path.isfile(fp):
    print('Invalid filepath: {}.'.format(repr(fp)))
    exit()


root, ext = os.path.splitext(fp)
out_fp = root + '.filtered_for_chemical_identification' + ext

if os.path.exists(out_fp):
    print('Output filepath already exists: {}.'.format(repr(out_fp)))
    exit()


from corpora import get_collection_from_json
from corpora import read_json
from elements import Collection


def get_nlmchemtestann_ids():
    ids = list()
    with open(NLM_CHEM_TEST_ANN_PMCID_PMID, mode='r', encoding='utf-8') as f:
        for line in f:
            pmcid, pmid = line.strip().split('\t')
            assert pmcid[:3] == 'PMC'
            i = pmcid[3:]
            ids.append(i)
    return ids


ids = get_nlmchemtestann_ids()

c = get_collection_from_json(d=read_json(fp),
                             ignore_non_contiguous_entities=False,
                             ignore_normalization_identifiers=False,
                             solve_overlapping_passages=False)


#
# Filtered collection, containing only the documents for the
# chemical identification task (NER and normalization).
#

cf = Collection()

i = 0
for di, doc in c:
    if di in ids:
        cf.add(di, doc)
        i += 1
        print('{:>3s}: PMC{} added'.format('#{}'.format(i), di))

s = cf.pretty_json()

with open(out_fp, mode='w', encoding='utf-8') as f:
    _ = f.write(s)

print('\nFile {} written with success!'.format(repr(out_fp)))
