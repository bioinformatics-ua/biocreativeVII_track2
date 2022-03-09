#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import sys
sys.path.extend(['.', '..'])

from annotator.config import NLMCHEM_GROUPS
from annotator.config import NLMCHEMTEST_GROUPS

import json


def read_json(filepath):
    with open(filepath, mode='r', encoding='utf-8') as f:
        d = json.load(f)
    return d


def get_pmcid2pmid(d):
    pmcid2pmid = dict()
    for document in d['documents']:
        pmcid = document['id']
        for passage in document['passages']:
            pmid = passage['infons'].get('article-id_pmid')
            if pmid is not None:
                assert pmcid not in pmcid2pmid
                pmcid2pmid[pmcid] = pmid
    
    return pmcid2pmid


nlmchem_pmcid2pmid = dict()

for group, filepath in NLMCHEM_GROUPS.items():
    d = read_json(filepath)
    for pmcid, pmid in get_pmcid2pmid(d).items():
        assert pmcid not in nlmchem_pmcid2pmid
        nlmchem_pmcid2pmid[pmcid] = pmid

nlmchemtest_pmcid2pmid = dict()

for group, filepath in NLMCHEMTEST_GROUPS.items():
    d = read_json(filepath)
    for pmcid, pmid in get_pmcid2pmid(d).items():
        assert pmcid not in nlmchemtest_pmcid2pmid
        nlmchemtest_pmcid2pmid[pmcid] = pmid

#
# Write PMCID-PMID mapping to output files.
#
nlmchem_pmcids = sorted(sorted(nlmchem_pmcid2pmid), key=len)

with open('nlmchem_pmcid_pmid.tsv', mode='w', encoding='utf-8') as f:
    for pmcid in nlmchem_pmcids:
        pmid = nlmchem_pmcid2pmid[pmcid]
        _ = f.write('PMC{}\t{}\n'.format(pmcid, pmid))

nlmchemtest_pmcids = sorted(sorted(nlmchemtest_pmcid2pmid), key=len)

with open('nlmchemtest_pmcid_pmid.tsv', mode='w', encoding='utf-8') as f:
    for pmcid in nlmchemtest_pmcids:
        pmid = nlmchemtest_pmcid2pmid[pmcid]
        _ = f.write('PMC{}\t{}\n'.format(pmcid, pmid))
