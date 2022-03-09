#!/usr/bin/env python3
# -*- coding: utf-8 -*-


#
# Convert the DrugProt dataset to JSON format.
# The DrugProt dataset follows the same structure as the ChemProt
# dataset. However, the DrugProt already contains the ChemProt dataset,
# so there is no need to convert the ChemProt dataset.
#
#
# DrugProt:
# https://biocreative.bioinformatics.udel.edu/tasks/biocreative-vii/track-1/
# Download link:
# https://zenodo.org/record/5042151
# https://zenodo.org/record/5042151/files/drugprot-gs-training-development.zip
#
#
# ChemProt (not needed, because it is included in the DrugProt dataset):
# https://biocreative.bioinformatics.udel.edu/tasks/biocreative-vi/track-5/
# Download link:
# https://biocreative.bioinformatics.udel.edu/resources/corpora/chemprot-corpus-biocreative-vi/
# https://biocreative.bioinformatics.udel.edu/media/store/files/2017/ChemProt_Corpus.zip
#
#
# Notes:
# - The ChemProt is included in the training and development subsets of
#   the DrugProt dataset.
# - Regarding the DrugProt we only consider the training and development
#   subsets, because at the time of writing (2021-08-19) we "don't have
#   access" to the gold standard test subset. We have the test subset
#   (750 records) but it is mixed with further 10.000 files (background
#   set), and therefore we don't know which 750 files are the true gold
#   standard.
# - We will create two different datasets: (i) DrugProt and
#   (ii) DrugProtFiltered. The first will have all PMIDs, and the second
#   one will have only the PMIDs that do not appear in the following
#   datasets: CDR, CHEMDNER, NLM-Chem, and NLM-Chem-Test.
#   This is to avoid repeated documents in different datasets.
#


import sys
sys.path.extend(['.', '..'])

import os

from annotator.config import NLMCHEM_PMCID_PMID
from annotator.config import NLMCHEMTEST_PMCID_PMID
from annotator.corpora import CDRCorpus
from annotator.corpora import CHEMDNERCorpus
from annotator.elements import Collection
from annotator.elements import Document
from annotator.elements import NormalizedEntity
from annotator.elements import NormalizedEntitySet
from annotator.elements import Passage
from annotator.elements import span_contains_span


args = sys.argv[1:]
n = len(args)

if n != 1:
    print('Usage examples:')
    print()
    print('    $ python3 convert_drugprot_to_json.py /path/to/drugprot-gs-training-development/training/')
    print('    $ python3 convert_drugprot_to_json.py /path/to/drugprot-gs-training-development/development/')
    print()
    exit()

dpath = os.path.abspath(args[0])
par_dpath, dname = os.path.split(dpath)

drugprot_fpath = os.path.join(par_dpath, 'DrugProt-' + dname + '.json')
drugprotfiltered_fpath = os.path.join(par_dpath, 'DrugProtFiltered-' + dname + '.json')
assert not os.path.exists(drugprot_fpath), '{} already exists.'.format(repr(drugprot_fpath))
assert not os.path.exists(drugprotfiltered_fpath), '{} already exists.'.format(repr(drugprotfiltered_fpath))

#
# Get the unique PMIDs from the following datasets:
# - CDR
# - CHEMDNER
# - NLMChem
# - NLMChemTest
#
# These PMIDS will be ignored in the DrugProtFiltered dataset.
#
pmids_to_ignore = set()

cdr = CDRCorpus()
chemdner = CHEMDNERCorpus()

for corpus in [cdr, chemdner]:
    for group, collection in corpus:
        pmids_to_ignore.update(collection.ids())


def read_pmids(fp):
    with open(fp, mode='r', encoding='utf-8') as f:
        pmids = set()
        for line in f:
            pmcid, pmid = line.strip().split('\t')
            pmids.add(pmid)
    return pmids

pmids_to_ignore.update(read_pmids(NLMCHEM_PMCID_PMID))
pmids_to_ignore.update(read_pmids(NLMCHEMTEST_PMCID_PMID))

#
# Find the two necessary files: (1) abstracts and (2) entities.
#
for fname in os.listdir(dpath):
    if fname.endswith('.tsv'):
        if 'abstr' in fname:
            abstracts_fpath = os.path.join(dpath, fname)
        if 'entit' in fname:
            entities_fpath = os.path.join(dpath, fname)

assert os.path.exists(abstracts_fpath)
assert os.path.exists(entities_fpath)


def get_collection_from_abstracts(fpath, pmids_to_ignore=None):
    
    if pmids_to_ignore is None:
        pmids_to_ignore = set()
    
    c = Collection()
    
    with open(fpath, mode='r', encoding='utf-8') as f:
        for line in f:
            pmid, title, abstract = line.strip().split('\t')
            pmid = pmid.strip()
            title = title.strip()
            abstract = abstract.strip()
            
            if pmid not in pmids_to_ignore:
                doc = Document(identifier=pmid)
                
                #
                # Add title passage.
                #
                title_span = (0, len(title))
                p = Passage(title, title_span, typ='title', section_type='')
                doc.add_passage(p)
                
                #
                # Add abstract passage.
                #
                offset = title_span[-1] + 1
                abstract_span = (offset, offset + len(abstract))
                p = Passage(abstract, abstract_span, typ='abstract', section_type='')
                doc.add_passage(p)
                
                c.add(pmid, doc)
    
    return c


def read_entities(fpath):
    
    pmid2nes = dict()
    
    with open(fpath, mode='r', encoding='utf-8') as f:
        for line in f:
            
            pmid, tn, typ, start, end, text = line.strip().split('\t')
            
            pmid = pmid.strip()
            tn = tn.strip()
            typ = typ.strip()
            start = start.strip()
            end = end.strip()
            text = text.strip()
            
            if typ == 'CHEMICAL':
                
                if pmid not in pmid2nes:
                    pmid2nes[pmid] = NormalizedEntitySet()
                
                start = int(start)
                end = int(end)
                
                e = NormalizedEntity(text, (start, end), typ='Chemical')
                
                pmid2nes[pmid].add(e)
    
    return pmid2nes


drugprot_collection = get_collection_from_abstracts(abstracts_fpath)
drugprotfiltered_collection = get_collection_from_abstracts(abstracts_fpath, pmids_to_ignore=pmids_to_ignore)
pmid2nes = read_entities(entities_fpath)

collections = [drugprot_collection, drugprotfiltered_collection]
out_fpaths = [drugprot_fpath, drugprotfiltered_fpath]

for c, fp in zip(collections, out_fpaths):
    #
    # Add entities to collection.
    #
    for di, doc in c:
        #
        # Not all documents have Chemical entities.
        #
        if di in pmid2nes:
            nes = pmid2nes[di]
            #
            # Add the entities in the respective passages.
            #
            for e in nes:
                found = False
                for p in doc:
                    if span_contains_span(p.span, e.span):
                        assert found is False, '{} was already found in a different passage.'.format(e)
                        p.add_entity(e)
                        found = True
                assert found, 'No valid Passage was found for adding {}.'.format(e)
    
    s = c.pretty_json()
    
    with open(fp, mode='w', encoding='utf-8') as f:
        _ = f.write(s)
