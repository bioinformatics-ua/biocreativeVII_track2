#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os


#
# Repository root absolute path.
#
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

#
# NLM-Chem corpus (directory path) and subsets (file paths).
#

NLMCHEM = os.path.join(ROOT, 'datasets', 'NLMChem')

NLMCHEM_GROUPS = {
    'train': os.path.join(NLMCHEM, 'BC7T2-NLMChem-corpus-train.BioC.json'),
    'dev': os.path.join(NLMCHEM, 'BC7T2-NLMChem-corpus-dev.BioC.json'),
    'test': os.path.join(NLMCHEM, 'BC7T2-NLMChem-corpus-test.BioC.json')
}

NLMCHEM_PMCID_PMID = os.path.join(NLMCHEM, 'nlmchem_pmcid_pmid.tsv')

NLMCHEMTEST = os.path.join(ROOT, 'datasets', 'NLMChemTest')

NLMCHEMTEST_GROUPS = {
    'test': os.path.join(NLMCHEMTEST, 'BC7T2-NLMChemTest-corpus_v1.BioC.json')
}

NLMCHEMTEST_PMCID_PMID = os.path.join(NLMCHEMTEST, 'nlmchemtest_pmcid_pmid.tsv')

#
# Other official datasets shared by BioCreative VII Track 2 organizers.
#

CDR = os.path.join(ROOT, 'datasets', 'CDR')

CDR_GROUPS = {
    'train': os.path.join(CDR, 'BC7T2-CDR-corpus-train.BioC.json'),
    'dev': os.path.join(CDR, 'BC7T2-CDR-corpus-dev.BioC.json'),
    'test': os.path.join(CDR, 'BC7T2-CDR-corpus-test.BioC.json')
}

CHEMDNER = os.path.join(ROOT, 'datasets', 'CHEMDNER')

CHEMDNER_GROUPS = {
    'train': os.path.join(CHEMDNER, 'BC7T2-CHEMDNER-corpus-training.BioC.json'),
    'dev': os.path.join(CHEMDNER, 'BC7T2-CHEMDNER-corpus-development.BioC.json'),
    'test': os.path.join(CHEMDNER, 'BC7T2-CHEMDNER-corpus-evaluation.BioC.json')
}

#
# DrugProt and DrugProtFiltered datasets.
#

DRUGPROT = os.path.join(ROOT, 'datasets', 'DrugProt')

DRUGPROT_GROUPS = {
    'train': os.path.join(DRUGPROT, 'DrugProt-train.json'),
    'dev': os.path.join(DRUGPROT, 'DrugProt-dev.json'),
}

DRUGPROTFILTERED = os.path.join(ROOT, 'datasets', 'DrugProtFiltered')

DRUGPROTFILTERED_GROUPS = {
    'train': os.path.join(DRUGPROTFILTERED, 'DrugProtFiltered-train.json'),
    'dev': os.path.join(DRUGPROTFILTERED, 'DrugProtFiltered-dev.json'),
}

#
# CTD chemical vocabulary.
# http://ctdbase.org/downloads/
# http://ctdbase.org/reports/CTD_chemicals.tsv.gz
#

CTD_CHEMICALS = os.path.join(ROOT, 'ctdbase', 'CTD_chemicals.tsv')
