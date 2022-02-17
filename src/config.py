#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os


#
# Repository root absolute path.
#
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#
# Logs files (results).
#
LOGS = os.path.join(ROOT, 'local', 'logs')

#
# NLM-Chem corpus (directory path) and subsets (file paths).
#

NLM_CHEM = os.path.join(ROOT, 'dataset', 'NLM-CHEM')

NLM_CHEM_GROUPS = {
    'train': os.path.join(NLM_CHEM, 'train/BC7T2-NLMChem-corpus-train.BioC.json'),
    'dev': os.path.join(NLM_CHEM, 'train/BC7T2-NLMChem-corpus-dev.BioC.json'),
    'test': os.path.join(NLM_CHEM, 'train/BC7T2-NLMChem-corpus-test.BioC.json')
    # 'test': os.path.join(NLM_CHEM, 'train/NLMChemCorpus_test(1).json')
}

NLM_CHEM_PMCID_PMID = os.path.join(NLM_CHEM, 'train/nlmchem_pmcid_pmid.tsv')

NLM_CHEM_TEST = os.path.join(ROOT, 'dataset', 'NLM-CHEM')

NLM_CHEM_TEST_GROUPS = {
    #'test': os.path.join(NLM_CHEM_TEST, 'test/BC7T2-NLMChemTest-corpus.BioC.json')
    'test': os.path.join(NLM_CHEM_TEST, 'test/deft-blaze-4-run-NLMChemTestCorpus_test.json') 
}

NLM_CHEM_TEST_PMCID_PMID = os.path.join(NLM_CHEM_TEST, 'test/BC7T2-NLMChemTest-ids_v1.tsv')




#
# Other official datasets shared by BioCreative VII Track 2 organizers.
#

CDR = os.path.join(ROOT, 'local', 'datasets', 'CDR')

CDR_GROUPS = {
    'train': os.path.join(CDR, 'BC7T2-CDR-corpus-train.BioC.json'),
    'dev': os.path.join(CDR, 'BC7T2-CDR-corpus-dev.BioC.json'),
    'test': os.path.join(CDR, 'BC7T2-CDR-corpus-test.BioC.json')
}

CHEMDNER = os.path.join(ROOT, 'local', 'datasets', 'CHEMDNER')

CHEMDNER_GROUPS = {
    'train': os.path.join(CHEMDNER, 'BC7T2-CHEMDNER-corpus-training.BioC.json'),
    'dev': os.path.join(CHEMDNER, 'BC7T2-CHEMDNER-corpus-development.BioC.json'),
    'test': os.path.join(CHEMDNER, 'BC7T2-CHEMDNER-corpus-evaluation.BioC.json')
}

#
# Corpora derived from:
# https://github.com/cambridgeltl/MTL-Bioinformatics-2016
#

BC5CDR = os.path.join(ROOT, 'local', 'datasets', 'BC5CDR')

BC5CDR_GROUPS = {
    'train': os.path.join(BC5CDR, 'BC5CDR-train.json'),
    'dev': os.path.join(BC5CDR, 'BC5CDR-dev.json'),
    'test': os.path.join(BC5CDR, 'BC5CDR-test.json'),
}

BIONLP11ID = os.path.join(ROOT, 'local', 'datasets', 'BioNLP11ID')

BIONLP11ID_GROUPS = {
    'train': os.path.join(BIONLP11ID, 'BioNLP11ID-train.json'),
    'dev': os.path.join(BIONLP11ID, 'BioNLP11ID-dev.json'),
    'test': os.path.join(BIONLP11ID, 'BioNLP11ID-test.json'),
}

BIONLP13CG = os.path.join(ROOT, 'local', 'datasets', 'BioNLP13CG')

BIONLP13CG_GROUPS = {
    'train': os.path.join(BIONLP13CG, 'BioNLP13CG-train.json'),
    'dev': os.path.join(BIONLP13CG, 'BioNLP13CG-dev.json'),
    'test': os.path.join(BIONLP13CG, 'BioNLP13CG-test.json'),
}

BIONLP13PC = os.path.join(ROOT, 'local', 'datasets', 'BioNLP13PC')

BIONLP13PC_GROUPS = {
    'train': os.path.join(BIONLP13PC, 'BioNLP13PC-train.json'),
    'dev': os.path.join(BIONLP13PC, 'BioNLP13PC-dev.json'),
    'test': os.path.join(BIONLP13PC, 'BioNLP13PC-test.json'),
}

CRAFT = os.path.join(ROOT, 'local', 'datasets', 'CRAFT')

CRAFT_GROUPS = {
    'train': os.path.join(CRAFT, 'CRAFT-train.json'),
    'dev': os.path.join(CRAFT, 'CRAFT-dev.json'),
    'test': os.path.join(CRAFT, 'CRAFT-test.json'),
}

#
# DrugProt and DrugProtFiltered datasets.
#

DRUGPROT = os.path.join(ROOT, 'local', 'datasets', 'DrugProt')

DRUGPROT_GROUPS = {
    'train': os.path.join(DRUGPROT, 'DrugProt-train.json'),
    'dev': os.path.join(DRUGPROT, 'DrugProt-dev.json'),
}

DRUGPROTFILTERED = os.path.join(ROOT, 'local', 'datasets', 'DrugProtFiltered')

DRUGPROTFILTERED_GROUPS = {
    'train': os.path.join(DRUGPROTFILTERED, 'DrugProtFiltered-train.json'),
    'dev': os.path.join(DRUGPROTFILTERED, 'DrugProtFiltered-dev.json'),
}