#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

LOGS = os.path.join(ROOT, 'local', 'logs')


NLM_CHEM = os.path.join(ROOT, 'dataset', 'NLM-CHEM')

NLM_CHEM_GROUPS = {
            'train': os.path.join(NLM_CHEM, 'train/BC7T2-NLMChem-corpus-train.BioC.json'),
                'dev': os.path.join(NLM_CHEM, 'train/BC7T2-NLMChem-corpus-dev.BioC.json'),
                    'test': os.path.join(NLM_CHEM, 'train/BC7T2-NLMChem-corpus-test.BioC.json')
                    }

NLM_CHEM_TEST = os.path.join(ROOT, 'dataset', 'NLM-CHEM')

NLM_CHEM_TEST_GROUPS = {
    #'test': os.path.join(NLM_CHEM_TEST, 'test/BC7T2-NLMChemTest-corpus.BioC.json')
    #'test': os.path.join(NLM_CHEM_TEST, 'test/Track2-Team-110-Subtask1-Run-1.json')
    #'test': os.path.join(NLM_CHEM_TEST, 'test/Track2-Team-110-Subtask1-Run-2.json')
    #'test': os.path.join(NLM_CHEM_TEST, 'test/Track2-Team-110-Subtask1-Run-5.json')
    #'test': os.path.join(NLM_CHEM_TEST, 'test/Track2-Team-110-Subtask1-Run-4.json')
    'test': os.path.join(NLM_CHEM_TEST, 'test/1_deft-blaze_nlm_chem_test_bioc.json')
}

NLM_CHEM_TEST_PMCID_PMID = os.path.join(NLM_CHEM_TEST, 'test/BC7T2-NLMChemTest-ids.tsv')

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
