#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

LOGS = os.path.join(ROOT, 'local', 'logs')


NLM_CHEM = os.path.join(ROOT, 'dataset', 'NLM-CHEM')

NLM_CHEM_GROUPS = {
            'train': os.path.join(NLM_CHEM, 'train/BC7T2-NLMChem-corpus-train.BioC.json'),
                'dev': os.path.join(NLM_CHEM, 'train/BC7T2-NLMChem-corpus-dev.BioC.json'),
                    #'test': os.path.join(NLM_CHEM, 'train/BC7T2-NLMChem-corpus-test.BioC.json')
                    'test': os.path.join(NLM_CHEM, 'train/NLMChemCorpus_test.json')
                    #'test': os.path.join(NLM_CHEM, 'test/BC7T2-NLMChemTest-corpus_v1.BioC.json')
                    }

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
