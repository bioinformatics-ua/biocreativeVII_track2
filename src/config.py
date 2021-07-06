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

NLM_CHEM = os.path.join(ROOT, 'local', 'NLM-Chem')

NLM_CHEM_TRN = os.path.join(NLM_CHEM, 'BC7T2-NLMChem-corpus-train.BioC.json')
NLM_CHEM_DEV = os.path.join(NLM_CHEM, 'BC7T2-NLMChem-corpus-dev.BioC.json')
NLM_CHEM_TST = os.path.join(NLM_CHEM, 'BC7T2-NLMChem-corpus-test.BioC.json')
