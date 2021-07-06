#!/usr/bin/env python3
# -*- coding: utf-8 -*-


#
# Created:
#   2021-07-06, 16:50
#
# Last updated:
#   2021-07-06, 16:51
#

import sys
sys.path.extend(['.', '..'])

from config import ROOT
from config import LOGS
from config import NLM_CHEM
from config import NLM_CHEM_TRN
from config import NLM_CHEM_DEV
from config import NLM_CHEM_TST


#
# To make sure paths are correct.
#

sf = '{}:\n    {}\n'

print(sf.format('ROOT', ROOT))
print(sf.format('LOGS', LOGS))
print(sf.format('NLM_CHEM', NLM_CHEM))
print(sf.format('NLM_CHEM_TRN', NLM_CHEM_TRN))
print(sf.format('NLM_CHEM_DEV', NLM_CHEM_DEV))
print(sf.format('NLM_CHEM_TST', NLM_CHEM_TST))
