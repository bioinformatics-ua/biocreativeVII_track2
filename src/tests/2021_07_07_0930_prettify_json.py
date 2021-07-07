#!/usr/bin/env python3
# -*- coding: utf-8 -*-


#
# Created:
#   2021-07-07, 09:30
#
# Last updated:
#   2021-07-07, 10:05
#   2021-07-07, 09:35
#

#
# References:
# https://stackoverflow.com/questions/12943819/how-to-prettyprint-a-json-file
#

import sys
sys.path.extend(['.', '..'])

import json
import os

from config import NLM_CHEM_TRN
from config import NLM_CHEM_DEV
from config import NLM_CHEM_TST


#
# Choose here if you want to output (write to disk) a prettified version
# of the JSON files.
#
WRITE_PRETTY_JSON = False
# WRITE_PRETTY_JSON = True

filepaths = [NLM_CHEM_TRN, NLM_CHEM_DEV, NLM_CHEM_TST]

for fp in filepaths:
    #
    with open(fp, mode='r', encoding='utf-8') as f:
        s = json.load(f)
    #
    s_pretty = json.dumps(s, indent=4, sort_keys=True)
    print(fp)
    print(s_pretty[:180], '...\n')
    #
    if WRITE_PRETTY_JSON:
        head, tail = os.path.split(fp)
        if not os.path.exists(tail):
            with open(tail, mode='w', encoding='utf-8') as f:
                    _ = f.write(s_pretty)
