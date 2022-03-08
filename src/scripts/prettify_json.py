#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import sys

import json
import os


args = sys.argv[1:]
n = len(args)

if n != 1:
    print('Usage example: $ python3 prettify_json.py filepath.json')
    exit()

fp = args[0]

with open(fp, mode='r', encoding='utf-8') as f:
    s = json.load(f)

s_pretty = json.dumps(s, indent=4, sort_keys=True)

root, ext = os.path.splitext(fp)
fp2 = root + '-pretty' + ext

if not os.path.exists(fp2):
    with open(fp2, mode='w', encoding='utf-8') as f:
            _ = f.write(s_pretty)
    print('Prettified JSON file saved to:\n    {}'.format(fp2))
else:
    print('Prettified JSON file already exists:\n    {}'.format(fp2))
