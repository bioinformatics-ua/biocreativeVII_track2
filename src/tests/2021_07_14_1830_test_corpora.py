#!/usr/bin/env python3
# -*- coding: utf-8 -*-


#
# Created:
#   2021-07-14, 18:28
#
# Last updated:
#   2021-07-14, 18:46
#

import sys
sys.path.extend(['.', '..'])

from corpora import NLMChemCorpus


corpus = NLMChemCorpus()

for group, collection in corpus.collections.items():
    #
    print(group)
    print()
    #
    for identifier, document in collection:
        print('    {}'.format(identifier))
        print()
        for p, passage in enumerate(document.pol, start=1):
            print('        Passage #{}'.format(p))
            print('        {}'.format(passage))
            for e in passage.entities():
                print('            {}'.format(e))
            for ii in passage.indexing_identifiers():
                print('            {}'.format(ii))
            print()
        #
        _ = input('Next document? ')
        print()
