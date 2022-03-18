#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import sys
sys.path.extend(['.', '..'])

import os

args = sys.argv[1:]
n = len(args)

if n == 0:
    print()
    print('Please make sure you run the script from the following directory:')
    print('    biocreativeVII_track2/src/annotator/')
    print()
    print('Usage example:')
    print('    $ python3 cli_entities_majority_voting.py file1.json file2.json [...] fileN.json')
    print()
    exit()

input_filepaths = args
for fp in input_filepaths:
    err = '{} is not a valid file path.'.format(repr(fp))
    assert os.path.isfile(fp), err

#
# Make sure you are in the following directory:
# biocreativeVII_track2/src/annotator/
#
cwd = os.getcwd()
cwd = os.path.normpath(cwd)
parents = cwd.split(os.sep)

if parents[-2:] != ['src', 'annotator']:
    print()
    print('Your current working directory:')
    print('    {}'.format(os.path.join(cwd, '')))
    print()
    print('Does not match the expected directory in which the script should be executed:')
    print('    biocreativeVII_track2/src/annotator/')
    print()
    exit()


#
# Create a new Collection with the "most likely" entities according to
# majority voting.
#
from copy import deepcopy
from corpora import Collection
from corpora import get_collection_from_json
from corpora import read_json
from elements import span_overlaps_span


#
# This threshold value between 0.0 and 1.0 specifies how much frequent
# an entity should occur, in the given input collections, to be even
# considered as a prediction.
#
# THRESHOLD = 0.10
# THRESHOLD = 0.30
THRESHOLD = 0.50
# THRESHOLD = 0.70
# THRESHOLD = 0.90

#
# Load input collections.
#
print('Loaded collections:')
input_collections = list()
for fp in input_filepaths:
    input_collections.append(get_collection_from_json(d=read_json(fp),
                                                      ignore_non_contiguous_entities=False,
                                                      ignore_normalization_identifiers=False,
                                                      solve_overlapping_passages=False))
    print('    {}'.format(fp))

print()

n_collections = len(input_collections)
first_collection = input_collections[0]
doc_ids = first_collection.ids()

for c in input_collections:
    assert doc_ids == c.ids(), 'The provided Collections do not share exactly the same Documents Identifiers.'

new_collection = deepcopy(first_collection)
new_collection.clear_entities()
for di in doc_ids:
    hash_to_ne = dict()
    for c in input_collections:
        for ne in c[di].nes():
            ne_hash = hash(ne)
            if ne_hash not in hash_to_ne:
                hash_to_ne[ne_hash] = {'ne': ne, 'count': 0}
            hash_to_ne[ne_hash]['count'] += 1
    
    ne_count_list = [v for k, v in hash_to_ne.items()]
    #
    # Sort the Normalized Entities according to the following three
    # criteria:
    # (1) Start offset. Entities that appear first in the text have
    #     higher priority.
    # (2) Entity span length. Larger spans have higher priority.
    # (3) Counts. The number of collections in which the entity exists.
    #
    sorted_ne_count_list = sorted(ne_count_list, key=lambda x: x['ne'].start)
    sorted_ne_count_list = sorted(sorted_ne_count_list, key=lambda x: x['ne'].n_characters, reverse=True)
    sorted_ne_count_list = sorted(sorted_ne_count_list, key=lambda x: x['count'], reverse=True)
    
    accepted_entities = list()
    for ne_count in sorted_ne_count_list:
        ne = ne_count['ne']
        count = ne_count['count']
        ratio = count / n_collections
        if ratio >= THRESHOLD:
            #
            # Make sure the entity does not overlap with existing
            # entities.
            #
            overlap = False
            for ae in accepted_entities:
                if span_overlaps_span(ne.span, ae.span):
                    overlap = True
                    break
            if not overlap:
                accepted_entities.append(ne)
    
    #
    # Add the accepted entities to the respective passage.
    #
    for ae in accepted_entities:
        added = False
        for p in new_collection[di]:
            if span_overlaps_span(ae.span, p.span):
                p.add_entity(ae)
                added = True
        assert added, 'Entity {} does not overlap with any passage in document {}.'.format(ae, repr(di))


#
# Find available file path.
#
def get_filename(n):
    assert isinstance(n, int) and n > 0
    return '{:03d}'.format(n) + '.json'


root_directory = os.path.abspath(os.path.join(os.path.join(cwd, os.pardir), os.pardir))
output_directory = os.path.join(root_directory, 'outputs', 'annotator')
os.makedirs(output_directory, exist_ok=True)


n = 1
output_filepath = os.path.join(output_directory, get_filename(n))
while os.path.exists(output_filepath):
    n += 1
    output_filepath = os.path.join(output_directory, get_filename(n))


s = new_collection.pretty_json()
with open(output_filepath, mode='w', encoding='utf-8') as f:
    _ = f.write(s)

print('Output collection (entities majority voting) saved:')
print('    {}'.format(output_filepath))
