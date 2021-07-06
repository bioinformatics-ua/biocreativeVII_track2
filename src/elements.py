#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from copy import deepcopy


class Document:
    #
    #todo
    #
    def __init__(self):
        pass


def sort_identifiers(identifiers):
    #
    # The identifiers are for example a list of PMCIDs.
    #
    assert isinstance(identifiers, list) or isinstance(identifiers, dict)
    for i in identifiers:
        assert isinstance(i, str)
    #
    return sorted(sorted(identifiers), key=len)


class Collection:
    #
    # A set of documents, using the above defined Document class.
    # Each document is associated with a specific identifier.
    # The identifier has to be a string.
    # For example, a PMCID (PubMed Central ID) is a valid identifier.
    #
    def __init__(self):
        self.i2d = dict()
        self.n_documents = 0
    #
    def __len__(self):
        return self.n_documents
    #
    def __getitem__(self, key):
        return self.i2d[key]
    #
    def __iter__(self):
        for i in self.ids():
            yield (i, self[i])
    #
    def add(self, i, d, make_deepcopy=False):
        #
        # Add document with the respective identifier.
        #
        assert isinstance(i, str)
        assert isinstance(d, Document)
        assert i not in self.i2d
        if make_deepcopy:
            d = deepcopy(d)
        self.i2d[i] = d
        self.n_documents += 1
    #
    def get(self, i):
        return self.i2d[i]
    #
    def ids(self):
        #
        # Return a sorted list with the identifiers.
        #
        return sort_identifiers(self.i2d)


def merge_collections(*args, make_deepcopy=False):
    #
    # Merge a list of Collection objects.
    # The collections cannot share equal identifiers. That is, they must
    # be disjoint.
    #
    merged = Collection()
    for c in args:
        assert isinstance(c, Collection)
        for i, d in c:
            merged.add(i, d, make_deepcopy=make_deepcopy)
    return merged
