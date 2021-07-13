#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from copy import deepcopy
import json


#
# A span is a tuple of two integer values (start, end).
# Its aim is to represent the location of a contiguous text span.
#
# Empty spans are valid because empty passages do exist in the NLM-Chem
# dataset.
#
def assert_valid_span(span):
    assert isinstance(span, tuple)
    assert len(span) == 2
    start, end = span
    assert isinstance(start, int) and (start >= 0)
    assert isinstance(end, int) and (end >= start)


def span_overlaps_span(span1, span2):
    assert_valid_span(span1)
    assert_valid_span(span2)
    if (span1[1] <= span2[0]) or (span1[0] >= span2[1]):
        return False
    return True


class Entity:
    #
    # An Entity has a textual mention, a span (start and end offsets), a
    # type, and a set of identifiers (for normalization).
    #
    # Example (taken from NLM-Chem, dev subset, PMCID 4200806):
    #   text: 'tyrosine'
    #   span: (2914, 2922)
    #   typ: 'Chemical'
    #   identifiers: {'MESH:D014443'}
    #
    # >>> Entity('tyrosine', (2914, 2922), 'Chemical', {'MESH:D014443'})
    #
    # Example (taken from NLM-Chem, train subset, PCMCID 5600090):
    #   text: 'MIDA boronate ester'
    #   span: (758, 777)
    #   typ: 'Chemical'
    #   identifier: 'MESH:C533766,MESH:D001897,MESH:D004952'
    #
    # >>> Entity('MIDA boronate ester', (758, 777), 'Chemical', {'MESH:C533766', 'MESH:D001897', 'MESH:D004952'})
    #
    def __init__(self, text, span, typ, identifiers):
        assert isinstance(text, str)
        assert_valid_span(span)
        assert isinstance(typ, str)
        assert isinstance(identifiers, set)
        for i in identifiers:
            assert isinstance(i, str)
        #
        start, end = span
        n_characters = end - start
        assert len(text) == n_characters
        #
        self.text = text
        self.span = span
        self.start = start
        self.end = end
        self.n_characters = n_characters
        self.typ = typ
        self.identifiers = identifiers
        self.identifiers_str = ','.join(sorted(self.identifiers))
    #
    def __repr__(self):
        return 'Entity{}'.format((self.text, self.span, self.typ,
                                  self.identifiers))
    #
    def __len__(self):
        return self.n_characters
    #
    def __eq__(self, other):
        #
        # Compare two entities to check if they have (i) the same text,
        # (ii) the same span, (iii) the same type, and (iv) the same
        # identifiers.
        #
        assert isinstance(other, Entity)
        if ((self.text == other.text) and
            (self.span == other.span) and
            (self.typ == other.typ) and
            (self.identifiers == other.identifiers)):
            return True
        else:
            return False
    #
    def __hash__(self):
        return hash((self.text, self.span, self.typ, self.identifiers_str))
    #
    def __lt__(self, other):
        #
        # This magic method allows to easily sort a list of Entity
        # objects with the sorted() function.
        #
        # Attributes sorting priority:
        #   1. start
        #   2. end
        #   3. typ
        #   4. text
        #   5. identifiers_str
        #
        assert isinstance(other, Entity)
        if self.start < other.start:
            return True
        elif self.start == other.start:
            if self.end < other.end:
                return True
            elif self.end == other.end:
                if self.typ < other.typ:
                    return True
                elif self.typ == other.typ:
                    if self.text < other.text:
                        return True
                    elif self.text == other.text:
                        if self.identifiers_str < other.identifiers_str:
                            return True
        return False
    #
    def json(self):
        return {
            'infons': {
                'identifier': self.identifiers_str,
                'type': self.typ
            },
            'locations': [
                {
                    'length': self.n_characters,
                    'offset': self.start
                }
            ],
            'text': self.text
        }
    #
    def pretty_json(self):
        return json.dumps(self.json(), indent=4, sort_keys=True)


class EntitySet:
    #
    # A set of Entity objects.
    #
    # Example (taken from NLM-Chem, train subset, PCMCID 1253656):
    #
    # >>> e1 = Entity('Carbaryl', (43, 51), 'Chemical', {'MESH:D012721'})
    # >>> e2 = Entity('Naphthalene', (52, 63), 'Chemical', {'MESH:C031721'})
    # >>> e3 = Entity('Chlorpyrifos', (68, 80), 'Chemical', {'MESH:D004390'})
    # >>>
    # >>> es = EntitySet([e1, e2, e3])
    #
    def __init__(self, entities=None):
        self.entities = set()
        if entities is not None:
            assert isinstance(entities, list) or isinstance(entities, set)
            for e in entities:
                self.add(e)
    #
    def __len__(self):
        return len(self.entities)
    #
    def __str__(self):
        s = ''
        for e in self:
            s += '{}\n'.format(e)
        return s.strip()
    #
    def __eq__(self, other):
        assert isinstance(other, EntitySet)
        return self.entities == other.entities
    #
    def __iter__(self):
        for e in sorted(self.entities):
            yield e
    #
    def has(self, e):
        assert isinstance(e, Entity)
        return e in self.entities
    #
    def add(self, e):
        assert isinstance(e, Entity)
        self.entities.add(e)
    #
    def get(self):
        #
        # Return a deepcopy.
        #
        return deepcopy(sorted(self.entities))
    #
    def json(self):
        return [e.json() for e in self]
    #
    def pretty_json(self):
        return json.dumps(self.json(), indent=4, sort_keys=True)


class IndexingIdentifier:
    #
    # An IndexingIdentifier has a single identifier (for example,
    # a MeSH ID), and a type (for example, "MeSH_Indexing_Chemical").
    #
    # Example (taken from NLM-Chem, train subset, PMCID 1253656):
    #   identifier: 'MESH:D009281'
    #   typ: 'MeSH_Indexing_Chemical'
    #
    # >>> IndexingIdentifier('MESH:D009281', 'MeSH_Indexing_Chemical')
    #
    def __init__(self, identifier, typ):
        assert isinstance(identifier, str)
        assert isinstance(typ, str)
        #
        # Only one identifier is allowed.
        #
        assert ',' not in identifier
        #
        self.identifier = identifier
        self.typ = typ
    #
    def __repr__(self):
        return 'IndexingIdentifier{}'.format((self.identifier, self.typ))
    #
    def __eq__(self, other):
        assert isinstance(other, IndexingIdentifier)
        if (self.identifier == other.identifier) and (self.typ == other.typ):
            return True
        else:
            return False
    #
    def __hash__(self):
        return hash((self.identifier, self.typ))
    #
    def __lt__(self, other):
        #
        # This magic method allows to easily sort a list of
        # IndexingIdentifier objects with the sorted() function.
        #
        # Attributes sorting priority:
        #   1. identifier
        #   2. typ
        #
        assert isinstance(other, IndexingIdentifier)
        if self.identifier < other.identifier:
            return True
        elif self.identifier == other.identifier:
            if self.typ < other.typ:
                return True
        return False
    #
    def json(self):
        return {
            'infons': {
                'identifier': self.identifier,
                'type': self.typ
            },
            'locations': [],
            'text': ''
        }
    #
    def pretty_json(self):
        return json.dumps(self.json(), indent=4, sort_keys=True)



class IndexingIdentifierSet:
    #
    # A set of IndexingIdentifier objects.
    #
    # Example (taken from NLM-Chem, train subset, PCMCID 1253656):
    #
    # >>> ii1 = IndexingIdentifier('MESH:D009281', 'MeSH_Indexing_Chemical')
    # >>> ii2 = IndexingIdentifier('MESH:D009284', 'MeSH_Indexing_Chemical')
    # >>> ii3 = IndexingIdentifier('MESH:D011728', 'MeSH_Indexing_Chemical')
    # >>> ii4 = IndexingIdentifier('MESH:C031721', 'MeSH_Indexing_Chemical')
    # >>>
    # >>> iis = IndexingIdentifierSet([ii1, ii2, ii3, ii4])
    #
    def __init__(self, indexing_identifiers=None):
        self.indexing_identifiers = set()
        if indexing_identifiers is not None:
            assert isinstance(indexing_identifiers, list) or isinstance(indexing_identifiers, set)
            for ii in indexing_identifiers:
                self.add(ii)
    #
    def __len__(self):
        return len(self.indexing_identifiers)
    #
    def __str__(self):
        s = ''
        for ii in self:
            s += '{}\n'.format(ii)
        return s.strip()
    #
    def __eq__(self, other):
        assert isinstance(other, IndexingIdentifierSet)
        return self.indexing_identifiers == other.indexing_identifiers
    #
    def __iter__(self):
        for ii in sorted(self.indexing_identifiers):
            yield ii
    #
    def has(self, ii):
        assert isinstance(ii, IndexingIdentifier)
        return ii in self.indexing_identifiers
    #
    def add(self, ii):
        assert isinstance(ii, IndexingIdentifier)
        self.indexing_identifiers.add(ii)
    #
    def get(self):
        #
        # Return a deepcopy.
        #
        return deepcopy(sorted(self.indexing_identifiers))
    #
    def json(self):
        return [ii.json() for ii in self]
    #
    def pretty_json(self):
        return json.dumps(self.json(), indent=4, sort_keys=True)


class Passage:
    #
    # A Passage is initialized with a text, a span (start and end
    # offsets), and a type (abstract, fig_caption, footnote, front,
    # paragraph, ref, table_caption, title, etc).
    #
    # At first, a Passage has no annotations. But these can be added
    # iteratively. Annotations are Entity or IndexingIdentifier objects.
    #
    def __init__(self, text, span, typ):
        assert isinstance(text, str)
        assert_valid_span(span)
        assert isinstance(typ, str)
        #
        start, end = span
        n_characters = end - start
        assert len(text) == n_characters
        #
        self.text = text
        self.span = span
        self.start = start
        self.end = end
        self.n_characters = n_characters
        self.typ = typ
        self.es = EntitySet()
        self.iis = IndexingIdentifierSet()
    #
    def __str__(self):
        s = 'Passage {} {}: {}.'
        return s.format(repr(self.typ), self.span, repr(self.text))
    #
    def add_entity(self, e):
        #
        # Make sure the Entity text matches the Passage text (in the
        # respective Entity span).
        #
        offset = self.start
        e_start = e.start - offset
        e_end = e.end - offset
        assert e_end <= self.n_characters
        assert e.text == self.text[e_start:e_end]
        self.es.add(e)
    #
    def add_entities(self, entities):
        for e in entities:
            self.add_entity(e)
    #
    def add_indexing_identifier(self, ii):
        self.iis.add(ii)
    #
    def add_indexing_identifiers(self, indexing_identifiers):
        for ii in indexing_identifiers:
            self.add_indexing_identifier(ii)
    #
    def entities(self):
        return self.es.get()
    #
    def indexing_identifiers(self):
        return self.iis.get()
    #
    def json(self):
        return {
            'annotations': self.es.json() + self.iis.json(),
            'infons': {
                'type': self.typ
            },
            'offset': self.start,
            'text': self.text
        }
    #
    def pretty_json(self):
        return json.dumps(self.json(), indent=4, sort_keys=True)


class PassageOrderedList:
    #
    # This class contains a list of Passage objects ordered by their
    # span offsets. Also Passage objects cannot overlap, that is, the
    # Passage objects must be disjoint.
    #
    # Example (made-up):
    #
    # >>> p1 = Passage('The title.', (0, 10), 'front')
    # >>> p2 = Passage('An abstract.', (11, 23), 'abstract')
    # >>> p3 = Passage('A first paragraph.', (24, 42), 'paragraph')
    # >>> p4 = Passage('A second paragraph.', (43, 62), 'paragraph')
    # >>>
    # >>> pol = PassageOrderedList()
    # >>> pol.add(p3)
    # >>> pol.add(p2)
    # >>> pol.add(p4)
    # >>> pol.add(p1)
    #
    def __init__(self):
        self.passages = list()
    #
    def __len__(self):
        return len(self.passages)
    #
    def __iter__(self):
        for p in self.passages:
            yield p
    #
    def __str__(self):
        s = ''
        for p in self:
            s += '{}\n'.format(p)
        return s.strip()
    #
    def add(self, p):
        assert isinstance(p, Passage)
        #
        # Assert that this passage does not overlap with any other
        # passage.
        #
        for other in self.passages:
            assert not span_overlaps_span(p.span, other.span)
        #
        # Find the position to insert the current passage so all the
        # passages are ordered by their span offsets.
        #
        found = False
        i = 0
        for i, other in enumerate(self.passages):
            if p.start < other.start:
                found = True
                break
        if found:
            self.passages = self.passages[:i] + [p] + self.passages[i:]
        else:
            self.passages += [p]
    #
    def span(self):
        #
        # Return the span containing all the passages.
        # The start offset is always zero.
        #
        start = 0
        if len(self.passages) == 0:
            end = 0
        else:
            end = self.passages[-1].end
        return (start, end)
    #
    def text(self):
        #
        # Return the whole text containing all the text passages.
        #
        _, n = self.span()
        text = [' '] * n
        #
        for p in self.passages:
            text[p.start:p.end] = p.text
        return ''.join(text)
    #
    def json(self):
        return [p.json() for p in self]
    #
    def pretty_json(self):
        return json.dumps(self.json(), indent=4, sort_keys=True)


class Document:
    #
    # A Document contains an identifier and a PassageOrderedList
    # object.
    #
    def __init__(self, identifier):
        assert isinstance(identifier, str)
        self.identifier = identifier
        self.pol = PassageOrderedList()
    #
    def __str__(self):
        s = 'Document {} with {} passages.'
        return s.format(repr(self.identifier), len(self.pol))
    #
    def add_passage(self, p):
        self.pol.add(p)
    #
    def add_passages(self, passages):
        for p in passages:
            self.add_passage(p)
    #
    def span(self):
        return self.pol.span()
    #
    def text(self):
        return self.pol.text()
    #
    def json(self):
        return {
            'id': self.identifier,
            'passages': self.pol.json()
        }
    #
    def pretty_json(self):
        return json.dumps(self.json(), indent=4, sort_keys=True)


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
    #
    def json(self):
        return {
            'documents': [d.json() for _, d in self]
        }
    #
    def pretty_json(self):
        return json.dumps(self.json(), indent=4, sort_keys=True)


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
