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
    else:
        return True


def span_contains_span(span1, span2):
    assert_valid_span(span1)
    assert_valid_span(span2)
    if (span1[0] <= span2[0]) and (span1[1] >= span2[1]):
        return True
    else:
        return False


def add_offset_to_spans(spans, offset):
    return [
        (start + offset, end + offset)
        for (start, end) in spans
    ]


class Entity:
    r"""
    An Entity has a textual mention, a span (start and end offsets),
    and a type.
    
    Example (taken from NLM-Chem, dev subset, PMCID 4200806):
        text: 'tyrosine'
        span: (2914, 2922)
        typ: 'Chemical'
    
    >>> Entity('tyrosine', (2914, 2922), 'Chemical')
    
    Example (taken from NLM-Chem, train subset, PMCID 5600090):
        text: 'MIDA boronate ester'
        span: (758, 777)
        typ: 'Chemical'
    
    >>> Entity('MIDA boronate ester', (758, 777), 'Chemical')
    """
    
    def __init__(self, text, span, typ):
        assert isinstance(text, str)
        assert_valid_span(span)
        assert isinstance(typ, str)
        
        start, end = span
        n_characters = end - start
        assert len(text) == n_characters
        
        self.text = text
        self.span = span
        self.start = start
        self.end = end
        self.n_characters = n_characters
        self.typ = typ
    
    def __repr__(self):
        return 'Entity{}'.format((self.text, self.span, self.typ))
    
    def __len__(self):
        return self.n_characters
    
    def __eq__(self, other):
        r"""
        Compare two entities to check if they have:
            - the same text
            - the same span
            - the same type
        """
        assert type(other) is Entity
        if ((self.text == other.text) and
            (self.span == other.span) and
            (self.typ == other.typ)):
            return True
        else:
            return False
    
    def __hash__(self):
        return hash((self.text, self.span, self.typ))
    
    def __lt__(self, other):
        r"""
        This magic method allows to easily sort a list of Entity
        objects with the sorted() function.
        
        Attributes sorting priority:
            1. start
            2. end
            3. typ
            4. text
        """
        assert type(other) is Entity
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
        return False
    
    def to_normalized_entity(self):
        return NormalizedEntity(self.text, self.span, self.typ)
    
    def json(self, i=1):
        assert isinstance(i, int)
        return {
            'id': str(i),
            'infons': {
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
    
    def pretty_json(self, i=1):
        return json.dumps(self.json(i), indent=4, sort_keys=True)


class EntitySet:
    r"""
    A set of Entity objects.
    
    Example (taken from NLM-Chem, train subset, PMCID 1253656):
    
    >>> e1 = Entity('Carbaryl', (43, 51), 'Chemical')
    >>> e2 = Entity('Naphthalene', (52, 63), 'Chemical')
    >>> e3 = Entity('Chlorpyrifos', (68, 80), 'Chemical')
    >>>
    >>> es = EntitySet([e1, e2, e3])
    """
    
    def __init__(self, entities=None):
        self.entities = set()
        if entities is not None:
            self.update(entities)
    
    def __len__(self):
        return len(self.entities)
    
    def __str__(self):
        s = ''
        for e in self.get(sort=True, make_deepcopy=False):
            s += '{}\n'.format(e)
        return s.strip()
    
    def __eq__(self, other):
        assert type(other) is EntitySet
        return self.entities == other.entities
    
    def __iter__(self):
        for e in self.entities:
            yield e
    
    def has(self, e):
        assert type(e) is Entity
        return e in self.entities
    
    def add(self, e):
        assert type(e) is Entity
        self.entities.add(e)
    
    def update(self, entities):
        assert isinstance(entities, list) or isinstance(entities, set)
        for e in entities:
            self.add(e)
    
    def get(self, sort=False, make_deepcopy=False):
        entities = self.entities
        if sort:
            entities = sorted(entities)
        if make_deepcopy:
            entities = deepcopy(entities)
        return entities
    
    def union(self, other, make_deepcopy=False):
        assert type(other) is EntitySet
        es = EntitySet(self.entities.union(other.entities))
        if make_deepcopy:
            es = deepcopy(es)
        return es
    
    def intersection(self, other, make_deepcopy=False):
        assert type(other) is EntitySet
        es = EntitySet(self.entities.intersection(other.entities))
        if make_deepcopy:
            es = deepcopy(es)
        return es
    
    def difference(self, other, make_deepcopy=False):
        assert type(other) is EntitySet
        es = EntitySet(self.entities.difference(other.entities))
        if make_deepcopy:
            es = deepcopy(es)
        return es
    
    def to_normalized_entity_set(self):
        return NormalizedEntitySet([e.to_normalized_entity() for e in self.entities])
    
    def json(self, start=1):
        entities = self.get(sort=True, make_deepcopy=False)
        return [e.json(i) for i, e in enumerate(entities, start=start)]
    
    def pretty_json(self, start=1):
        return json.dumps(self.json(start), indent=4, sort_keys=True)


class NormalizedEntity(Entity):
    r"""
    A NormalizedEntity inherits from Entity.
    
    Besides having a textual mention, a span, and a type, additionally
    has a list of identifiers (for normalization).
    
    Note: it has to be a list of identifiers (not a set) because the
          order of the identifiers matters. For example, if the entity
          text mention refers to three diferent terms, its identifiers
          follow their order of appearance (see Example 2 below).
    
    Example 1 (taken from NLM-Chem, dev subset, PMCID 4200806):
        text: 'tyrosine'
        span: (2914, 2922)
        typ: 'Chemical'
        identifiers: ['MESH:D014443']
    
    >>> NormalizedEntity('tyrosine', (2914, 2922), 'Chemical', ['MESH:D014443'])
    
    Example 2 (taken from NLM-Chem, train subset, PMCID 5600090):
        text: 'MIDA boronate ester'
        span: (758, 777)
        typ: 'Chemical'
        identifiers: ['MESH:C533766', 'MESH:D001897', 'MESH:D004952']
    
    >>> NormalizedEntity('MIDA boronate ester', (758, 777), 'Chemical', ['MESH:C533766', 'MESH:D001897', 'MESH:D004952'])
    
    Example 3 (taken from NLM-Chem, train subset, PMCID 4988499)
        text: 'cyclic, aromatic, and monoterpenoid enones, enals, and enols'
        span: (2793, 2853)
        typ: 'Chemical'
        identifiers: ['MESH:D007659', 'MESH:D000447', '-']
    
    >>> NormalizedEntity('cyclic, aromatic, and monoterpenoid enones, enals, and enols', (2793, 2853), 'Chemical', ['MESH:D007659', 'MESH:D000447', '-'])
    """
    
    def __init__(self, text, span, typ, identifiers=None):
        super().__init__(text, span, typ)
        self.set_identifiers(identifiers)
    
    def set_identifiers(self, identifiers):
        if identifiers is None:
            identifiers = list()
        
        assert isinstance(identifiers, list)
        for i in identifiers:
            assert isinstance(i, str)
        #
        # If the identifiers list is empty, it means there are no
        # identifiers.
        # In this case, add the '-' identifier meaning there is no
        # identifier.
        #
        if len(identifiers) == 0:
            identifiers = ['-']
        
        self.identifiers = identifiers
        self.identifiers_str = ','.join(identifiers)
    
    def __repr__(self):
        return 'NormalizedEntity{}'.format((self.text, self.span, self.typ,
                                            self.identifiers))
    
    def __eq__(self, other):
        r"""
        Compare two normalized entities to check if they have:
            - the same text
            - the same span
            - the same type
            - the same identifiers
        """
        assert type(other) is NormalizedEntity
        if ((self.text == other.text) and
            (self.span == other.span) and
            (self.typ == other.typ) and
            (self.identifiers == other.identifiers)):
            return True
        else:
            return False
    
    def __hash__(self):
        return hash((self.text, self.span, self.typ, self.identifiers_str))
    
    def __lt__(self, other):
        r"""
        This magic method allows to easily sort a list of
        NormalizedEntity objects with the sorted() function.
        
        Attributes sorting priority:
            1. start
            2. end
            3. typ
            4. text
            5. identifiers_str
        """
        assert type(other) is NormalizedEntity
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
    
    def to_entity(self):
        return Entity(self.text, self.span, self.typ)
    
    def json(self, i=1):
        assert isinstance(i, int)
        return {
            'id': str(i),
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


class NormalizedEntitySet(EntitySet):
    r"""
    A set of NormalizedEntity objects.
    
    Example (taken from NLM-Chem, train subset, PMCID 1253656):
    
    >>> ne1 = NormalizedEntity('Carbaryl', (43, 51), 'Chemical', ['MESH:D012721'])
    >>> ne2 = NormalizedEntity('Naphthalene', (52, 63), 'Chemical', ['MESH:C031721'])
    >>> ne3 = NormalizedEntity('Chlorpyrifos', (68, 80), 'Chemical', ['MESH:D004390'])
    >>>
    >>> nes = NormalizedEntitySet([ne1, ne2, ne3])
    """
    
    def __init__(self, entities=None):
        super().__init__(entities)
    
    def __eq__(self, other):
        assert type(other) is NormalizedEntitySet
        return self.entities == other.entities
    
    def has(self, e):
        assert type(e) is NormalizedEntity
        return e in self.entities
    
    def add(self, e):
        assert type(e) is NormalizedEntity
        self.entities.add(e)
    
    def union(self, other, make_deepcopy=False):
        assert type(other) is NormalizedEntitySet
        nes = NormalizedEntitySet(self.entities.union(other.entities))
        if make_deepcopy:
            nes = deepcopy(nes)
        return nes
    
    def intersection(self, other, make_deepcopy=False):
        assert type(other) is NormalizedEntitySet
        nes = NormalizedEntitySet(self.entities.intersection(other.entities))
        if make_deepcopy:
            nes = deepcopy(nes)
        return nes
    
    def difference(self, other, make_deepcopy=False):
        assert type(other) is NormalizedEntitySet
        nes = NormalizedEntitySet(self.entities.difference(other.entities))
        if make_deepcopy:
            nes = deepcopy(nes)
        return nes
    
    def to_entity_set(self):
        return EntitySet([e.to_entity() for e in self.entities])


class IndexingIdentifier:
    r"""
    An IndexingIdentifier has a single identifier (for example,
    a MeSH ID), and a type (for example, "MeSH_Indexing_Chemical").
    
    Example (taken from NLM-Chem, train subset, PMCID 1253656):
      identifier: 'MESH:D009281'
      typ: 'MeSH_Indexing_Chemical'
    
    >>> IndexingIdentifier('MESH:D009281', 'MeSH_Indexing_Chemical')
    """
    
    def __init__(self, identifier, typ):
        assert isinstance(identifier, str)
        assert isinstance(typ, str)
        #
        # Only one identifier is allowed.
        #
        assert ',' not in identifier
        assert '|' not in identifier
        
        self.identifier = identifier
        self.typ = typ
    
    def __repr__(self):
        return 'IndexingIdentifier{}'.format((self.identifier, self.typ))
    
    def __eq__(self, other):
        assert isinstance(other, IndexingIdentifier)
        if (self.identifier == other.identifier) and (self.typ == other.typ):
            return True
        else:
            return False
    
    def __hash__(self):
        return hash((self.identifier, self.typ))
    
    def __lt__(self, other):
        r"""
        This magic method allows to easily sort a list of
        IndexingIdentifier objects with the sorted() function.
        
        Attributes sorting priority:
          1. identifier
          2. typ
        """
        assert isinstance(other, IndexingIdentifier)
        if self.identifier < other.identifier:
            return True
        elif self.identifier == other.identifier:
            if self.typ < other.typ:
                return True
        return False
    
    def json(self, i=1):
        assert isinstance(i, int)
        
        if self.typ == 'MeSH_Indexing_Chemical':
            prefix = 'MIC'
        else:
            prefix = 'II'
        
        return {
            'id': '{}{}'.format(prefix, i),
            'infons': {
                'identifier': self.identifier,
                'type': self.typ
            },
            'locations': [],
            'text': ''
        }
    
    def pretty_json(self, i=1):
        return json.dumps(self.json(i), indent=4, sort_keys=True)


class IndexingIdentifierSet:
    r"""
    A set of IndexingIdentifier objects.
    
    Example (taken from NLM-Chem, train subset, PMCID 1253656):
    
    >>> ii1 = IndexingIdentifier('MESH:D009281', 'MeSH_Indexing_Chemical')
    >>> ii2 = IndexingIdentifier('MESH:D009284', 'MeSH_Indexing_Chemical')
    >>> ii3 = IndexingIdentifier('MESH:D011728', 'MeSH_Indexing_Chemical')
    >>> ii4 = IndexingIdentifier('MESH:C031721', 'MeSH_Indexing_Chemical')
    >>>
    >>> iis = IndexingIdentifierSet([ii1, ii2, ii3, ii4])
    """
    
    def __init__(self, indexing_identifiers=None):
        self.indexing_identifiers = set()
        if indexing_identifiers is not None:
            self.update(indexing_identifiers)
    
    def __len__(self):
        return len(self.indexing_identifiers)
    
    def __str__(self):
        s = ''
        for ii in self.get(sort=True, make_deepcopy=False):
            s += '{}\n'.format(ii)
        return s.strip()
    
    def __eq__(self, other):
        assert isinstance(other, IndexingIdentifierSet)
        return self.indexing_identifiers == other.indexing_identifiers
    
    def __iter__(self):
        for ii in self.indexing_identifiers:
            yield ii
    
    def has(self, ii):
        assert isinstance(ii, IndexingIdentifier)
        return ii in self.indexing_identifiers
    
    def add(self, ii):
        assert isinstance(ii, IndexingIdentifier)
        self.indexing_identifiers.add(ii)
    
    def update(self, indexing_identifiers):
        assert isinstance(indexing_identifiers, list) or isinstance(indexing_identifiers, set)
        for ii in indexing_identifiers:
            self.add(ii)
    
    def get(self, sort=False, make_deepcopy=False):
        indexing_identifiers = self.indexing_identifiers
        if sort:
            indexing_identifiers = sorted(indexing_identifiers)
        if make_deepcopy:
            indexing_identifiers = deepcopy(indexing_identifiers)
        return indexing_identifiers
    
    def union(self, other, make_deepcopy=False):
        assert isinstance(other, IndexingIdentifierSet)
        iis = IndexingIdentifierSet(self.indexing_identifiers.union(other.indexing_identifiers))
        if make_deepcopy:
            iis = deepcopy(iis)
        return iis
    
    def intersection(self, other, make_deepcopy=False):
        assert isinstance(other, IndexingIdentifierSet)
        iis = IndexingIdentifierSet(self.indexing_identifiers.intersection(other.indexing_identifiers))
        if make_deepcopy:
            iis = deepcopy(iis)
        return iis
    
    def difference(self, other, make_deepcopy=False):
        assert isinstance(other, IndexingIdentifierSet)
        iis = IndexingIdentifierSet(self.indexing_identifiers.difference(other.indexing_identifiers))
        if make_deepcopy:
            iis = deepcopy(iis)
        return iis
    
    def json(self, start=1):
        indexing_identifiers = self.get(sort=True, make_deepcopy=False)
        return [ii.json(i) for i, ii in enumerate(indexing_identifiers, start=start)]
    
    def pretty_json(self, start=1):
        return json.dumps(self.json(start), indent=4, sort_keys=True)


class Passage:
    r"""
    A Passage is initialized with a text, a span (start and end
    offsets), a type (abstract, fig_caption, footnote, front,
    paragraph, ref, table_caption, title, etc), and a section type
    (ABSTRACT, INTRO, METHODS, RESULTS, etc). Note that, in the
    NLM-Chem dataset, frequently the section types are undefined.
    
    At first, a Passage has no annotations. But these can be added
    iteratively. Annotations are NormalizedEntity or IndexingIdentifier
    objects.
    """
    
    def __init__(self, text, span, typ, section_type):
        assert isinstance(text, str)
        assert_valid_span(span)
        assert isinstance(typ, str)
        assert isinstance(section_type, str)
        
        start, end = span
        n_characters = end - start
        assert len(text) == n_characters
        
        self.text = text
        self.span = span
        self.start = start
        self.end = end
        self.n_characters = n_characters
        self.typ = typ
        self.section_type = section_type
        self.nes = NormalizedEntitySet()
        self.iis = IndexingIdentifierSet()
    
    def __str__(self):
        s = 'Passage {} ({}, {}): {}.'
        return s.format(self.span, repr(self.typ),
                        repr(self.section_type), repr(self.text))
    
    def add_entity(self, e):
        assert type(e) is NormalizedEntity
        #
        # Make sure the NormalizedEntity text matches the Passage text
        # (in the respective NormalizedEntity span).
        #
        offset = self.start
        e_start = e.start - offset
        e_end = e.end - offset
        assert e_end <= self.n_characters
        assert e.text == self.text[e_start:e_end]
        self.nes.add(e)
    
    def add_entities(self, entities):
        for e in entities:
            self.add_entity(e)
    
    def add_indexing_identifier(self, ii):
        self.iis.add(ii)
    
    def add_indexing_identifiers(self, indexing_identifiers):
        for ii in indexing_identifiers:
            self.add_indexing_identifier(ii)
    
    def entities(self, sort=False, make_deepcopy=False):
        return self.nes.get(sort=sort, make_deepcopy=make_deepcopy)
    
    def indexing_identifiers(self, sort=False, make_deepcopy=False):
        return self.iis.get(sort=sort, make_deepcopy=make_deepcopy)
    
    def get_entity_set(self):
        return self.nes.to_entity_set()
    
    def json(self, nes_i=1, iis_i=1):
        return {
            'annotations': self.nes.json(nes_i) + self.iis.json(iis_i),
            'infons': {
                'section_type': self.section_type,
                'type': self.typ
            },
            'offset': self.start,
            'text': self.text
        }
    
    def pretty_json(self, nes_i=1, iis_i=1):
        return json.dumps(self.json(nes_i, iis_i), indent=4, sort_keys=True)


class PassageOrderedList:
    r"""
    This class contains a list of Passage objects ordered by their
    span offsets. Also Passage objects cannot overlap, that is, the
    Passage objects must be disjoint.
    
    Example (made-up):
    
    >>> p1 = Passage('The title.', (0, 10), 'front', 'TITLE')
    >>> p2 = Passage('An abstract.', (11, 23), 'abstract', 'ABSTRACT')
    >>> p3 = Passage('A first paragraph.', (24, 42), 'paragraph', 'INTRO')
    >>> p4 = Passage('A second paragraph.', (43, 62), 'paragraph', 'INTRO')
    >>>
    >>> pol = PassageOrderedList()
    >>> pol.add(p3)
    >>> pol.add(p2)
    >>> pol.add(p4)
    >>> pol.add(p1)
    """
    
    def __init__(self):
        self.passages = list()
    
    def __len__(self):
        return len(self.passages)
    
    def __iter__(self):
        for p in self.passages:
            yield p
    
    def __str__(self):
        s = ''
        for p in self:
            s += '{}\n'.format(p)
        return s.strip()
    
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
    
    def span(self):
        r"""
        Return the span containing all the passages.
        The start offset is always zero.
        """
        start = 0
        if len(self.passages) == 0:
            end = 0
        else:
            end = self.passages[-1].end
        return (start, end)
    
    def text(self):
        r"""
        Return the whole text containing all the text passages.
        """
        _, n = self.span()
        text = [' '] * n
        
        for p in self.passages:
            text[p.start:p.end] = p.text
        
        return ''.join(text)
    
    def passages_texts(self):
        passages_texts = list()
        for passage in self:
            passages_texts.append(passage.text)
        return passages_texts
    
    def passages_spans(self):
        passages_spans = list()
        for passage in self:
            passages_spans.append(passage.span)
        return passages_spans
    
    def passages_entities(self, sort=False, make_deepcopy=False):
        passages_entities = list()
        for passage in self:
            entities = passage.entities(sort=sort, make_deepcopy=make_deepcopy)
            passages_entities.append(entities)
        return passages_entities
    
    def nes(self):
        nes = NormalizedEntitySet()
        for p in self.passages:
            for e in p.nes:
                nes.add(e)
        return nes
    
    def iis(self):
        iis = IndexingIdentifierSet()
        for p in self.passages:
            for ii in p.iis:
                iis.add(ii)
        return iis
    
    def entities(self, sort=False, make_deepcopy=False):
        return self.nes().get(sort=sort, make_deepcopy=make_deepcopy)
    
    def indexing_identifiers(self, sort=False, make_deepcopy=False):
        return self.iis().get(sort=sort, make_deepcopy=make_deepcopy)
    
    def get_entity_set(self):
        return self.nes().to_entity_set()
    
    def json(self):
        d = list()
        nes_i = 1
        iis_i = 1
        for p in self:
            d.append(p.json(nes_i, iis_i))
            nes_i += len(p.nes)
            iis_i += len(p.iis)
        return d
    
    def pretty_json(self):
        return json.dumps(self.json(), indent=4, sort_keys=True)


class Document:
    r"""
    A Document contains an identifier and a PassageOrderedList object.
    """
    
    def __init__(self, identifier):
        assert isinstance(identifier, str)
        self.identifier = identifier
        self.pol = PassageOrderedList()
        self.n_passages = 0
    
    def __iter__(self):
        for p in self.pol:
            yield p
    
    def __str__(self):
        s = 'Document {} with '.format(repr(self.identifier))
        if self.n_passages == 0:
            s += 'no passages.'
        elif self.n_passages == 1:
            s += '1 passage.'
        else:
            s += '{} passages.'.format(self.n_passages)
        return s
    
    def add_passage(self, p):
        self.pol.add(p)
        self.n_passages += 1
    
    def add_passages(self, passages):
        for p in passages:
            self.add_passage(p)
    
    def span(self):
        return self.pol.span()
    
    def text(self):
        return self.pol.text()
    
    def passages_texts(self):
        return self.pol.passages_texts()
    
    def passages_spans(self):
        return self.pol.passages_spans()
    
    def passages_entities(self, sort=False, make_deepcopy=False):
        return self.pol.passages_entities(sort=sort, make_deepcopy=make_deepcopy)
    
    def nes(self):
        return self.pol.nes()
    
    def iis(self):
        return self.pol.iis()
    
    def entities(self, sort=False, make_deepcopy=False):
        return self.pol.entities(sort=sort, make_deepcopy=make_deepcopy)
    
    def indexing_identifiers(self, sort=False, make_deepcopy=False):
        return self.pol.indexing_identifiers(sort=sort, make_deepcopy=make_deepcopy)
    
    def get_entity_set(self):
        return self.pol.get_entity_set()
    
    def clear_entities(self):
        r"""
        Remove all the entities from the document.
        """
        for p in self:
            p.nes = NormalizedEntitySet()
    
    def set_entities(self, entities):
        #
        # This step is just done to verify that the input is valid.
        #
        entities = NormalizedEntitySet(entities).get()
        
        for p in self:
            #
            # First, delete existent entities.
            #
            p.nes = NormalizedEntitySet()
            #
            # Then, add only the entities that are within the respective
            # span.
            #
            span_entities = get_entities_within_span(entities, p.span)
            p.add_entities(span_entities)
    
    def set_indexing_identifiers(self, indexing_identifiers):
        err = 'You cannot set indexing identifiers for a document without passages.'
        assert self.n_passages > 0, err
        #
        # First, delete existent indexing identifiers from all passages.
        #
        for p in self:
            p.iis = IndexingIdentifierSet()
        #
        # Then, add the new indexing identifiers only in the first
        # passage.
        #
        for p in self:
            break
        p.iis = IndexingIdentifierSet(indexing_identifiers)
    
    def set_mesh_indexing_identifiers(self, mesh_indexing_identifiers):
        self.set_indexing_identifiers([
            IndexingIdentifier(mii, typ='MeSH_Indexing_Chemical')
            for mii in mesh_indexing_identifiers
        ])
    
    def json(self):
        return {
            'id': self.identifier,
            'passages': self.pol.json()
        }
    
    def pretty_json(self):
        return json.dumps(self.json(), indent=4, sort_keys=True)


def sort_identifiers(identifiers):
    #
    # The identifiers are for example a list of PMCIDs.
    #
    assert isinstance(identifiers, list) or isinstance(identifiers, dict)
    for i in identifiers:
        assert isinstance(i, str)
    
    return sorted(sorted(identifiers), key=len)


class Collection:
    r"""
    A set of documents, using the above defined Document class.
    Each document is associated with a specific identifier.
    The identifier has to be a string.
    For example, a PMCID (PubMed Central ID) is a valid identifier.
    """
    
    def __init__(self):
        self.i2d = dict()
        self.n_documents = 0
    
    def __len__(self):
        return self.n_documents
    
    def __getitem__(self, key):
        return self.i2d[key]
    
    def __iter__(self):
        for i in self.ids():
            yield (i, self[i])
    
    def __str__(self):
        return self.corpus + '_' + self.group
    
    def add_metadata(self, corpus, group):
        r"""
        Allow the collection to know the corpus where it belongs.
        """
        self.corpus = corpus
        self.group = group
        #
        # This is needed to chain generator names.
        #
        self.__name__ = self.corpus + '_' + self.group
    
    def add(self, i, d, make_deepcopy=False):
        r"""
        Add document with the respective identifier.
        """
        assert isinstance(i, str)
        assert isinstance(d, Document)
        assert i not in self.i2d
        if make_deepcopy:
            d = deepcopy(d)
        self.i2d[i] = d
        self.n_documents += 1
    
    def get(self, i):
        return self.i2d[i]
    
    def ids(self):
        r"""
        Return a sorted list with the identifiers.
        """
        return sort_identifiers(self.i2d)
    
    def clear_entities(self):
        r"""
        Remove the entities from all the documents. This is useful in
        the inference phase. It serves as a sanity measure.
        """
        for i, d in self:
            d.clear_entities()
    
    def json(self):
        return {
            'documents': [d.json() for _, d in self]
        }
    
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


def get_non_overlapping_and_overlapping_entities(entities):
    #
    # This is a lazy function to get the (i) non overlapping and
    # (ii) overlapping entities.
    #
    non_overlapping_entities = list()
    overlapping_entities = list()
    n = len(entities)
    for i in range(n):
        entity = entities[i]
        others = entities[:i] + entities[i+1:]
        overlap = False
        for other in others:
            if span_overlaps_span(entity.span, other.span):
                overlap = True
                break
        if overlap:
            overlapping_entities.append(entity)
        else:
            non_overlapping_entities.append(entity)
    
    return non_overlapping_entities, overlapping_entities


def get_entities_within_span(entities, span):
    #
    # This is a lazy function to get only the entities that are within
    # a given span.
    # Entities that are outside the given span are filtered out.
    #
    filtered_entities = list()
    
    for e in entities:
        if span_contains_span(span, e.span):
            filtered_entities.append(e)
    
    return filtered_entities