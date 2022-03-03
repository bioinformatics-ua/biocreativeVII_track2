#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from elements import Entity
from elements import EntitySet
import copy


def longer_entities_first(entities):
    #
    # Sort entities according to their length.
    # Longer entities come first (have higher priority).
    #
    return sorted(entities, key=len, reverse=True)


def entity_fits_spans(entity, spans):
    #
    # An Entity fits a list of spans only if it corresponds to an exact
    # match (exact left and right boundaries).
    # If true it returns the indexes of the spans, otherwise it returns
    # False.
    #
    start_found = False
    end_found = False
    indexes = list()
    
    for i, span in enumerate(spans):
        
        if not start_found:
            if entity.start == span[0]:
                start_found = True
            elif entity.start < span[0]:
                break
        
        if start_found and (not end_found):
            indexes.append(i)
            if entity.end == span[1]:
                end_found = True
                break
            elif entity.end < span[1]:
                break
    
    if start_found and end_found:
        return indexes
    
    return False


def update_tags(tags, spans, entity):
    #
    # Only annotate entities (with BIO tags) that correspond to exact
    # matches, since we are only interested in evaluating exact matches.
    # If an entity does not exactly match within the given spans, that
    # is, if their boundaries do not fit any combination of consecutive
    # spans, then it is discarded and it is not annotated using the BIO
    # tags.
    #
    # All the corresponding tags of the input entity should not have
    # been attributed to another entity. They should be free to use,
    # that is, they all should have the 'O' (other) label.
    # Otherwise, if any corresponding tag of the input entity has
    # already an annotation from another entity, the given (input)
    # entity is discarded and it is not represented. In this case,
    # the tags are not updated.
    #
    # Beginning and inside tag strings. The type of the entity is
    # redundant (in this case, is always a "Chemical"). However, for the
    # sake of readability is always present.
    #
    begin = 'B-{}'
    insid = 'I-{}'
    
    entity_indexes = entity_fits_spans(entity, spans)
    if entity_indexes:
        #
        # Check if all the respective tags are available
        # ('O' other label).
        #
        available = True
        for i in entity_indexes:
            if tags[i] != 'O':
                available = False
                break
        #
        # If all the tags are available, then change the tags to add the
        # annotation.
        #
        if available:
            first = entity_indexes[0]
            tags[first] = begin.format(entity.typ)
            for i in entity_indexes[1:]:
                tags[i] = insid.format(entity.typ)


def get_bio(spans, entities):
    #
    # Get the BIO tags for all tokens.
    #
    # Before starting the BIO annotation, first sort the entities
    # according to their length. Entities with longer spans are
    # considered to have more priority. (This is only relevant for the
    # case of overlapping entities.)
    # This choice also seems to be in more accordance with the
    # indications from the official NLM-Chem annotation guidelines:
    #   """
    #   If multiple chemicals are listed as an overlapping ellipsis (or
    #   span), annotate the full phrase as a single mention, and assign
    #   MeSH IDs to each chemical in the elliptical string (or span) in
    #   order of appearance.
    #   """
    #
    tags = ['O' for span in spans]
    
    for e in longer_entities_first(entities):
        update_tags(tags, spans, e)
    
    return tags


def decode_bio(tags, spans, text, passage_numbers=None,
    allow_errors=False):
    #
    # Return the entities (EntitySet) by decoding the BIO tags.
    #
    es = EntitySet()
    #
    # Count the total number of tags, and the number of decoding errors.
    # There are four causes of errors:
    # (1) An inside tag (I) at the start of the sequence (first token).
    # (2) An Inside tag (I) after an Other tag (O).
    # (3) An Inside tag (I) with a different entity type from the last
    #     one. Example: I-Gene follows B-Chemical.
    # (4) An Inside tag (I) at the start (first token) of a new passage,
    #     independent from the tag of the previous token.
    #
    counts = {
        'tags': len(tags),
        'inside_tag_at_start': 0,
        'inside_tag_after_other_tag': 0,
        'inside_tag_with_different_entity_type': 0,
        'inside_tag_at_passage_start': 0,
    }
    
    #
    # If there are no passage numbers, assume that all the tokens are
    # from the same passage (number 0).
    #
    if passage_numbers is None:
        passage_numbers = [0] * len(tags)
    
    first_token = True
    previous_tag_is_entity = False
    previous_number = passage_numbers[0]
    
    for tag, span, number in zip(tags, spans, passage_numbers):
        
        start, end = span
        
        if tag == 'O':
            if previous_tag_is_entity:
                ent = Entity(text[s:e], (s, e), t)
                es.add(ent)
            previous_tag_is_entity = False
        
        elif tag[0] == 'B':
            if previous_tag_is_entity:
                ent = Entity(text[s:e], (s, e), t)
                es.add(ent)
            s = start
            e = end
            t = tag.split('-')[1]
            previous_tag_is_entity = True
        
        elif tag[0] == 'I':
            t_new = tag.split('-')[1]
            
            if allow_errors:
                if not previous_tag_is_entity:
                    s = start
                    t = t_new
                    previous_tag_is_entity = True
                    if first_token:
                        counts['inside_tag_at_start'] += 1
                    else:
                        counts['inside_tag_after_other_tag'] += 1
                    if previous_number != number:
                        counts['inside_tag_at_passage_start'] += 1
                elif t_new != t:
                    ent = Entity(text[s:e], (s, e), t)
                    es.add(ent)
                    s = start
                    t = t_new
                    counts['inside_tag_with_different_entity_type'] += 1
                    if previous_number != number:
                        counts['inside_tag_at_passage_start'] += 1
                elif previous_number != number:
                    ent = Entity(text[s:e], (s, e), t)
                    es.add(ent)
                    s = start
                    t = t_new
                    counts['inside_tag_at_passage_start'] += 1
            else:
                err = 'Previous token does not belong to an Entity.'
                assert previous_tag_is_entity, err
                err = 'Previous Entity type is different.'
                assert t_new == t, err
                err = 'Previous passage number is different.'
                assert previous_number == number, err
            
            e = end
        
        previous_number = number
        first_token = False
    #
    # Add remaining entity if last tag is 'B' (beginning) or
    # 'I' (inside).
    #
    if previous_tag_is_entity:
        ent = Entity(text[s:e], (s, e), t)
        es.add(ent)
    
    return es, counts


def reconstruct_bio(tags, spans, tokens, strategy_threshold=0):
    """
    Reconstruct the tags based on token level, instead of subtoken level
    """
    tags = copy.deepcopy(tags)
    # holds the current reconstruction state
    def clear_state():
        return  {
            "tags": [],
            "n_bi": 0,
            "start_index" : -1
        }
    
    def add_to_state(state, index):
        state["tags"].append(tags[index])
        state["n_bi"] += 0 if tags[index]=="O" else 1
        if state["start_index"] == -1:
            state["start_index"] = index
            
    def make_token_reconstruction(state):
        
        b_i_ratio = state["n_bi"]/len(state["tags"])
        
        if b_i_ratio>strategy_threshold:
            # consider as an entity
            tags[state["start_index"]] = "B-Chemical"
            for i in range(len(state["tags"][1:])):
                tags[state["start_index"]+i+1] = "I-Chemical"
        else:
            # consider as other
            for i in range(len(state["tags"])):
                tags[state["start_index"]+i] = "O"

    state = clear_state()
    for i, token in enumerate(tokens):
        if token.startswith("##"):
            add_to_state(state, i-1)
        elif len(state["tags"])>0:
            add_to_state(state, i-1)
            make_token_reconstruction(state)
            state = clear_state()
            
    return tags
