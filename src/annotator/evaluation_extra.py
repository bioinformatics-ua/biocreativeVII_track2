#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from math import nan

# from elements import Entity
from elements import EntitySet


def precision_recall_f1(tp, fp, fn, return_nan=True):
    
    try:
        precision = tp / (tp + fp)
    except ZeroDivisionError:
        precision = nan if return_nan else 0.0
    
    try:
        recall = tp / (tp + fn)
    except ZeroDivisionError:
        recall = nan if return_nan else 0.0
    
    try:
        f1 = tp / (tp + 0.5 * (fp + fn))
    except ZeroDivisionError:
        f1 = nan if return_nan else 0.0
    
    return precision, recall, f1


def empty_results(counts=True):
    if counts:
        return {
            'tp': 0,
            'fp': 0,
            'fn': 0,
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
        }
    else:
        return {
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
        }


def eval_list_of_entity_sets(true, pred, return_nan=True):
    #
    # Only one evaluation mode is defined and calculated:
    # (1) Strict:
    #     For a predicted entity to count as True Positive (TP), its
    #     string, span (left and right boundaries), and type must be
    #     correct. That is, it must match (exactly) the gold standard
    #     (true) entity.
    #
    assert isinstance(true, list)
    assert isinstance(pred, list)
    assert len(true) == len(pred)
    #
    # Initialize dictionary with empty results.
    #
    results = empty_results(counts=True)
    #
    # Go through each pair of entity sets (true, predicted).
    #
    for t_es, p_es in zip(true, pred):
        assert isinstance(t_es, EntitySet)
        assert isinstance(p_es, EntitySet)
        #
        # Total number of true and predicted entities.
        #
        n_true_entities = len(t_es)
        n_pred_entities = len(p_es)
        
        tp = len(t_es.intersection(p_es))
        
        fp = n_pred_entities - tp
        fn = n_true_entities - tp
        
        results['tp'] += tp
        results['fp'] += fp
        results['fn'] += fn
    
    results['precision'], results['recall'], results['f1'] = precision_recall_f1(results['tp'], results['fp'], results['fn'], return_nan=return_nan)
    
    return results