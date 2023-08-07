#!/usr/bin/env python3
"""Task 0"""
import numpy as np


def uni_bleu(references, sentence):
    """
    calculates the unigram BLEU score for a sentence:

    references is a list of reference translations
        each reference translation is a list of the words in the translation
    sentence is a list containing the model proposed sentence
    Returns: the unigram BLEU score
    """
    c = len(sentence)
    r = min([len(reference) for reference in references])
    BP = 1 if c > r else np.exp(1 - (r / c))

    max_ref_count = {word: 0 for word in sentence}
    for reference in references:
        for word in sentence:
            max_ref_count[word] = max(reference.count(word),
                                      max_ref_count[word])

    Pn = np.sum(list(max_ref_count.values())) / c

    bleu = BP * Pn

    return bleu
