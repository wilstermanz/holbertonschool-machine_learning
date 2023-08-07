#!/usr/bin/env python3
"""Task 2"""
import numpy as np


def cumulative_bleu(references, sentence, N):
    """
    calculates the cumulative n-gram BLEU score for a sentence:

        references is a list of reference translations
            each reference translation is a list of the words in the
            translation
        sentence is a list containing the model proposed sentence
        n is the size of the largest n-gram to use for evaluation
        All n-gram scores should be weighted evenly
        Returns: the cumulative n-gram BLEU score
    """
    def make_ngrams(sentence, n):
        """converts a sentence to ngrams"""
        ngrams = []
        for i in range(0, len(sentence) - n + 1):
            grams = [sentence[i + j] for j in range(n)]
            ngram = ''
            for j, word in enumerate(grams):
                ngram += word
                if j != n - 1:
                    ngram += ' '
            ngrams.append(ngram)

        return ngrams

    def count_ngrams(sentence, ngrams, n):
        ngram_sentence = make_ngrams(sentence, n)
        ngram_count = {ngram: 0 for ngram in ngrams}
        for ngram in ngram_sentence:
            if ngram in ngrams:
                ngram_count[ngram] += 1

        return ngram_count

    Pn = []
    for n in range(1, N + 1):
        ngrams = make_ngrams(sentence, n)

        c = len(sentence)
        r = min([len(reference) for reference in references])
        BP = 1 if c > r else np.exp(1 - (r / c))

        max_ref_count = {ngram: 0 for ngram in ngrams}
        for reference in references:
            ngram_count = count_ngrams(reference, ngrams, n)
            for ngram in ngrams:
                max_ref_count[ngram] = max(ngram_count[ngram],
                                           max_ref_count[ngram])

        Pn.append(np.sum(list(max_ref_count.values())) / np.sum(
            list(count_ngrams(sentence, ngrams, n).values())))
        for value in Pn:
            if value == 0:
                Pn.remove(value)

    bleu = BP * np.exp(np.sum((1/N) * np.log(Pn)))

    return bleu
