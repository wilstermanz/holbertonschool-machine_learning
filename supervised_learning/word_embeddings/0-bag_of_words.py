#!/usr/bin/env python3
"""task 0"""
import nltk
from nltk.tokenize import word_tokenize as tokenize
from sklearn.feature_extraction.text import CountVectorizer

nltk.download('punkt', quiet=True)

def bag_of_words(sentences, vocab=None):
    """
    creates a bag of words embedding matrix:

    sentences is a list of sentences to analyze
    vocab is a list of the vocabulary words to use for the analysis
        If None, all words within sentences should be used
    Returns: embeddings, features
        embeddings is a numpy.ndarray of shape (s, f) containing the embeddings
            s is the number of sentences in sentences
            f is the number of features analyzed
        features is a list of the features used for embeddings
    """
    corpus = []

    for sentence in sentences:
        tokens = tokenize(sentence)
        words = ""
        for token in tokens:
            if token.isalpha():
                words += token.lower() + ' '
        corpus.append(words)
    
    if not vocab:
        vocab = []
        for sentence in corpus:
            for token in tokenize(sentence):
                vocab.append(token)
        vocab = list(set(vocab))
        vocab.sort()
    
    vectorizer = CountVectorizer(vocabulary=vocab)
    embeddings = vectorizer.fit_transform(corpus).toarray()

    return embeddings, vocab
