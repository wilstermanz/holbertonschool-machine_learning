#!/usr/bin/env python3
"""Task 1 - Regular chains"""
import numpy as np


def markov_chain(P, s, t=1):
    """
    determines the probability of a markov chain being in a particular state
    after a specified number of iterations:

    P is a square 2D numpy.ndarray of shape (n, n) representing the transition
    matrix
        P[i, j] is the probability of transitioning from state i to state j
        n is the number of states in the markov chain
    s is a numpy.ndarray of shape (1, n) representing the probability of
    starting in each state
    t is the number of iterations that the markov chain has been through
    Returns: a numpy.ndarray of shape (1, n) representing the probability of
    being in a specific state after t iterations, or None on failure
    """
    for _ in range(t):
        s = s @ P
    return s


def regular(P):
    """
    determines the steady state probabilities of a regular markov chain:

    P is a is a square 2D numpy.ndarray of shape (n, n) representing the
    transition matrix
        P[i, j] is the probability of transitioning from state i to state j
        n is the number of states in the markov chain
    Returns: a numpy.ndarray of shape (1, n) containing the steady state
    probabilities, or None on failure
    """
    if P.shape[0] != P.shape[1]:
        return None
    if np.any(P == 0) or np.any(P > 1) or np.any(P < 0):
        return None

    s = np.ones((1, P.shape[0])) / P.shape[0]
    return markov_chain(P, s, 1000)
