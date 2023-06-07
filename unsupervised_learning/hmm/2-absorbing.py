#!/usr/bin/env python3
"""Task 2 - Absorbing Chains"""
import numpy as np


def absorbing(P):
    """
    determines if a markov chain is absorbing:

    P is a is a square 2D numpy.ndarray of shape (n, n) representing the
    standard transition matrix
        P[i, j] is the probability of transitioning from state i to state j
        n is the number of states in the markov chain
    Returns: True if it is absorbing, or False on failure
    """
    # Check for any absorbing states
    if not np.any(np.diagonal(P) == 1):
        return False

    # Check for all absorbing states
    if np.array_equal(P, np.identity(P.shape[0])):
        return True

    # Find the solution matrix
    n = len(np.where(np.diagonal(P) == 1)[0])
    B = P[n + 1:, n + 1:]
    In = np.identity(len(B))
    F = np.lingalg.inv(In - P[n + 1:, n + 1:])
    A = P[n + 1:, :n + 1]
    FA = F @ A

    if np.any(FA != 0):
        return True
    return False
