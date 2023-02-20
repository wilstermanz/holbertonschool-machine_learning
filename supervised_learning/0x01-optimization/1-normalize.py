#!/usr/bin/env python3
"""Task 1"""


def normalize(X, m, s):
    """normalizes (standardizes) a matrix"""
    return (X - m) / (s**2 + 10**(-8))**(1/2)
