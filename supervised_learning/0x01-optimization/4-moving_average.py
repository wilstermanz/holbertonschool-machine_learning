#!/usr/bin/env python3
"""Task 4"""


def moving_average(data, beta):
    """calculates the weighted moving average of a data set"""
    weighted_average = 0
    list_of_averages = []
    for i in range(len(data)):
        weighted_average = (beta * weighted_average) + (1 - beta) * data[i]
        bias_correction = 1 - (beta**(i + 1))
        list_of_averages.append(weighted_average / bias_correction)
    return list_of_averages
