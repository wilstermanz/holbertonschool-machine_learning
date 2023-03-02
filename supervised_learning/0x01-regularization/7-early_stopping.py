#!/usr/bin/env python3
"""Task 7"""


def early_stopping(cost, opt_cost, threshold, patience, count):
    """
    determines if you should stop gradient descent early:

        * Early stopping should occur when the validation cost of the network
          has not decreased relative to the optimal validation cost by more
          than the threshold over a specific patience count
        * cost is the current validation cost of the neural network
        * opt_cost is the lowest recorded validation cost of the neural network
        * threshold is the threshold used for early stopping
        * patience is the patience count used for early stopping
        * count is the count of how long the threshold has not been met
        * Returns: a boolean of whether the network should be stopped early,
          followed by the updated count
    """
    count += 1
    stop = False

    # Check for count outside of patience
    if count >= patience:
        stop = True

    # Check for cost within threshold
    if cost < opt_cost - threshold:
        count = 0

    return stop, count
