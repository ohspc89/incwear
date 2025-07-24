"""
Written by Jinseok Oh, Ph.D.
2022/9/13 - present (as of 2025/7/16)

Calibration related functions

© 2023-2025 Infant Neuromotor Control Laboratory. All rights reserved.
"""
from itertools import repeat
import numpy as np


def get_axis_bias(ground_gs):
    """
    A function to calculate axis/orientation specific biases

    Parameters
    ----------
    grounds_gs : list
        measured gravitational accelerations in the following order/
        x-, x+, y-, y+, z-, z+

    Returns
    -------
    x : list
        list of axis/orientation specific offsets
    """
    # If it is a list of TWO lists...
    refvec = np.array([-1, 1, -1, 1, -1, 1])
    if len(ground_gs) == 2:
        return list(map(lambda x: refvec - x, ground_gs))
    return refvec - ground_gs


def get_axis_bias_v2(ground_gs):
    """ ground_gs are offset_removed gs """

    def calc_error(arr, idxlist):
        """ Return: a list of three gain values """
        # idx as a list...
        errs = []
        for idx in idxlist:
            gain = np.diff(arr[idx[0]:idx[1]])[0]/2
            errs.append(gain)
        return errs

    if len(ground_gs) == 2:
        # Left and Right
        lxyz, rxyz = map(calc_error,
                         list(ground_gs.values()),
                         repeat([[0, 2], [2, 4], [4, 6]], 2))
        return np.array((lxyz, rxyz))

    lxyz = calc_error(ground_gs, [[0, 2], [2, 4], [4, 6]])
    return np.array(lxyz)
