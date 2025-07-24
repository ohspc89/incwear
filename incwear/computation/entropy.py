"""
Written by Jinseok Oh, Ph.D.
2022/9/13 - present (as of 2025/7/16)

Entropy related functions

© 2023-2025 Infant Neuromotor Control Laboratory. All rights reserved.
"""
import numpy as np
from EntropyHub import SampEn, FuzzEn


def calc_entropy(arr, fuzzy=False, m=2, r=None):
    """
    A function to calculate either Sample Entropy (SampEn)
    or Fuzzy Entropy (FuzzEn) of a given array.
    This uses basic features of EntropyHub module's SampEn and
    FuzzEn functions. Regarding FuzzEn, if you're interested in
    exploring more, using EntropyHub.FuzzEn directly is recommended.
    For more detail, please check
    github.com/MattWillFlood/EntropyHub/blob/main/EntropyHub%20Guide.pdf

    Parameters
    ----------
    arr : list or numpy.ndarray
        Time series whose entropy value will be calculated.
    fuzzy : bool
        if True, then calculate FuzzEn; default is False
    m : int
        Size of the embedding dimension; default is 2
    r : float | tuple | None
        Radius of the neighbourhood (default=None)
        This will make use of the default options of SampEn and FuzzEn
        from EntropyHub (r=0.2*SD(arr) for SampEn, r=(0.2, 2) for FuzzEn).

    Returns
    -------
    ent : float
        value of SampEn or FuzzEn
    """
    if not fuzzy:
        ent = SampEn(arr, m=m, r=r)
    else:
        if r is None:
            ent = FuzzEn(arr, m=m)
        else:
            ent = FuzzEn(arr, m=m, r=r)

    return ent
