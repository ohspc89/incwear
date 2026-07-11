"""
Written by Jinseok Oh, Ph.D.
2022/9/13 - present (as of 2026/4/16)

Threshold-related utilities for movement detection.

© 2023-2026 Infant Neuromotor Control Laboratory. All rights reserved.
"""
from __future__ import annotations

import logging
from typing import Callable

import numpy as np
from scipy.signal import find_peaks

from incwear.utils.data_models import Thresholds
from incwear.errors import ThresholdComputationError

logger = logging.getLogger(__name__)

DEFAULT_REJECT = 3.2501
DEFAULT_HEIGHT = 1.0


def _sample_std(values: np.ndarray) -> float:
    """Sample standard deviation (ddof=1), matching the article's SD.

    A single peak has no dispersion to estimate, so return 0.0 instead of
    the NaN that ``np.std(..., ddof=1)`` yields for n < 2.
    """
    values = np.asarray(values, dtype=float)
    if values.size < 2:
        return 0.0
    return float(np.std(values, ddof=1))


def _validate_accmags(accmags: np.ndarray) -> np.ndarray:
    arr = np.asarray(accmags, dtype=float)
    if arr.ndim != 1:
        raise ValueError("accmags must be a 1D array")
    if arr.size == 0:
        raise ValueError("accmags must not be empty")
    if not np.isfinite(arr).all():
        raise ValueError("accmags contains NaN or infinite values")
    return arr


def _safe_peak_threshold(peak_heights: np.ndarray, *, label: str) -> float:
    values = np.asarray(peak_heights, dtype=float)
    if values.size == 0:
        raise ThresholdComputationError(
            f"No valid {label} peaks were found for threshold computation."
        )
    return float(np.mean(values) - _sample_std(values))


def calc_ind_threshold(maxprop: dict) -> float:
    """
    Calculate individual threshold based on peak heights:
        mean(peak_heights) - C * std(peak_heights)

    C = 1       Smith et al. (2015)
        0.5     Trujillo-Priego et al. (2017), ivan=True

    Parameters
    ----------
    maxprop : dict
        Output of `find_peaks` function.

    Returns
    -------
    float
        Threshold value.
    """
    return _safe_peak_threshold(maxprop.get("peak_heights", []), label="signal")


# This is equivalent to the MATLAB script
# calc_thresh_fn is expected to consume the properties
# returned by scipy.find_peaks
# and return a scalar threshold.
def get_ind_acc_threshold(
    accmags: np.ndarray,
    calc_thresh_fn: Callable[[dict], float] = calc_ind_threshold,
    reject: float = DEFAULT_REJECT,
    height: float = DEFAULT_HEIGHT
) -> Thresholds:
    """
    Compute individual thresholds from acceleration magnitudes.

    Parameters
    ----------
    accmags : numpy.ndarray
        Detrended acceleration magnitudes.
    calc_thresh_fn : function
        Function to compute individual threshold from peaks.
    reject : float
        Peak value rejection threshold.
    height : float
        Minimum peak height

    Returns
    -------
    Thresholds.
    """
    mags = _validate_accmags(accmags)

    posvals = np.where((mags > 0) & (mags < reject), mags, 0.0)
    negvals = np.where((mags < 0) & (mags > -reject), np.abs(mags), 0.0)

    _, pos_props = find_peaks(posvals, height=height)
    _, neg_props = find_peaks(negvals, height=height)

    pos_th = float(calc_thresh_fn(pos_props))
    neg_th = float(calc_thresh_fn(neg_props))

    return Thresholds(
            accth=pos_th,
            naccth=-neg_th,
            method="get_ind_acc_threshold",
            reject=reject,
            height=height
    )


def get_ind_acc_threshold2(
        accmags: np.ndarray,
        reject: float = DEFAULT_REJECT,
        height: float = DEFAULT_HEIGHT
) -> Thresholds:
    """
    Alternate threshold calculation: a more correct form of Smith et al. (2015)

    Parameters
    ----------
    accmags : numpy.ndarray
        Detrended acceleration magnitudes.

    Returns
    -------
    Thresholds.
    """
    mags = _validate_accmags(accmags)

    rectified_negative = np.abs(np.where(mags < 0, mags, 0.0))

    pos_locs, pos_props = find_peaks(mags, height=height)
    neg_locs, neg_props = find_peaks(rectified_negative, height=height)

    valid_pos = pos_locs[pos_props["peak_heights"] < reject]
    valid_neg = neg_locs[neg_props["peak_heights"] < reject]

    if valid_pos.size == 0 or valid_neg.size == 0:
        raise ThresholdComputationError(
            "Could not compute thresholds because one or both peak sets are empty "
            f"after applying reject={reject} and height={height}."
        )

    pos_vals = mags[valid_pos]
    neg_vals = rectified_negative[valid_neg]

    return Thresholds(
            accth=float(np.mean(pos_vals) - _sample_std(pos_vals)),
            naccth=-float(np.mean(neg_vals) - _sample_std(neg_vals)),
            method="get_ind_acc_threshold2",
            reject=reject,
            height=height,
    )
