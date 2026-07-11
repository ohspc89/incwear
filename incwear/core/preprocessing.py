"""
Written by Jinseok Oh, Ph.D.
2022/9/13 - present (as of 2026/4/16)

Preprocessing related functions

© 2023-2026 Infant Neuromotor Control Laboratory. All rights reserved.
"""
from fractions import Fraction
import logging

import numpy as np
from scipy.signal import firwin, lfilter, resample_poly

logger = logging.getLogger(__name__)


def get_mag(arr, row_idx=None, det_option='median'):
    """
    Calculate the norm (magnitude) with optional detrending.

    Parameters
    ----------
    arr : numpy.ndarray
        Nx3 array (acc/gyro)
    row_idx : list | numpy.ndarray | slice | None
        Indices to extract relevant range
    det_option : str
         'median' (default) or 'customfunc' (subtract 1g)

    Returns
    -------
    numpy.ndarray
         Detrended acceleration or angular velocity norm
    """
    if det_option not in ['median', 'customfunc']:
        logger.warning(
            "Unknown detrending option %r; defaulting to 'median'.",
            det_option,
        )
        det_option = 'median'

    selected = arr if row_idx is None else arr[row_idx]
    mag = np.linalg.norm(selected, axis=1)
    logger.debug("Computed magnitude series with median %.6f", float(np.median(mag)))

    # MATLAB's detrend function is not used,
    #   so we can consider that the default option
    #   for detrending the data is subtracting the median
    if det_option == 'median':
        return mag - np.median(mag)
    
    return mag - 9.80665


def correct_gain(arr, gains):
    """
    Adjust sensor data by axis-specific gain factors.

    Parameters
    ----------
    arr : numpy.ndarray
        Nx3 array of raw sensor values.
    gains : list or numpy.array
        Gain values for x, y, z axes.

    Returns
    -------
    numpy.ndarray
        Gain-corrected sensor data.
    """
    return arr / np.array(gains)


def low_pass(fc, fs, arr, window='hamming'):
    """
    Apply a simple 1st-order FIR low-pass filter to the input signal.

    Parameters
    ----------
    fc : float
        Cut-off frequency in Hz.
    fs : float
        Sampling frequency in Hz.
    arr : numpy.ndarray
        Input signal to filter.
    window : str
        FIR window type (default is 'hamming'). See scipy.signal.get_window
        for a list of windows

    Returns
    -------
    numpy.ndarray
        Filtered signal.
    """
    h = firwin(2, cutoff=fc, window=window, fs=fs)
    return lfilter(h, 1, arr)


def interpolate_to_uniform_time(arr, ts, target_fs):
    ts = np.asarray(ts)
    arr = np.asarray(arr)

    if ts.ndim != 1:
        raise ValueError("ts must be 1D")
    if arr.ndim != 1:
        raise ValueError("arr must be 1D for now")
    if len(ts) != len(arr):
        raise ValueError("ts and arr must have the same length")

    start = 0.0
    stop = ts[-1]

    new_time = np.arange(start, stop + 1e-12, 1.0 / target_fs)
    interp_arr = np.interp(new_time, ts, arr)
    return new_time, interp_arr


def resample_to(arr, orig_fs, re_fs, ts,
                max_denominator=100):
    """
    Resample signal from orig_fs to re_fs using polyphase filtering.

    Parameters
    ----------
    arr : numpy.ndarray
        Input 1D signal array.
    orig_fs : float
        Original sampling frequency.
    re_fs : float
        Target sampling frequency.
    ts : numpy.ndarray
        A series of intervals from the start (0)
    max_denominator : int, optional
        Maximum denominator for rational resampling ratio.

    Returns
    -------
    tuple of (numpy.ndarray, numpy.ndarray)
        (new_time_array, resampled_data)
    """
    # --- interpolation should be done ---
    # Fix the option to linear... cubic may be too much
    uniform_time, uniform_arr = interpolate_to_uniform_time(arr, ts, orig_fs)

    # If you're just 'correcting the interval', why bother resample?
    logger.info("Interpolation completed")
    if re_fs == orig_fs:
        return uniform_time, uniform_arr

    frac = Fraction(re_fs / orig_fs).limit_denominator(max_denominator)

    # --- Resampling (polyphase FIR) ---
    resampled = resample_poly(
        uniform_arr,
        up=frac.numerator,
        down=frac.denominator,
        axis=0)

    # --- Trim ---
    trim = int(np.ceil(5 * max(frac.numerator, frac.denominator)))

    if 2 * trim >= resampled.shape[0]:
        logger.warning(
            "Trim legnth %d too large for resampled array of length %d; skipping trim.",
            trim, resampled.shape[0]
        )
    else:
        resampled = resampled[trim:-trim]

    # --- Correct time axis ---
    new_time = np.arange(resampled.shape[0]) / re_fs

    return new_time, resampled
