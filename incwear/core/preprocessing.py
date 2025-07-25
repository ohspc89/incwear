"""
Written by Jinseok Oh, Ph.D.
2022/9/13 - present (as of 2025/7/16)

Preprocessing related functions

© 2023-2025 Infant Neuromotor Control Laboratory. All rights reserved.
"""
from fractions import Fraction
import numpy as np
from scipy.signal import firwin, lfilter, resample_poly


def get_mag(arr, row_idx=None, det_option='median'):
    """
    Calculate the norm (magnitude) with optional detrending.

    Parameters
    ----------
    arr : numpy.ndarray
        Nx3 array (acc/gyro)
    row_idx : list | None
        Indices to extract relevant range
    det_option : str
         'median' (default) or 'customfunc' (subtract 1g)

    Returns
    -------
    out : numpy.ndarray
         Detrended acceleration or angular velocity norm
    """
    if det_option not in ['median', 'customfunc']:
        det_option = 'median'
        print("Unknown detrending option - setting it to [median]")

    # Axivity sensors differ in length.
    # Let's create another function and use map()
    # 7/28/23, Sanity check - print median
    def linalg_norm(arr, row_idx):
        nrow = arr.shape[0]

        if row_idx is None:
            row_idx = list(range(nrow))  # targeting the entire dataset
        mag = np.linalg.norm(arr[row_idx], axis=1)
        print('median: ', np.median(mag))
        return mag

    # Use np.linalg.norm... amazing!
    mag = linalg_norm(arr, row_idx)

    # MATLAB's detrend function is not used,
    #   so we can consider that the default option
    #   for detrending the data is subtracting the median
    if det_option == 'median':
        return mag - np.median(mag)
    else:
        # detrend by subtracting 1g
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


def resample_to(arr, orig_fs, re_fs):
    """
    Resample signal from orig_fs to re_fs using polyphase filtering.

    Parameters
    ----------
    arr : numpy.ndarray
        Input signal array (1D or 2D).
    orig_fs : float
        Original sampling frequency.
    re_fs : float
        Target sampling frequency.

    Returns
    -------
    tuple of (numpy.ndarray, numpy.ndarray)
        (new_time_array, resampled_data)
    """
    ratio = re_fs / orig_fs
    frac = Fraction(ratio).limit_denominator(100)

    resampled = resample_poly(arr, up=frac.numerator,
                              down=frac.denominator, axis=0)

    duration = arr.shape[0] / orig_fs
    new_len = resampled.shape[0]
    new_time = np.linspace(0, duration, new_len, endpoint=False)

    return new_time, resampled
