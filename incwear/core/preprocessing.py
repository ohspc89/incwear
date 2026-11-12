"""
Written by Jinseok Oh, Ph.D.
2022/9/13 - present (as of 2026/11/11)

Preprocessing related functions

© 2023-2026 Infant Neuromotor Control Laboratory. All rights reserved.
"""

import logging

import numpy as np
from scipy.signal import firwin, lfilter, resample_poly

logger = logging.getLogger(__name__)


def get_mag(arr, row_idx=None, det_option="median"):
    """
    Compute the Euclidean norm of each selected row and subtract a baseline.

    With "median", subtract the median norm of the selected rows. With
    "customfunc", subtract 9.80665, which assumes acceleration in m/s^2.
    An unrecognized option falls back to "median" with a warning.
    Return one value per selected row in the input signal's units.

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
    if det_option not in ["median", "customfunc"]:
        logger.warning(
            "Unknown detrending option %r; defaulting to 'median'.",
            det_option,
        )
        det_option = "median"

    selected = arr if row_idx is None else arr[row_idx]
    mag = np.linalg.norm(selected, axis=1)
    logger.debug("Computed magnitude series with median %.6f", float(np.median(mag)))

    # MATLAB's detrend function is not used,
    #   so we can consider that the default option
    #   for detrending the data is subtracting the median
    if det_option == "median":
        return mag - np.median(mag)

    return mag - 9.80665


def correct_gain(arr, gains):
    """
    Divide each measurement axis by its corresponding gain factor.

    Return the corrected array in the input signal's units.

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


def low_pass(fc, fs, arr, window="hamming"):
    """
    Apply a causal two-tap FIR low-pass filter to a uniformly sampled signal.

    The default Hamming design yields coefficients [0.5, 0.5], a two-sample
    moving average. For this design, changing the valid cutoff argument fc
    does not change the coefficients. Return an array of the input length.

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


def interpolate_to_uniform_time(
    arr,
    ts,
    target_fs,
    *,
    max_gap_seconds,
):
    """Linearly interpolate a continuous signal onto a uniform time grid.

    Parameters
    ----------
    arr : array_like
        One-dimensional signal values, one per timestamp.
    ts : array_like
        Finite, strictly increasing sample times in seconds. May start at
        any time.
    target_fs : float
        Sampling rate of the interpolation grid, in Hz.
    max_gap_seconds : float
        Maximum permitted time between adjacent input samples. Larger gaps
        raise ValueError.

    Returns
    -------
    time, values : numpy.ndarray
        uniform sample times in seconds from ts[0] and the corresponding
        interpolated values. The grid stays within the observed interval.
    """
    ts = np.asarray(ts, dtype=float)
    arr = np.asarray(arr, dtype=float)

    if ts.ndim != 1 or arr.ndim != 1:
        raise ValueError("ts  and arr must be 1D")
    if len(ts) != len(arr):
        raise ValueError("ts and arr must have the same length")
    if len(ts) < 2:
        raise ValueError("At least two samples are required")
    if not np.isfinite(target_fs) or target_fs <= 0:
        raise ValueError("target_fs must be finite and positive")
    if not np.isfinite(max_gap_seconds) or max_gap_seconds <= 0:
        raise ValueError("max_gap_seconds must be finite and positive")
    if not np.all(np.isfinite(ts)) or not np.all(np.isfinite(arr)):
        raise ValueError("ts and arr must contain only finite values")

    # Work in seconds relative to the first observed sample.
    relative_time = ts - ts[0]
    intervals = np.diff(relative_time)

    if np.any(intervals <= 0):
        raise ValueError("Timestamps must be strictly increasing")
    if np.any(intervals > max_gap_seconds):
        raise ValueError(
            "A time gap exceeds max_gap_seconds; "
            "split the recording into continuous segments first"
        )

    # Integer sample indices avoid accumulating arange step errors.
    n = int(np.floor(relative_time[-1] * target_fs)) + 1
    new_time = np.arange(n, dtype=float) / target_fs

    # Keep the grid within the observed interval.
    new_time = new_time[new_time <= relative_time[-1]]
    interp_arr = np.interp(new_time, relative_time, arr)

    return new_time, interp_arr


def resample_to(
    arr,
    ts,
    target_fs=25.0,
    *,
    max_gap_seconds=0.2,
):
    """Resample one continuous signal onto a uniform target time grid.

    Linearly interpolate to an intermediate rate, then apply polyphase
    filtering and integer downsampling. The intermediate rate is at least
    twice target_fs and no lower than the observed average input rate.

    Parameters
    ----------
    arr : array_like
        One-dimensional signal values, one per timestamp.
    ts : array_like
        Finite, strictly increasing sample times in seconds. May start at
        any time.
    target_fs : float, optional
        Output sampling rate in Hz. Defaults to 25.
    max_gap_seconds : float, optional
        Maximum permitted time between adjacent input samples, in seconds.
        Larger gaps raise ValueError. Defaults to 0.2.

    Returns
    -------
    time, values: numpy.ndarray
        Output sample times in seconds from ts[0] and the corresponding resampled values.
    """
    if not np.isfinite(target_fs) or target_fs <= 0:
        raise ValueError("target_fs must be finite and positive")

    ts = np.asarray(ts, dtype=float)
    if ts.ndim != 1 or ts.size < 2:
        raise ValueError("ts must be 1D and contain at least two samples")

    duration = ts[-1] - ts[0]
    if not np.isfinite(duration) or duration <= 0:
        raise ValueError("ts must span a finite, positive duration")

    observed_fs = (ts.size - 1) / duration
    down = max(2, int(np.ceil(observed_fs / target_fs)))
    uniform_fs = down * target_fs

    # --- interpolation should be done ---
    # Fix the option to linear... cubic may be too much
    uniform_time, uniform_arr = interpolate_to_uniform_time(
        arr,
        ts,
        uniform_fs,
        max_gap_seconds=max_gap_seconds,
    )
    logger.info("Interpolation completed")

    resampled = resample_poly(uniform_arr, up=1, down=down, padtype="line")
    return uniform_time[::down], resampled
