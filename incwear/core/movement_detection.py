"""
Written by Jinseok Oh, Ph.D.
2022/9/13 - present (as of 2025/7/16)

Movement detection related functions

© 2023-2025 Infant Neuromotor Control Laboratory. All rights reserved.
"""
import numpy as np


def get_count(accmags, velmags):
    """
    Calculate sign-based count vector (acc sign * vel > 0 mask).

    Parameters
    ----------
    accmags : numpy.ndarray
        Acceleration magnitudes
    velmags : numpy.ndarray
        Angular velocity magnitudes

    Returns
    -------
    acounts2: list
        List of count vectors (same order as accmags keys).
    """
    # raw_counts = map(np.sign, accmags.values())
    raw_counts = np.sign(accmags)
    #acounts = [np.multiply(a, np.greater(v, 0)) for a, v in
    #           zip(raw_counts, velmags.values())]
    acounts = np.multiply(raw_counts,
                          np.greater(velmags, 0))
    return acounts


def mark_tcount(over_th_arr, acc_counts, pos=True):
    """
    Parameters
    ----------
    over_th_arr : numpy.ndarray
        Indices of data points over a threshold (ex: over_posth_l).
    acc_counts : numpy.ndarray
        Output of get_count(accmags, angvels)

    Returns
    -------
    t_count : numpy.ndarray
        Nonzero counts that crossed a positive or a negative threshold.
    """
    corrsign = 1 if pos else -1
    arr_len = len(acc_counts)
    t_count = np.zeros(arr_len)
    for j in over_th_arr:
        # "over_th_arr" has the indices of the acceleration values
        #   that are over a threshold (left or right).
        # [j] gives one of those indices.
        # It could be that the Tcount value at the current index
        #   may have been set by the previous index
        #   (ex. [j-2] satisfied the "else" condition so Tcount[j] = 1 or -1)
        # If so, skip to the next index.
        # Also, if the next index is not zero, then the current
        #   Tcount[j] is considered as a redundant count and
        #   marked off (this is from the original MATLAB code).
        #   So we can skip such indices here.
        if all((j < (arr_len-2), all(t_count[j:j+2] != corrsign))):
            if all(acc_counts[j:j+3] == corrsign):
                t_count[j+2] = corrsign
            else:
                t_count[j] = corrsign

    nz_tcount = np.nonzero(t_count)[0]      # non-zero Tcounts
    nztc_len = len(nz_tcount)

    # Remove duplicates
    for i, j in enumerate(nz_tcount):
        if (i <= (nztc_len-2)) and (nz_tcount[i+1] == (j+1)):
            t_count[j] = 0
        elif (i == (nztc_len-1)) and (nz_tcount[i-1] == (j-1)):
            t_count[j-1] = 0

    return t_count


def get_cntc(accmags, velmags, over_accth, under_naccth):
    """
    Compute count and thresholded count (t-count) vectors.

    This function uses sign-based acceleration counts, angular velocity masking,
    and threshold-crossing logic to compute movement-related signal indices.

    Parameters
    ----------
    accmags : numpy.ndarray
        Acceleration magnitude arrays
    velmags : numpy.ndarray
        Angular velocity magnitude arrays, same structure.
    over_accth : numpy.ndarray
        Boolean masks for acceleration values above the positive threshold.
    under_naccth : numpy.ndarray
        Boolean masks for acceleration values below the negative threshold.

    Returns
    -------
    tcounts: list
        [count array, t-count array], where:
            - count: sign(acc) * (vel > 0) mask
            - t-count: threshold-crossing events (+/- 1)
    """
    acounts = get_count(accmags, velmags)

    # Let's start with collecting all acc values over the threshold
    # The output of np.where would be a tuple - so take the first value
    # I do this to reduce the preprocessing time...
    over_posth = np.where(over_accth)[0]
    under_negth = np.where(under_naccth)[0]

    # angular velocity should be taken into account...
    # let's just make sure that the detrended angvel[i] > 0
    angvel_gt = np.nonzero(acounts)[0]

    over_posth = np.intersect1d(over_posth, angvel_gt)
    under_negth = np.intersect1d(under_negth, angvel_gt)

    ttcounts = mark_tcount(over_posth, acounts, pos=True)
    tntcounts = mark_tcount(under_negth, acounts, pos=False)

    return [acounts, ttcounts + tntcounts]


def get_mov(accmags, velmags, fs, th_crossed, over_accth, under_naccth,
            ttdist=8):
    """
    Detect movement segments based on threshold crossings and signal polarity.

    Parameters
    ----------
    accmags : numpy.ndarray
        Detrended acceleration magnitudes.
    velmags : numpy.ndarray
        Angular velocity magnitudes.
    fs : int
        Sampling frequency.
    th_crossed : numpy.ndarray
        Threshold crossing polarity array.
    over_accth, under_naccth : numpy.ndarray
        Threshold masks for acceleration.
    ttdist : int
        Movement threshold-to-threshold distance (in # of samples @ 20 Hz).

    Returns
    -------
    np.ndarray
        Movement matrix: [start, mid(tcount), end, pre-tc, lag].
    """
    tcounts = get_cntc(accmags, velmags, over_accth, under_naccth)
    counts, tcount_arr = tcounts
    n = len(counts)
    # index | count | tcount | th_crossed
    arr = np.column_stack((np.arange(n),
                           counts,
                           tcount_arr,
                           th_crossed)).astype(int)
    arr_b = arr[np.nonzero(arr[:, 2])[0], :]
    # arr_b[0, ] would be the row with the first nonzero tcount.
    # Smith et al. (2015) wrote:
    #   "The start of a movement was defined as simultaneous acceleration
    #    above a magnitude threshold and angular velocity greater than 0."
    # So I originally thought that the start of a movement should be
    # a datapoint with tcount value 1 or -1.
    # However, a typical movement's acceleration magnitude profile would
    # rather be sinusoidal.

    movidx = np.zeros((arr_b.shape[0], 4), dtype=int)

    maxmov_dt = 1.5
    ttdiff = int(ttdist * fs / 20)

    for i in range(arr_b.shape[0] - 1):
        pairdiff = np.diff(arr_b[i:i+2, :], axis=0).ravel()
        if pairdiff[2] != 0 and pairdiff[0] <= ttdiff:
            sidx = arr_b[i + 1, 0]
            if arr_b[i, 3] == arr_b[i, 2]:
                first_tc = arr_b[i, 0]
            elif arr[arr_b[i, 0] - 1, 3] == arr_b[i, 2]:
                first_tc = arr_b[i, 0] - 1
            else:
                first_tc = arr_b[i, 0] - 2

            fstep = int(maxmov_dt * fs / 2 + 1) - sidx + first_tc

            if 0 < fstep < first_tc:
                k = 0
                while (arr[first_tc - k, 1] == arr[first_tc, 3]):
                    k += 1
                    if k > fstep:
                        break
                k2 = np.argmin(abs(accmags[first_tc-k:first_tc]))
                movstart = first_tc - k + k2 + 1
                start_tc_diff = k - k2 - 1
            else:
                movstart = first_tc
                start_tc_diff = 0

            addi = int(movstart + np.ceil(maxmov_dt * fs))
            try:
                movend = np.where(
                        arr[sidx:addi, 1] == -arr[sidx, 2])[0][0]
                movidx[i] = [movstart, sidx, sidx + movend, start_tc_diff]
            except:
                continue

    movidx_nz = movidx[np.nonzero(movidx[:, 0])[0], :]

    for i in range(1, movidx_nz.shape[0]):
        if not any(movidx_nz[i-1, :]):
            j = i
            while j < movidx_nz.shape[0]:
                if movidx_nz[j, 0] < movidx_nz[i-2, 2]:
                    movidx_nz[j, :] = np.zeros(4)
                    j += 1
                else:
                    break
        elif movidx_nz[i, 0] < movidx_nz[i-1, 2]:
            if movidx_nz[i, 3] > 0:
                backstep = movidx_nz[i, 3]
                ith_st = movidx_nz[i, 0]
                broken = 0
                while backstep > -1:
                    if ith_st >= movidx_nz[i-1, 2]:
                        movidx_nz[i, 0] = ith_st
                        broken = 1
                        break
                    ith_st += 1
                    backstep -= 1
                    if not broken:
                        movidx_nz[i, ] = np.zeros(4)
            else:
                movidx_nz[i, ] = np.zeros(4)

    movidx_nz2 = movidx_nz[np.nonzero(movidx_nz[:, 0])[0],]

    return movidx_nz2
