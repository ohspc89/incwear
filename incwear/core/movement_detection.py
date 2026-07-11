"""
Written by Jinseok Oh, Ph.D.
2022/9/13 - present (as of 2026/4/16)

Movement detection related functions

© 2023-2026 Infant Neuromotor Control Laboratory. All rights reserved.
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
    list 
        List of count vectors (same order as accmags keys).
    """
    # raw_counts = map(np.sign, accmags.values())
    raw_counts = np.sign(accmags)
    #acounts = [np.multiply(a, np.greater(v, 0)) for a, v in
    #           zip(raw_counts, velmags.values())]
    return np.multiply(raw_counts, np.greater(velmags, 0))


def mark_tcount(t_count, acc_counts, over_th, pos=True):
    """
    Mark threshold-crossing counts (Tcount) for one polarity, in place.

    This ports the marking + collapsing passes of move_updated_v2.m
    (lines 258-357). For a run of same-sign counts starting above the
    threshold, the mark is placed at the end of the run; a second pass
    then collapses adjacent marks, keeping the later one. The original
    processes the positive polarity first and the negative polarity
    second on the SAME array, so later negative marks may overwrite
    earlier positive ones - hence t_count is modified in place.

    Parameters
    ----------
    t_count : numpy.ndarray
        Tcount array to mark (shared between polarities).
    acc_counts : numpy.ndarray
        Output of get_count(accmags, angvels)
    over_th : numpy.ndarray
        Boolean mask of acceleration values beyond the threshold of
        this polarity (over_accth if pos, under_naccth otherwise).
    pos : bool
        Polarity to mark.

    Returns
    -------
    t_count : numpy.ndarray
        The same array, with counts that crossed the threshold marked.
    """
    corrsign = 1 if pos else -1
    arr_len = len(acc_counts)

    # Marking pass: only indices beyond the threshold whose count has
    # the right sign can act, so iterating over them is equivalent to
    # the original full-array loop (which reads only counts/thresholds).
    candidates = np.where(over_th & (acc_counts == corrsign))[0]
    for j in candidates[candidates < arr_len - 3]:
        if acc_counts[j+1] == corrsign and acc_counts[j+2] == corrsign:
            t_count[j] = 0
            t_count[j+1] = 0
            t_count[j+2] = corrsign
        else:
            t_count[j] = corrsign

    # Collapse pass: adjacent marks of this polarity are redundant;
    # keep the later one. The pass never creates new marks, so
    # iterating over the marks present at its start is equivalent
    # to the original full-array loop.
    marked = np.where(t_count == corrsign)[0]
    for j in marked[marked < arr_len - 2]:
        if t_count[j] == corrsign and t_count[j+1] == corrsign and over_th[j]:
            t_count[j] = 0

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

    # The original marks positive Tcounts first, then negative ones,
    # on the same array. mark_tcount already restricts the work to
    # indices whose count has the right sign (which implies the
    # detrended angular velocity is > 0).
    t_count = np.zeros(len(acounts))
    mark_tcount(t_count, acounts, np.asarray(over_accth, dtype=bool),
                pos=True)
    mark_tcount(t_count, acounts, np.asarray(under_naccth, dtype=bool),
                pos=False)

    return [acounts, t_count]


def _find_movement_end_offset(arr, sidx, addi):
    candidates = np.where(arr[sidx:addi, 1] == -arr[sidx, 2])[0]
    if candidates.size == 0:
        return None
    return int(candidates[0])


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
        Movement matrix: [start, mid(tcount), end, start-to-tc lag].
    """
    counts, tcount_arr = get_cntc(accmags, velmags, over_accth, under_naccth)
    n = len(counts)
    # index | count | tcount | th_crossed
    # Fill an int matrix column-by-column: avoids the float64 column_stack
    # temporary plus a second copy from .astype(int) on long recordings.
    arr = np.empty((n, 4), dtype=np.int64)
    arr[:, 0] = np.arange(n)
    arr[:, 1] = counts
    arr[:, 2] = tcount_arr
    arr[:, 3] = th_crossed
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
        if pairdiff[2] == 0 or pairdiff[0] > ttdiff:
            continue

        sidx = arr_b[i + 1, 0]
        if arr_b[i, 3] == arr_b[i, 2]:
            first_tc = arr_b[i, 0]
        elif arr_b[i, 0] > 0 and arr[arr_b[i, 0] - 1, 3] == arr_b[i, 2]:
            first_tc = arr_b[i, 0] - 1
        else:
            first_tc = max(arr_b[i, 0] - 2, 0)

        fstep = int(maxmov_dt * fs / 2 + 1) - sidx + first_tc

        if 0 < fstep < first_tc:
            k = 0
            while (arr[first_tc - k, 1] == arr[first_tc, 3]):
                k += 1
                if k > fstep or first_tc - k < 0:
                    break
            search_slice = accmags[first_tc - k:first_tc]
            if search_slice.size == 0:
                movstart = first_tc
                start_tc_diff = 0
            else:
                k2 = int(np.argmin(np.abs(search_slice)))
                movstart = first_tc - k + k2 + 1
                start_tc_diff = k - k2 - 1
        else:
            movstart = first_tc
            start_tc_diff = 0

        addi = int(movstart + np.ceil(maxmov_dt * fs))
        movend_offset = _find_movement_end_offset(arr, sidx, addi)
        if movend_offset is None:
            continue

        movidx[i] = [movstart, sidx, sidx + movend_offset, start_tc_diff]

    movidx_nz = movidx[np.nonzero(movidx[:, 0])[0], :]

    for i in range(1, movidx_nz.shape[0]):
        if not any(movidx_nz[i - 1, :]):
            j = i
            while j < movidx_nz.shape[0] and i >= 2:
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

    return movidx_nz[np.nonzero(movidx_nz[:, 0])[0],]
