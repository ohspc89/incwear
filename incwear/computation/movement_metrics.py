"""
Written by Jinseok Oh, Ph.D.
2022/9/13 - present (as of 2025/7/16)

Movement metrics

© 2023-2025 Infant Neuromotor Control Laboratory. All rights reserved.
"""
import numpy as np


def time_asleep(movmat, recordlen, fs, t0=0, t1_user=None):
    """
    Estimate the amount of time the infant was inactive.

    Parameters
    ----------
    movmat : numpy.ndarray
        Movement matrix.
    recordlen : int
        Total length of the recording.
        ex) info.recordlen['R'] of a class object
    fs : int
        Sampling rate
    t0 : int
        Start index (default 0).
    t1_user : int or None
        User-specified end index (default None).

    Returns
    -------
    time_sleep: int
        Estimated inactive times for left and right (in samples).
    """
    # time_asleep calculated based on the original MATLAB code...
    # If 6000 consequent data points (5 min) show:
    #   1) angular velocity less than 0.3
    #   2) acceleration norm less than 1.001
    # then this period is considered as the baby sleeping or inactive.
    # Technically, this is a bit weird... because you would 'ignore'
    # movements with negative peaks. The second condition should be applied
    # to the absolute values of the detrended acc magnitudes.

    # For Axivity sensor recording, there are just too many data points.
    # This approach would not work.

    # Another method to calculate [time asleep] is introduced in
    # Trujillo-Priego et al. (2017) - less than 3 movements in 5 min.
    # An approximation of this method could be:
    #
    #   a'  a  1   bc 2           d   3           4
    #   -----[-+-]--[-+--]---------[--+-]------[--+-]-
    #
    # Let's suppose that 1,2,3,4 are the movement counts (+).
    # Their starts and ends are marked by [ and ].
    # First, you define an anchor and a target.
    # The anchor and the target are the indices of movements separated by
    # no more than two movements.
    # From the image above, the first anchor is 1 and the first target is 2.
    # You then calculate the distance (D):
    #   one point before the start of a target (b) - start of an anchor (a)
    # If the anchor is 1, then instead of a, use a'.

    # i. If D > 6000, you increase [time asleep] by D
    #    Then the new anchor will be the previous target (2), and the new
    #    target will be target + 1 (3).
    #    D is newly defined: d-c (refer to the image above)
    #
    # ii. If FALSE, increase the target by 1 and check if the new distance
    #     b' - a' is greater than 6000 (refer to the imave below).
    #     If that's TRUE, do i. and move on.
    #     If that's FALSE, increase the target one more time do the same.
    #     (D = b'' - a'; D > 6000?)
    #
    #   a'  a  1   b  2           b'  3       b'' 4
    #   -----[-+-]--[-+--]---------[--+-]------[--+-]-
    #
    #     If you still see that D = b'' - a' < 6000, increase the anchor
    #     by 1, and repeat ii.
    #     If D > 6000, do i and move on.

    anchor = 0
    target = 1
    time_sleep = 0
    mvcnt = movmat.shape[0]
    len_5_min = 5 * 60 * fs  # 6000 for fs=20 samples/sec

    while target < (mvcnt-1):
        if target - anchor > 3:
            anchor += 1
            t0 = movmat[anchor][0]

        t1 = movmat[target][0] - 1

        if t1 - t0 > len_5_min:
            time_sleep += t1 - t0
            anchor = target
            t0 = movmat[anchor][0]
            target += 1
        else:
            target += 1
    # When you break out from the loop, make t0 the end of the last move
    t0 = movmat[-1][2] + 1
    t1 = t1_user - 1 if t1_user else recordlen - 1
    if t1-t0 > len_5_min:
        time_sleep += t1 - t0

    return time_sleep


def cycle_filt(movmat, fs=20, thratio=0.2):
    """
    Detect and filter out highly cyclical movements.

    Parameters
    ----------
    movmat : numpy.ndarray
        Movement index matrix.
    fs : int
        Sampling frequency (Hz)
    thratio : float
        Threshold duration between successive movements (in seconds).

    Returns
    -------
    numpy.ndarray
        Filtered movement matrix.
    """
    threshold = int(fs * thratio)
    to_del = []
    i = 0
    while i < (movmat.shape[0] - 2):
        diff = movmat[i + 1, 0] - movmat[i, 2]
        if diff <= threshold:
            j = i + 1   # j could be the last mov idx
            while j < (movmat.shape[0]-1):
                if (movmat[j+1, 0]-movmat[j, 2]) <= threshold:
                    j += 1
                else:
                    break
            # If more than 8 'cycles' are observed, discard the movements
            # (should it be ten or more?)
            # Discussion with Dr. Smith (2/13/23) -> increasing the number
            # testing - doubling the number to 16? -> 40
            if j - i > 39:
                to_del.extend(range(i, j+1))
                # further remove 2 movements within 12 seconds of j'th movement
                # Why two? just...
                k = j+1
                counter = 0
                while k < (movmat.shape[0]-1) and counter < 3:
                    if (movmat[k, 0] - movmat[j, 2]) < 12 * fs:
                        to_del.append(k)
                        counter += 1
                        k += 1
                    else:
                        break
                i = k
            else:
                i = j+1
        else:
            i += 1

    return np.delete(movmat, to_del, axis=0)


def rate_calc(lmovmat, rmovmat, recordlen, fs, thratio=0.2):
    """
    Calculate movement rate and rest-related measures.

    Parameters
    ----------
    lmovmat, rmovmat: numpy.ndarray
        Movement matrices for left and right sides.
    recordlen : tuple
        Lengths of recordings (L, R).
    fs : int
        Sampling frequency.
    thratio : float
        Cycle threshold ratio for filtering.
        Default is 0.2 (seconds) -> 4 samples for 20 S/s, 5 samples for 25 S/s

    Returns
    -------
    dict
        Dictionary of movement rates and durations
        lrate, rrate: movement rates
        lrec_hr, rrec_hr: recording length in hours
        lsleep_hr, rsleep_hr: sleep hours
    """
    lmovs_del = cycle_filt(lmovmat, fs, thratio)
    rmovs_del = cycle_filt(rmovmat, fs, thratio)
    lsleep = time_asleep(lmovs_del, recordlen[0], fs)
    rsleep = time_asleep(rmovs_del, recordlen[1], fs)

    lsleep_5m = lsleep / (60 * fs) - (lsleep / (60 * fs)) % 5
    rsleep_5m = rsleep / (60 * fs) - (rsleep / (60 * fs)) % 5

    lrec_hr = recordlen[0] / (3600 * fs)
    rrec_hr = recordlen[1] / (3600 * fs)
    lsleep_hr = lsleep_5m / 60
    rsleep_hr = rsleep_5m / 60

    lmovrate = lmovs_del.shape[0] / max((lrec_hr - lsleep_hr), 1e-5)
    rmovrate = rmovs_del.shape[0] / max((rrec_hr - rsleep_hr), 1e-5)

    return {'lrate': lmovrate,
            'rrate': rmovrate,
            'lrec_hr': lrec_hr,
            'rrec_hr': rrec_hr,
            'lsleep_hr': lsleep_hr,
            'rsleep_hr': rsleep_hr}

    
def acc_per_mov(accvec, movmat):
    """
    Calculate mean and peak acc magnitude per movement

    Parameters
    ----------
    accvec : numpy.ndarray
        Vector of (absolute) acceleration magnitudes.
    movmat : numpy.ndarray
        Movement index matrix (start, mid, end, lag).

    Returns
    -------
    numpy.ndarray
        [start index | avg acc | peak acc] per movement
    """
    # In case raw, not absolute magnitude is provided
    accvec = abs(accvec)
    return np.array([
        [start, np.mean(accvec[start:end + 1]), max(accvec[start:end + 1])]
        for start, _, end, _ in movmat
        ])
