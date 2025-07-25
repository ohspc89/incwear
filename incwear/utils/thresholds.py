"""
Written by Jinseok Oh, Ph.D.
2022/9/13 - present (as of 2025/7/16)

base.py is a python script porting MakeDataStructure_v2.m (+ α),
    a file prepared to extract data from APDM OPAL V2 sensors.
The class - BaseProcess - is inherited by other classes,
accommodating the need of processing data from different sensors.

© 2023-2025 Infant Neuromotor Control Laboratory. All rights reserved.
"""
import numpy as np
from scipy.signal import find_peaks


def calc_ind_threshold(maxprop):
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
    return (np.mean(maxprop['peak_heights']) -
            np.std(maxprop['peak_heights']))


# This is equivalent to the MATLAB script
def get_ind_acc_threshold(accmags, calc_thresh_fn=calc_ind_threshold,
                          reject=3.2501, height=1.0):
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
    thresholds: dict
         accth: positive threshold
         naccth: negative threshold
    """
    mags2 = accmags.copy()
    posvals = [u if 0 < u < reject else 0 for u in mags2]
    negvals = [abs(u) if -reject < u < 0 else 0 for u in mags2]

    pnpks = []
    for mag in [posvals, negvals]:
        pnpks.extend(find_peaks(np.array(mag), height=height)[1])

    accths = list(map(calc_thresh_fn, pnpks))

    tkeys = ['accth', 'naccth']
    signvec = [1, -1]

    return dict(zip(tkeys, np.multiply(signvec, accths)))


def get_ind_acc_threshold2(accmags, reject=3.2501, height=1.0):
    """
    Alternate threshold calculation: a more correct form of Smith et al. (2015)

    Parameters
    ----------
    accmags : numpy.ndarray
        Detrended acceleration magnitudes.

    Returns
    -------
    dict
        Thresholds.
    """
    mags2 = accmags.copy()

    def thresholds_new(mag):
        # make positive values 0
        magconvt = [x if x < 0 else 0 for x in mag]
        # then rectify
        magrect = np.array([abs(y) for y in magconvt])
        loc, pks = find_peaks(mag, height=height)
        locn, pksn = find_peaks(magrect, height=height)
        hehe = loc[np.where(pks['peak_heights'] < reject)[0]]
        hehen = locn[np.where(pksn['peak_heights'] < reject)[0]]
        pos_th = np.mean(mag[hehe]) - np.std(mag[hehe])
        neg_th = np.mean(magrect[hehen]) - np.std(magrect[hehen])
        return [pos_th, neg_th]

    accths = thresholds_new(mags2)

    tkeys = ['accth', 'naccth']
    signvec = [1, -1]

    return dict(zip(tkeys, np.multiply(signvec, accths)))
