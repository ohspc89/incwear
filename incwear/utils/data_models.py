"""
Written by Jinseok Oh, Ph.D.
2022/9/13 - present (as of 2026/4/16)

Script storing dataclasses

© 2023-2026 Infant Neuromotor Control Laboratory. All rights reserved.
"""
from dataclasses import dataclass, field
import numpy as np


@dataclass(frozen=True)
class Thresholds:
    """
    Threshold values used for movement detection.

    Attributes
    ----------
    accth : float
        Positive acceleration threshold.
    naccth : float
        Negative acceleration threshold.
    method : str | None
        Optional label describing how thresholds were computed.
    reject : float | None
        Optional peak rejection cutoff used during threshold computation.
    height : float | None
        Optional minimum peak height used during threshold computation.
    """
    accth: float
    naccth: float
    method: str | None = None
    reject: float | None = None
    height: float | None = None


@dataclass
class Processed:
    """
    Store preprocessed measures from raw IMU data.
    All measures are side-specific (Left vs. Right)

    Attributes
    ----------
    accmags : numpy.ndarray
        Detrended linear acceleration magnitude
    velmags : numpy.ndarray
        Detrended angular velocity magnitude
    thresholds : dict
        Dictionary of Positive vs. Negative thresholds
    over_accth, under_naccth : numpy.ndarray
        An indicator array telling if values are above or below a threshold.
    th_crossed : numpy.ndarray
        An indicator array telling if values crossed a threshold
                ( 1: over the positive threshold
                 -1: under the negative threshold
                  0: otherwise)
    """
    accmags: np.ndarray
    velmags: np.ndarray
    thresholds: Thresholds
    over_accth: np.ndarray = field(init=False)
    under_naccth: np.ndarray = field(init=False)
    th_crossed: np.ndarray = field(init=False)

    def __post_init__(self):
        pos_temp = self.accmags > self.thresholds.accth
        neg_temp = self.accmags < self.thresholds.naccth

        self.over_accth = pos_temp
        self.under_naccth = neg_temp

        self.th_crossed = pos_temp.astype(int) - neg_temp.astype(int)


@dataclass
class RecordingInfo:
    """
    Recording-specific information

    Attributes
    ----------
    fname : str
        File names (.cwa or .h5)
    configured_record_times : dict
        Dictionary of times that were configured
    record_times : dict
        Dictionary of timestamps of the first and the last VALID data points
    fs : int
        Sampling frequency (Hz).
    fc : int | None
        Cut off frequency (Hz).
    timezone : str
        Timezone of the location where sensors were setup
    label_r : str
        String used to indicate the right side (required for Opal V2)
    rowidx : numpy.ndarray | list | None
        List of indices of the rows to be processed.
    recordlen : int
        Length of a recording in the number of data points.
    """
    fname: str
    configured_record_times: dict
    record_times: dict
    fs: int
    fc: int | None
    timezone: str
    label_r: str | None
    rowidx: np.ndarray | list | None
    recordlen: int
