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
from incwear.utils.data_models import RecordingInfo, Processed
from incwear.utils.thresholds import (get_ind_acc_threshold,
                                      get_ind_acc_threshold2)
from incwear.utils.plot_segment import plot_segment
from incwear.core.preprocessing import (get_mag, correct_gain,
                                        low_pass, resample_to)
from incwear.core.movement_detection import get_cntc, get_mov


class BaseProcess:
    """
    An object that saves preprocessing outcome of the raw IMU data.

    Inherited by classes including: axivity.Ax6, apdm.OpalV2
    """
    def __str__(self):
        return self._name

    def __repr__(self):
        ret = f"{self._name}("
        for k in self._kw:
            ret += f"{k}={self._kw[k]!r}, "
        if ret[-1] != "(":
            ret = ret[:-2]
        ret += ")"
        return ret

    def __eq__(self, other):
        if isinstance(other, type(self)):
            return self._kw == other._kw
        return False

    def __init__(self, **kwargs):
        """ Parameters will be inherited-class specific """
        self._name = self.__class__.__name__
        self._kw = kwargs

        # Placehodler - will be updated
        self.info = RecordingInfo(
                fname=None,
                record_times={'Start': None, 'End': None},
                fs=0,
                fc=0,
                timezone='',
                label_r=None,
                rowidx=None,
                recordlen=0,
                )

        # Update if file names are provided?
        if 'filename' in kwargs:
            self.info.fname = str(kwargs.get('filename'))

        self.measures = Processed(
            accmags = np.array([0, 0, 0, 0, 0]),
            velmags = np.array([0, 0, 0, 0, 0]),
            thresholds={'accth': 1, 'naccth': -1}
            )

    def compute_magnitudes(self, arr, row_idx=None, det_option='median'):
        self.measures.accmags = get_mag(arr, row_idx, det_option)

    def compute_velocities(self, arr, row_idx=None, det_option='median'):
        self.measures.velmags = get_mag(arr, row_idx, det_option)

    def compute_thresholds(self, use_v2=True):
        if not use_v2:
            self.measures.thresholds = \
                    get_ind_acc_threshold(self.measures.accmags)
        else:
            self.measures.thresholds = \
                    get_ind_acc_threshold2(self.measures.accmags)

        # trigger re-evaluation of dependent flags
        self.measures.__post_init__()

    def detect_movements(self, ttdist=8):
        return get_mov(
                accmags=self.measures.accmags,
                velmags=self.measures.velmags,
                fs=self.info.fs,
                th_crossed=self.measures.th_crossed,
                over_accth=self.measures.over_accth,
                under_naccth=self.measures.under_naccth,
                ttdist=ttdist)

    # You don't need this though
    def get_counts(self):
        return get_cntc(
                accmags=self.measures.accmags,
                velmags=self.measures.velmags,
                over_accth=self.measures.over_accth,
                under_naccth=self.measures.under_naccth
                )

    def plot_segments(self, time_passed, duration=20, side='L',
                      movmat=None, title=None, show=True, savepath=None):
        plot_segment(
                fs=self.info.fs,
                accmags_dict=self.measures.accmags,
                velmags_dict=self.measures.velmags,
                thresholds=self.measures.thresholds,
                time_passed=time_passed,
                duration=duration,
                side=side,
                movmat=movmat,
                title=title,
                show=show,
                savepath=savepath
                )

    @staticmethod
    def correct_gain(arr, gains):
        return correct_gain(arr, gains)

    @staticmethod
    def low_pass(fc, fs, arr, window='hamming'):
        return low_pass(fc, fs, arr, window)

    @staticmethod
    def resample_to(arr, orig_fs, re_fs):
        return resample_to(arr, orig_fs, re_fs)
