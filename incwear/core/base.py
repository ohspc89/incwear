"""
Written by Jinseok Oh, Ph.D.
2022/9/13 - present (as of 2026/4/16)

base.py is a python script porting MakeDataStructure_v2.m (+ α),
    a file prepared to extract data from APDM OPAL V2 sensors.
The class - BaseProcess - is inherited by other classes,
accommodating the need of processing data from different sensors.

© 2023-2026 Infant Neuromotor Control Laboratory. All rights reserved.
"""
import numpy as np
from incwear.utils.data_models import RecordingInfo, Processed, Thresholds
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
                configured_record_times={'Start': None, 'End': None},
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
            thresholds=Thresholds(accth=1.0, naccth=-1.0, method="placeholder")
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
            ttdist=ttdist
        )

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
        return plot_segment(
            self,
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
    def resample_to(arr, orig_fs, re_fs, ts, **kwargs):
        return resample_to(arr, orig_fs, re_fs, ts, **kwargs)


class MergedLeg(BaseProcess):
    """
    A single limb reconstructed from several recording segments collected by
    different physical sensors (e.g. after a left/right sensor swap).

    Rationale
    ---------
    Calibration and thresholding sit at different altitudes:

    * Calibration corrects a specific sensor's offset/gain, so it is a
      per-sensor (hence per-segment) operation, already applied when each
      segment object is built.
    * The acceleration threshold is a property of the infant's limb movement
      over the whole collection, not of the sensor. It is therefore computed
      once on the concatenation of all segments (the full duration), which
      avoids deriving a threshold from a short sub-recording.

    Detection runs *within* each segment using that shared global threshold,
    and the per-segment movement matrices are offset-merged, so no movement is
    ever paired across a splice boundary between two sensors.

    Because this is a ``BaseProcess``, a merged limb is a drop-in for a normal
    leg object: ``plot_segments``, ``computation.movement_metrics`` helpers,
    and ``detect_movements`` all work on it unchanged.

    Parameters
    ----------
    segments : sequence of BaseProcess
        Same-limb segments in chronological order. Each must already have
        ``measures.accmags`` / ``measures.velmags`` computed (i.e. a fully
        constructed ``Ax6`` or similar) and share the same sampling frequency.
    use_v2 : bool
        Threshold method, matching ``compute_thresholds`` (default True).
    ttdist : int
        Threshold-to-threshold distance passed to movement detection.

    Attributes
    ----------
    splice_indices : list[int]
        Global sample index of each internal segment boundary (length n - 1).
    segment_thresholds : list[Thresholds]
        Each segment's own threshold, kept for data-quality comparison against
        the global threshold actually used.
    movmat : numpy.ndarray
        Cached merged movement matrix (global indices; columns
        [start, mid, end, lag]).
    """

    def __init__(self, segments, use_v2=True, ttdist=8):
        super().__init__()
        if len(segments) == 0:
            raise ValueError("MergedLeg requires at least one segment.")

        fs = segments[0].info.fs
        if any(s.info.fs != fs for s in segments):
            raise ValueError(
                "All segments must share the same sampling frequency."
            )

        # 1. Global threshold over the full concatenated limb signal.
        concat_acc = np.concatenate([s.measures.accmags for s in segments])
        concat_vel = np.concatenate([s.measures.velmags for s in segments])
        thresh_fn = get_ind_acc_threshold2 if use_v2 else get_ind_acc_threshold
        global_th = thresh_fn(concat_acc)

        # Per-segment thresholds, retained as a data-quality check on how well
        # calibration reconciled the different sensors.
        self.segment_thresholds = [
            thresh_fn(s.measures.accmags) for s in segments
        ]

        # 2. Detect within each segment using the shared threshold, then shift
        #    the index columns (start, mid, end) into the global timeline.
        movmats = []
        splice_indices = []
        offset = 0
        for seg in segments:
            acc = seg.measures.accmags
            vel = seg.measures.velmags
            over = acc > global_th.accth
            under = acc < global_th.naccth
            th_crossed = over.astype(int) - under.astype(int)

            mm = get_mov(acc, vel, fs, th_crossed, over, under, ttdist=ttdist)
            if mm.size:
                mm = mm.copy()
                mm[:, :3] += offset  # index columns only; column 3 is a lag
                movmats.append(mm)

            offset += seg.info.recordlen
            splice_indices.append(offset)

        # The last boundary equals recordlen; keep only the internal joins.
        self.splice_indices = splice_indices[:-1]
        self.movmat = (
            np.vstack(movmats) if movmats else np.empty((0, 4), dtype=int)
        )
        self._ttdist = ttdist

        # Populate the standard BaseProcess containers so downstream code that
        # reaches into `.measures` / `.info` works without special-casing.
        self.measures = Processed(
            accmags=concat_acc,
            velmags=concat_vel,
            thresholds=global_th,
        )
        self.info.fs = fs
        self.info.recordlen = offset

    def detect_movements(self, ttdist=None):
        """Return the cached merged movement matrix.

        Detection already ran per segment at merge time (to avoid pairing a
        movement across a sensor-swap boundary), so the result is cached rather
        than recomputed on the concatenation.
        """
        if ttdist is not None and ttdist != self._ttdist:
            raise ValueError(
                "MergedLeg movements were detected with "
                f"ttdist={self._ttdist}; rebuild the MergedLeg to change it."
            )
        return self.movmat


def merge_segments(segments, use_v2=True, ttdist=8):
    """
    Merge chronologically ordered same-limb recording segments into one
    :class:`MergedLeg`. See :class:`MergedLeg` for the rationale and the
    parameters. Returns a leg-like object usable anywhere a normal leg is.
    """
    return MergedLeg(segments, use_v2=use_v2, ttdist=ttdist)
