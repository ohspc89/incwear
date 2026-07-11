"""
Script prepared to process Ax6 sensor data by Jinseok Oh, Ph.D.

© 2023-2026 Infant Neuromotor Control Laboratory, All Rights Reserved
"""
# Will delete this ultimately...
from pathlib import Path
import logging

import numpy as np

import incwear.core.cwa_metadata as cmd
from incwear.core.base import BaseProcess, merge_segments
from incwear.core.calib_process import CalibProcess
from incwear.utils.io_utils import read_cwa, read_tsv
from incwear.utils.calib_utils import get_axis_bias_v2

logger = logging.getLogger(__name__)


class Ax6(BaseProcess):
    """
    This class stores preprocessed information from Axivity Ax6 files (.cwa).
    """
    def __init__(self, recording, calibration=None,
                 **kwargs):
        """
        Attributes
        ----------
        recording : Path | str
            Path to the movement recording file
        calibration : Path | str | None
            Path to the calibration file
        **kwargs
            re_fs : int
                Frequency to resample at
            trim_window : tuple [float | None, float | None]
                Start and end time (in seconds) to trim before processing.
            det_option : str
                Detrending option ('median' or 'customfunc').
            apply_filter : bool
                Whether to apply low-pass filtering.
            fc : int
                Cut-off frequency for low-pass filtering.
            timezone : str
                Timezone of the study site.
        """
        super().__init__(filename=recording)
        self.suffix = Path(recording).suffix
        # If two file extensions are different
        if self.suffix not in ['.cwa', '.tsv']:
            raise ValueError(
                "A recording filename's extension should be '.cwa' or '.tsv'."
                f"Provided name is {recording}."
            )

        self._load()

        if self.suffix == '.cwa':
            metadata = cmd.cwa_info(recording)
            fs = metadata["header"]["sampleRate"]
            self.info.configured_record_times['Start'] = metadata["header"]["loggingStartTime"]
            self.info.configured_record_times['End'] = metadata["header"]["loggingEndTime"]
            self.info.record_times['Start'] = metadata["first"]["timestampTime"]
            self.info.record_times["End"] = metadata["last"]["timestampTime"]
            self.skdh['time'] = self.skdh['time'] - self.skdh['time'][0]
        else:
            fs = round(np.mean([1/x for x in np.diff(self.skdh['time'][0:9])]))

        # Keep only the raw channels the pipeline consumes; drop any extra
        # full-length arrays the reader may have loaded (e.g. temperature).
        self.skdh = {k: self.skdh[k] for k in ('accel', 'gyro', 'time')}

        if fs is None or fs != 25:
            raise ValueError(
                "A sensor recorded at sampling frequency other than 25 Hz."
                f"{self.info.fname} at {fs} Hz."
            )
        self.info.fs = fs
        self.info.rowidx = self._build_row_index(kwargs.get("trim_window"))


        # Here we can prepare for calibrating values.
        if calibration is not None:
            self.CalibObj = Ax6Calib(calibration)

            self._normalize(self.CalibObj.info.offset,
                            self.CalibObj.info.off_gs)
        else:
            self._normalize(None, None)

        det_option = kwargs.get('det_option', 'median')
        # Convert g -> m/s^2 in place (self.calibrated is a fresh array from
        # _normalize), so no full-size temporary is allocated. Once the
        # magnitude series is extracted, the raw acceleration channel and the
        # calibrated copy are no longer needed, so release them.
        self.calibrated *= 9.80665
        self.compute_magnitudes(self.calibrated, self.info.rowidx,
                                det_option=det_option)
        del self.calibrated
        self.skdh.pop('accel', None)

        self.compute_velocities(self.skdh['gyro'], self.info.rowidx)
        self.skdh.pop('gyro', None)

        #  deg to radian
        self.measures.velmags *= .017453

        # If you decide to resample...
        self.re_fs = kwargs.get('re_fs', None)
        if self.re_fs is not None:
            print(f"[INFO] Resampling at {self.re_fs} Hz...")
            if not isinstance(self.re_fs, int):
                raise ValueError(f"re_fs should be an integer: {self.re_fs}")
            if self.re_fs <= self.info.fs/2:
                raise ValueError(
                    "re_fs cannot be smaller than the nyquist frequency: "
                    f"{self.info.fs/2}")
            self._resample()

        self.info.fc = kwargs.get('fc', 8)
        if kwargs.get('apply_filter', False):
            self._filter()

        self.info.timezone = kwargs.get('timezone', '')
        self._update()

    def _build_row_index(self, trim_window):
        if trim_window is None:
            return None

        start_sec, end_sec = trim_window
        n_rows = self.skdh['accel'].shape[0]

        if start_sec is None and end_sec is None:
            return None

        start_idx = 0 if start_sec is None else int(start_sec * self.info.fs)
        end_idx = n_rows if end_sec is None else int(end_sec * self.info.fs) + 1

        if start_idx < 0 or end_idx < 0:
            raise ValueError("trim_window values must be non-negative or None.")
        if start_idx >= end_idx:
            raise ValueError("trim_window start must be earlier than trim_window end.")

        end_idx = min(end_idx, n_rows)
        return np.arange(start_idx, end_idx)

    def _load(self):
        if self.suffix == '.cwa':
            self.skdh = read_cwa(self.info.fname)
            return

        if self.suffix == '.tsv':
            tsv = read_tsv(self.info.fname)
            if tsv.ndim != 2 or tsv.shape[1] < 7:
                raise ValueError(
                    "TSV input must have at least 7 columns in the order: "
                    "elapsed_time, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z."
                )
            self.skdh = {
                'accel': tsv[:, 1:4],
                'gyro': tsv[:, 4:7],
                'time': tsv[:, 0],
            }
            return
        
        raise ValueError(f"Unsupported file extension: {self.suffix}")

    def _normalize(self, offset, gs):
        # Produce a single fresh, float array (self.calibrated) using at most
        # one full-size copy: subtract the offset (or copy the raw channel)
        # once, then divide by the per-axis gains in place. This avoids the
        # previous two-copy path (aligned + gain_corrected).
        accel = self.skdh['accel']
        if offset is not None:
            calibrated = accel - np.asarray(offset, dtype=float)
        else:
            calibrated = accel.astype(float, copy=True)

        if gs is not None:
            calibrated /= get_axis_bias_v2(gs)

        self.calibrated = calibrated

    def _resample(self):
        if self.info.rowidx is None:
            ts = self.skdh['time']
        else:
            ts = self.skdh['time'][self.info.rowidx]
        # resample
        _, self.measures.accmags = self.resample_to(
                self.measures.accmags, self.info.fs,
                self.re_fs, ts)
        _, self.measures.velmags = self.resample_to(
                self.measures.velmags, self.info.fs,
                self.re_fs, ts)
        # update the sampling frequency information
        self.info.fs = self.re_fs

    def _filter(self):
        self.measures.accmags = self.low_pass(self.info.fc, self.info.fs, self.measures.accmags)
        self.measures.velmags = self.low_pass(self.info.fc, self.info.fs, self.measures.velmags)

    def _update(self):
        self.compute_thresholds()
        self.info.recordlen = self.measures.accmags.shape[0]


class Ax6Calib(CalibProcess):
    """
    Class that takes calibration cwa file(s) and stores offset values.
    """
    def __init__(self, filename, absolute=False, **kwargs):
        """
        accept filename as either Path or str
        """
        super().__init__()  # thresholds = [0.9, 1,1]

        # Filling in class parameters
        filename = Path(filename)
        self.info.fname = filename
        self.absolute = absolute
        if 'stdcut' in kwargs:
            self.stdcut = kwargs.get('stdcut')
        if 'winlen' in kwargs:
            self.winlen = kwargs.get('winlen')

        if filename.suffix == '.tsv':
            temp = read_tsv(filename)
            self.raw = temp[:, [1, 2, 3]]
            self.info.fs = round(np.mean([1/x for x in np.diff(temp[0:9, 0])]))
        else:
            temp = read_cwa(filename)
            self.raw = temp['accel']
            self.info.fs = cmd.cwa_info(filename)['header']['sampleRate']

        calib1_vals = self.get_gs(self.raw)

        self.info.offset = calib1_vals['offset']
        self.info.misalign = calib1_vals['misalign']
        self.info.raw_gs = calib1_vals['gs_orig']
        self.info.off_gs = calib1_vals['gs']
        self.info.boost_gs = calib1_vals['gs_boost']
        self.info.samp_num = calib1_vals['samp_num']


class Ax6Subject:
    """A subject would have two recordings and two calibrations.

    Arguments are keyword-only and recording-first, matching :class:`Ax6`
    (``Ax6(recording, calibration=None)``) and :class:`Ax6SwapSubject`.
    Calibration files are optional.

    Parameters
    ----------
    recording_left, recording_right : Path | str
        Movement recording files for the left and right legs.
    calibration_left, calibration_right : Path | str | None
        Per-sensor calibration files (default None).
    **kwargs
        Forwarded to each :class:`Ax6` (e.g. trim_window, det_option).
    """
    def __init__(self, *, recording_left, recording_right,
                 calibration_left=None, calibration_right=None, **kwargs):

        self.left = Ax6(recording_left, calibration_left, **kwargs)
        self.right = Ax6(recording_right, calibration_right, **kwargs)


class Ax6SwapSubject:
    """
    A subject whose sensors were swapped mid-collection.

    Each limb is provided as an ordered (chronological) list of segment specs,
    one per physical sensor placement. Each segment is calibrated with its own
    sensor's calibration file; a single global acceleration threshold is then
    computed per limb over the full duration, and movements are detected within
    each segment and merged (see :class:`~incwear.core.base.MergedLeg`).

    Parameters
    ----------
    left, right : sequence of (recording, calibration, trim_window)
        One tuple per segment, in time order. ``recording`` and
        ``calibration`` are file paths (the recording is often the same file
        for every segment, trimmed differently); ``calibration`` may be None.
        ``trim_window`` is the ``(start_sec, end_sec)`` window for that
        segment, following the same convention as :class:`Ax6`.
    **kwargs
        Forwarded to every :class:`Ax6` segment (e.g. det_option, apply_filter,
        re_fs, timezone). ``use_v2`` and ``ttdist`` control the merge and are
        consumed here rather than passed to :class:`Ax6`.

    Attributes
    ----------
    left, right : MergedLeg
        Drop-in leg objects, usable exactly like a normal leg.

    Examples
    --------
    >>> subj = Ax6SwapSubject(
    ...     left=[(recording, calibA, (None, 86400)),
    ...           (recording, calibB, (86400, None))],
    ...     right=[(recording_r, calibB, (None, 86400)),
    ...            (recording_r, calibA, (86400, None))],
    ... )
    >>> lmovs = subj.left.detect_movements()
    """
    def __init__(self, left, right, use_v2=True, ttdist=8, **kwargs):
        self.left = self._build_side(left, use_v2, ttdist, **kwargs)
        self.right = self._build_side(right, use_v2, ttdist, **kwargs)

    @staticmethod
    def _build_side(specs, use_v2, ttdist, **kwargs):
        if len(specs) == 0:
            raise ValueError(
                "Each side needs at least one (recording, calibration, "
                "trim_window) segment spec."
            )
        segments = [
            Ax6(recording, calibration, trim_window=trim_window, **kwargs)
            for recording, calibration, trim_window in specs
        ]
        return merge_segments(segments, use_v2=use_v2, ttdist=ttdist)
