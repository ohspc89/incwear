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
from incwear.utils.io_utils import read_cwa, read_tsv, read_parquet
from incwear.utils.calib_utils import get_axis_bias_v2

logger = logging.getLogger(__name__)


class Ax6(BaseProcess):
    """Load and preprocess one Axivity AX6 recording.

    Parameters
    ----------
    recording : str or pathlib.Path
        Path to a CWA, TSV, or Parquet recording.
    calibration : str or pathlib.Path or None, optional
        Path to a calibration recording. If None, skip axis offset and gain
        correction.
    re_fs : int or None, optional
        Target sampling rate in Hz. If None, skip resampling.
    trim_window : tuple of (float or None, float or None), optional
        Half-open interval [start, end) in seconds from the first source sample.
        Use None for either bound to leave that side unrestricted.
    det_option : {"median", "customfunc"}, optional
        Baseline subtraction for acceleration magnitude. The default, "median",
        subtracts the median of the selected rows. "customfunc" subtracts
        9.80665 m/s^2. Angular velocity magnitude is always median-centered.
    apply_filter : bool, optional
        Apply the two-tap FIR low-pass filter after optional resampling.
    fc : float, optional
        Cutoff argument passed to the FIR design, in Hz. Defaults to 8.
    timezone : str, optional
        Study-site time zone label stored in the recording metadata.

    Attributes
    ----------
    info : RecordingInfo
        Recording metadata, selected row indices, and processing rate.
        For CWA files, record_times contains the first and last block
        reference times.
    measures : Processed
        Processed acceleration and angular velocity signals, in m/s^2 and
        rad/s respectively, and the acceleration thresholds.
    processed_time : numpy.ndarray
        Output sample times in seconds from the first source sample.
        Available only when resampling is enabled.
    """

    def __init__(self, recording, calibration=None, **kwargs):

        super().__init__(filename=recording)
        self.suffix = Path(recording).suffix
        # If two file extensions are different
        accepted_formats = [".cwa", ".tsv", ".parquet"]
        if self.suffix not in accepted_formats:
            raise ValueError(
                f"A recording filename's extension should be one of {accepted_formats}"
                f"Provided name is {recording}."
            )

        self._load()

        if self.suffix == ".cwa":
            metadata = cmd.cwa_info(recording)
            fs = metadata["header"]["sampleRate"]
            self.info.configured_record_times["Start"] = metadata["header"][
                "loggingStartTime"
            ]
            self.info.configured_record_times["End"] = metadata["header"][
                "loggingEndTime"
            ]
            self.info.record_times["Start"] = metadata["first"]["timestampTime"]
            self.info.record_times["End"] = metadata["last"]["timestampTime"]
            self.skdh["time"] = self.skdh["time"] - self.skdh["time"][0]
        else:
            fs = round(np.mean([1 / x for x in np.diff(self.skdh["time"][0:9])]))

        # Keep only the raw channels the pipeline consumes; drop any extra
        # full-length arrays the reader may have loaded (e.g. temperature).
        self.skdh = {k: self.skdh[k] for k in ("accel", "gyro", "time")}

        if fs is None or fs != 25:
            raise ValueError(
                "A sensor recorded at sampling frequency other than 25 Hz."
                f"{self.info.fname} at {fs} Hz."
            )
        self.info.fs = fs
        self.info.rowidx = self._build_row_index(kwargs.get("trim_window"))

        self._compute_calibrated_magnitudes(
            calibration,
            det_option=kwargs.get("det_option", "median"),
        )

        # If you decide to resample...
        self.re_fs = kwargs.get("re_fs", None)
        if self.re_fs is not None:
            print(f"[INFO] Resampling at {self.re_fs} Hz...")
            if not isinstance(self.re_fs, int):
                raise ValueError(f"re_fs should be an integer: {self.re_fs}")
            if self.re_fs <= self.info.fs / 2:
                raise ValueError(
                    "re_fs cannot be smaller than the nyquist frequency: "
                    f"{self.info.fs / 2}"
                )
            self._resample()

        self.info.fc = kwargs.get("fc", 8)
        if kwargs.get("apply_filter", False):
            self._filter()

        self.info.timezone = kwargs.get("timezone", "")
        self._update()

    def _build_row_index(self, trim_window):
        """Return row indices for [start, end), in seconds from the first sample.

        Return None if neither bound is set. Raise ValueError if the bounds
        are invalid or the interval contains no samples.
        """
        if trim_window is None:
            return None

        start_sec, end_sec = trim_window
        if start_sec is None and end_sec is None:
            return None

        for name, value in (("start", start_sec), ("end", end_sec)):
            if value is not None and (not np.isfinite(value) or value < 0):
                raise ValueError(f"trim_window {name} must be finite and non-negative.")

        if start_sec is not None and end_sec is not None and start_sec >= end_sec:
            raise ValueError("trim_window start must be earlier than end.")

        ts = np.asarray(self.skdh["time"], dtype=float)
        n_rows = self.skdh["accel"].shape[0]

        if ts.ndim != 1 or ts.size == 0 or ts.size != n_rows:
            raise ValueError("Expected one timestamp per acceleration row.")
        if not np.all(np.isfinite(ts)) or np.any(ts[1:] <= ts[:-1]):
            raise ValueError("Timestamps must be finite and strictly increasing.")

        elapsed = ts - ts[0]
        start_idx = (
            0 if start_sec is None else np.searchsorted(elapsed, start_sec, side="left")
        )
        end_idx = (
            n_rows
            if end_sec is None
            else np.searchsorted(elapsed, end_sec, side="left")
        )

        if start_idx >= end_idx:
            raise ValueError("trim_window start must be earlier than trim_window end.")

        return np.arange(start_idx, end_idx)

    def _load(self):
        """Load the recording info self.skdh with time, accel, and gyro channels.

        For TSV and Parquet, read elapsed time from the first column, followed
        by three acceleration columns and three angular velocity columns.
        """
        if self.suffix == ".cwa":
            self.skdh = read_cwa(self.info.fname)
            return

        if self.suffix == ".tsv":
            tsv = read_tsv(self.info.fname)
            if tsv.ndim != 2 or tsv.shape[1] < 7:
                raise ValueError(
                    "TSV input must have at least 7 columns in the order: "
                    "elapsed_time, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z."
                )
            self.skdh = {
                "accel": tsv[:, 1:4],
                "gyro": tsv[:, 4:7],
                "time": tsv[:, 0],
            }
            return

        if self.suffix == ".parquet":
            parquet = read_parquet(self.info.fname)
            if parquet.ndim != 2 or parquet.shape[1] < 7:
                raise ValueError(
                    "Parquet input must have at least 7 columns in the order: "
                    "elapsed_time, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z."
                )
            self.skdh = {
                "accel": parquet[:, 1:4],
                "gyro": parquet[:, 4:7],
                "time": parquet[:, 0],
            }
            return

        raise ValueError(f"Unsupported file extension: {self.suffix}")

    def _normalize(self, offset, gs):
        """Store a float copy of the acceleration with optional offset and gain correction.

        Subtract offset before dividing by the gains derived from gs.
        Store the result in self.calibrated, in the input acceleration units.
        """
        accel = self.skdh["accel"]
        if offset is not None:
            calibrated = accel - np.asarray(offset, dtype=float)
        else:
            calibrated = accel.astype(float, copy=True)

        if gs is not None:
            calibrated /= get_axis_bias_v2(gs)

        self.calibrated = calibrated

    def _compute_calibrated_magnitudes(self, calibration, *, det_option):
        """Compute baseline-centered magnitudes for the selected source rows.

        Apply optional axis calibration to acceleration and convert it to m/s^2
        before computing the norm. Median-center the angular velocity norm and
        convert it to rad/s. Release the raw channel references afterward.
        """
        if calibration is not None:
            self.CalibObj = Ax6Calib(calibration)
            self._normalize(self.CalibObj.info.offset, self.CalibObj.info.off_gs)
        else:
            self._normalize(None, None)

        self.calibrated *= 9.80665
        self.compute_magnitudes(
            self.calibrated, self.info.rowidx, det_option=det_option
        )
        del self.calibrated
        self.skdh.pop("accel", None)

        self.compute_velocities(self.skdh["gyro"], self.info.rowidx)
        self.skdh.pop("gyro", None)
        self.measures.velmags *= 0.017453

    def _resample(self, *, max_gap_seconds=0.2):
        """Resample both magnitude signals on a shared time grid.

        Store output times in seconds from the first sample of the source
        recording, preserving the offset of the selected segment.
        """
        original_time = np.asarray(self.skdh["time"], dtype=float)
        rowidx = self.info.rowidx

        ts = original_time if rowidx is None else original_time[rowidx]

        accmags = self.measures.accmags
        velmags = self.measures.velmags

        if not (len(ts) == len(accmags) == len(velmags)):
            raise ValueError(
                "Selected timestamps and both magnitude arrays must align."
            )

        acc_time, new_accmags = self.resample_to(
            accmags, ts, target_fs=self.re_fs, max_gap_seconds=max_gap_seconds
        )

        vel_time, new_velmags = self.resample_to(
            velmags, ts, target_fs=self.re_fs, max_gap_seconds=max_gap_seconds
        )

        if not np.array_equal(acc_time, vel_time):
            raise ValueError("Resampled acceleration and gyro time grids differ.")
        if not (len(acc_time) == len(new_accmags) == len(new_velmags)):
            raise ValueError("Resampled timestamps and magnitude arrays must align.")

        processed_time = (ts[0] - original_time[0]) + acc_time

        self.measures.accmags = new_accmags
        self.measures.velmags = new_velmags
        self.processed_time = processed_time
        self.info.fs = self.re_fs

    def _filter(self):
        """Filter both processed magnitude signals in place using info.fc and info.fs"""
        self.measures.accmags = self.low_pass(
            self.info.fc, self.info.fs, self.measures.accmags
        )
        self.measures.velmags = self.low_pass(
            self.info.fc, self.info.fs, self.measures.velmags
        )

    def _update(self):
        """Recompute acceleration thresholds and store the processed sample count."""
        self.compute_thresholds()
        self.info.recordlen = self.measures.accmags.shape[0]


class Ax6Calib(CalibProcess):
    """
    Estimate axis-specific calibration values from one Ax6 recording.

    Parameters
    ----------
    filename : str or pathlib.Path
        Path to a CWA, TSV, or Parquet calibration recording.
    absolute : bool, optional
        Use absolute deviations when calculating the legacy misalignment estimates.
    **kwargs
        Options forwarded to CalibProcess, including winlen, stdcut, g_thresholds, and strict.

    Attributes
    ----------
    raw : numpy.ndarray
        Calibration acceleration samples with shape (n_samples, 3).
    info : CalibFileInfo
        Axis offsets, posture means, and the row indices of the selected
        calibration windows.
    """

    def __init__(self, filename, absolute=False, **kwargs):
        """
        accept filename as either Path or str
        """
        super().__init__(absolute=absolute, **kwargs)

        # Filling in class parameters
        self.info.fname = filename
        accepted_formats = [".cwa", ".tsv", ".parquet"]

        filename = Path(filename)
        if filename.suffix not in accepted_formats:
            raise ValueError(
                f"A recording filename's extension should be one of {accepted_formats}"
                f"Provided name is {filename}."
            )

        if filename.suffix == ".tsv":
            temp = read_tsv(filename)
            self.raw = temp[:, [1, 2, 3]]
            self.info.fs = round(np.mean([1 / x for x in np.diff(temp[0:9, 0])]))
        elif filename.suffix == ".cwa":
            temp = read_cwa(filename)
            self.raw = temp["accel"]
            self.info.fs = cmd.cwa_info(filename)["header"]["sampleRate"]
        else:
            temp = read_parquet(filename)
            self.raw = temp[:, [1, 2, 3]]
            self.info.fs = round(np.mean([1 / x for x in np.diff(temp[0:9, 0])]))

        calib1_vals = self.get_gs(self.raw)

        self.info.offset = calib1_vals["offset"]
        self.info.misalign = calib1_vals["misalign"]
        self.info.raw_gs = calib1_vals["gs_orig"]
        self.info.off_gs = calib1_vals["gs"]
        self.info.boost_gs = calib1_vals["gs_boost"]
        self.info.samp_num = calib1_vals["samp_num"]


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

    def __init__(
        self,
        *,
        recording_left,
        recording_right,
        calibration_left=None,
        calibration_right=None,
        **kwargs,
    ):

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
        Merged limb objects. Movement indices refer to the concatenated sample
        arrays; elapsed-time gaps between source segments are not represented.

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
        """Process one limb's segments in the supplied order and merge them."""
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
