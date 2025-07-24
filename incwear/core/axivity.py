"""
Script prepared to process Ax6 sensor data by Jinseok Oh, Ph.D.

© 2023-2024 Infant Neuromotor Control Laboratory, All Rights Reserved
"""
# Will delete this ultimately...
import site
from pathlib import Path
import numpy as np
import cwa_metadata as cmd
site.addsitedir('/Users/joh/Downloads/incwear/base')
site.addsitedir('/Users/joh/Downloads/incwear/utils')
from base import BaseProcess
from calib_process import CalibProcess
from io_utils import read_cwa, read_tsv
from calib_utils import get_axis_bias_v2


class Ax6(BaseProcess):
    """
    This is a class that contains preprocessed info of cwa files.
    """
    # (7/18/24) Removing `study_tz` from the parameters
    def __init__(self, calibration, recording, **kwargs):
        """
        Attributes
        ----------
        calibration : str
            Path to the calibration file
        recording : str
            Path to the movement recording file
        **kwargs
        resample : bool
            Whether to downsample to 20Hz (Smith et al. 2015)
        in_en_dts : list or None
            list of timestamps marking the start and the end of the
            dataset to be processed. If None, entire dataset is processed.
        det_option : str
            subtracting magnitude 'median' or 1 g ('customfunc')
        apply_filter : bool
            Whether to apply filter (True) or not (False)
        fc : int
            Cut-off frequency for low-pass filtering
        timezone : str
            timezone of the study site
        """
        super().__init__(filename=recording)
        self.suffix = Path(recording).suffix
        # If two file extensions are different
        if self.suffix not in ['.cwa', '.tsv'] :
            raise ValueError(
                    f"A recording filename's extension should be "
                    "'.cwa' or '.tsv'. "
                    f"Provided name is {recording}, "
                    )

        self._load()    # read files

        self.info.rowidx = None     # How are you going to handle it?
        if self.suffix == '.cwa':
            fs = cmd.cwa_info(recording)['header']['sampleRate']
        else:
            # Rough estimation of fs based on the first 8 timestamps
            fs = round(np.mean([1/x for x in np.diff(self.skdh['time'][0:9])]))
        if any((fs != 25, fs is None)):
            raise ValueError(
                    "A sensor recorded at sampling frequency "
                    "other than 25 Hz. "
                    f"{self.info.fname} at {fs} Hz."
                    )
        self.info.fs = fs

        # Here we can prepare for calibrating values.
        # If you don't provide it, you will just fail.
        self.CalibObj = Ax6Calib(calibration)

        self._normalize(self.CalibObj.info.offset,
                        self.CalibObj.info.off_gs)

        self.compute_magnitudes(9.80665 * self.calibrated, self.info.rowidx)
        self.compute_velocities(self.skdh['gyro'], self.info.rowidx)

        #  convert here? deg to radian
        self.measures.velmags = .017453 * self.measures.velmags

        # If you decide to resample...
        if kwargs.get('resample', False):
            self._resample()

        # If you decide to filter, check if you provided a cut-off frequency
        self.info.fc = kwargs.get('fc', 8)

        # Just make this a self's attribute...
        if kwargs.get('apply_filter', False):
            self._filter()

        # if timezone is provided...
        if 'timezone' in kwargs:
            self.info.timezone = kwargs.get('timezone')

        # final update
        self._update()

    def _load(self):
        if self.suffix == '.cwa':
            self.skdh = read_cwa(self.info.fname)
        elif self.suffix == '.tsv':
            tsv = read_tsv(self.info.fname)
            # This is dictated by the conversion... or is it too risky?
            # elapsed_time | acc_x | acc_y | acc_z | gyro_x | gyro_y | gyro_z
            self.skdh = {'accel': tsv[:, 1:4],
                         'gyro': tsv[:, 4:7],
                         'time': tsv[:, 0]}
        else:
            raise ValueError(f"Unsupported file extension: {self.suffix}")

    def _normalize(self, offset, gs):
        # Remove offset from the measurements
        if offset is not None:
            aligned = self.skdh['accel'] - np.array(offset)
        else:
            aligned = self.skdh['accel']

        # Gain corrected...
        if gs is not None:
            gain_corrected = self.correct_gain(aligned, get_axis_bias_v2(gs))
            
        else:
            gain_corrected = aligned

        # Export offset removed files
        self.calibrated = gain_corrected

    def _resample(self):
        # resample
        re_fs = 20
        self.measures.accmags = self.resample_to(
                self.measures.accmags, self.info.fs, re_fs)
        self.measures.velmags = self.resample_to(
                self.measures.velmags, self.info.fs, re_fs)
        # update the sampling frequency information
        self.info.fs = re_fs

    def _filter(self):
        self.measures.accmags = self.low_pass(
                self.info.fc,
                self.info.fs,
                self.measures.accmags,
        )
        self.measures.velmags = self.low_pass(
                self.info.fc,
                self.info.fs,
                self.measures.velmags
        )

    def _update(self):
        # update thresholds
        self.compute_thresholds()

        self.info.recordlen = self.measures.accmags.shape[0]


class Ax6Calib(CalibProcess):
    """
    Class that takes calibration cwa file(s) and stores
    offset values.
    """
    def __init__(self, filename, absolute=False, **kwargs):
        super().__init__()  # thresholds = [0.9, 1,1]

        # Filling in class parameters
        self.info.fname = filename
        self.absolute = absolute
        if 'stdcut' in kwargs:
            self.stdcut = kwargs.get('stdcut')
        if 'winlen' in kwargs:
            self.winlen = kwargs.get('winlen')

        if filename.endswith('tsv'):
            temp = read_tsv(filename)
            self.raw = temp[:, [1, 2, 3]]
            self.info.fs = round(np.mean([1/x for x in np.diff(temp[0:9, 0])]))
        else:
            temp = read_cwa(filename)
            self.raw = temp['accel']
            self.info.fs = cmd.cwa_info(filename)['header']['sampleRate']

        # offset, gs, ...
        calib1_vals = self.get_gs(self.raw)

        self.info.offset = calib1_vals['offset']
        # this is just for the consistency across classes...
        self.info.misalign = calib1_vals['misalign']
        self.info.raw_gs = calib1_vals['gs_orig']
        self.info.off_gs = calib1_vals['gs']
        self.info.boost_gs = calib1_vals['gs_boost']
        self.info.samp_num = calib1_vals['samp_num']


class Ax6Subject:
    """A subject would have two recordings and two calibrations"""
    def __init__(self, calibration_left, recording_left,
                 calibration_right, recording_right, **kwargs):

        self.left = Ax6(calibration_left, recording_left, **kwargs)
        self.right = Ax6(calibration_right, recording_right, **kwargs)
