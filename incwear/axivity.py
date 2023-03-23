from datetime import datetime, timezone, timedelta
import numpy as np
import matplotlib.pyplot as plt
from skdh.io import ReadCwa
from scipy.interpolate import interp1d
from base import BaseProcess, cycle_filt, time_asleep

class Ax6(BaseProcess):
    def __init__(self, Lfilename, Rfilename, det_option='median'):
        super().__init__(Lfilename=Lfilename, Rfilename=Rfilename)
        reader = ReadCwa()
        l_skdh = reader.predict(Lfilename)
        r_skdh = reader.predict(Rfilename)

        # interpolation
        intp = interp1d(l_skdh['time'], 9.81*l_skdh['accel'], axis=0)
        intp_r = interp1d(r_skdh['time'], 9.81*r_skdh['accel'], axis=0)
        intpv = interp1d(l_skdh['time'], l_skdh['gyro'], axis=0)
        intpv_r = interp1d(r_skdh['time'], r_skdh['gyro'], axis=0)

        nt_l = np.round(np.arange(l_skdh['time'][0],
            l_skdh['time'][-1],
            1/20),2)
        nt_r = np.round(np.arange(r_skdh['time'][0],
            r_skdh['time'][-1],
            1/20),2)
        # This is a feature originally prepared for OPAL sensor processing
        # For Axivity sensor, you use the entire recording, so rowidx = None
        rowidx = None
        accmags = self._get_mag(
                {'L': intp(nt_l), 'R': intp_r(nt_r)},
                rowidx,
                det_option=det_option)
        velmags = self._get_mag(
                {'L': intpv(nt_l), 'R': intpv_r(nt_r)},
                rowidx)
        thresholds = self._get_ind_acc_threshold(accmags)

        rts = {'L': [datetime.fromtimestamp(l_skdh['time'][0], timezone.utc),
            datetime.fromtimestamp(l_skdh['time'][-1], timezone.utc)],
            'R': [datetime.fromtimestamp(r_skdh['time'][0], timezone.utc),
                datetime.fromtimestamp(r_skdh['time'][-1], timezone.utc)]}

        # update relevant information
        self.info.fname = [Lfilename, Rfilename]
        self.info.record_times = rts
        self.info.rowidx = rowidx
        self.info.recordlen = {'L': accmags['lmag'].shape[0],
                'R': accmags['rmag'].shape[0]}

        self.measures.accmags = accmags
        self.measures.velmags = velmags
        self.measures.thresholds = thresholds
        self.measures.__post_init__() # update fields
