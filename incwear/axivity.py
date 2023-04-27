from datetime import datetime, timezone, timedelta
import numpy as np
import matplotlib.pyplot as plt
from skdh.io import ReadCwa
from scipy.interpolate import interp1d, CubicSpline
from base import BaseProcess, cycle_filt, time_asleep, get_axis_offsets

class Ax6(BaseProcess):
    """
    This is a class that contains preprocessed info of cwa files.
    """
    def __init__(self, Lfilename, Rfilename, ground_gs, det_option='median',
            intp_option='cubic', fs=25):
        """
            Parameters:
                ground_gs: list
                    list of offsets (left/right sensor)
                intp_option: str
                    'linear' or 'cubic' (spline)
        """
        super().__init__(Lfilename=Lfilename, Rfilename=Rfilename)
        reader = ReadCwa()
        l_skdh = reader.predict(Lfilename)
        r_skdh = reader.predict(Rfilename)

        # get axis/orientation-wise offsets
        offsets_lr = get_axis_offsets(ground_gs)

        def remove_offset(arr, offsets):
            """
            a local function that removes offsets from measurements

            Parameters:
                arr: numpy.array
                    raw accelerometer data
                offsets: list
                    list of axis/orientation specific offsets

            Returns:
                offset removed accelerometer data
            """
            xoffs = np.array([offsets[0], 0, offsets[1]])
            yoffs = np.array([offsets[2], 0, offsets[3]])
            zoffs = np.array([offsets[4], 0, offsets[5]])
            xidx = np.sign(arr[:,0]).astype(int)+1
            yidx = np.sign(arr[:,1]).astype(int)+1
            zidx = np.sign(arr[:,2]).astype(int)+1
            newx = [x + xoffs[xi] for x, xi in zip(arr[:,0], xidx)]
            newy = [y + yoffs[yi] for y, yi in zip(arr[:,1], yidx)]
            newz = [z + zoffs[zi] for z, zi in zip(arr[:,2], zidx)]
            return np.vstack((newx, newy, newz)).T

        assert(intp_option in ['cubic', 'linear']), "intp_option should be cubic or linear"
        if intp_option == 'cubic':
            intp = CubicSpline(l_skdh['time'],
                    remove_offset(l_skdh['accel'], offsets_lr[0]),
                    axis=0)
            intp_r = CubicSpline(r_skdh['time'],
                    remove_offset(r_skdh['accel'], offsets_lr[1]),
                    axis=0)
            intpv = CubicSpline(l_skdh['time'], l_skdh['gyro'], axis=0)
            intpv_r = CubicSpline(r_skdh['time'], r_skdh['gyro'], axis=0)
        else:
            intp = interp1d(l_skdh['time'],
                    remove_offset(l_skdh['accel'], offsets_lr[0]),
                    axis=0)
            intp_r = interp1d(r_skdh['time'],
                    remove_offset(r_skdh['accel'], offsets_lr[1]),
                    axis=0)
            intpv = interp1d(l_skdh['time'], l_skdh['gyro'], axis=0)
            intpv_r = interp1d(r_skdh['time'], r_skdh['gyro'], axis=0)

        nt_l = np.round(
                np.arange(l_skdh['time'][0], l_skdh['time'][-1], 1/fs), 2)
        nt_r = np.round(
                np.arange(r_skdh['time'][0], r_skdh['time'][-1], 1/fs), 2)
        # This is a feature originally prepared for OPAL sensor processing
        # For Axivity sensor, you use the entire recording, so rowidx = None
        rowidx = None
        accmags = self._get_mag(
                {'L': 9.80665*intp(nt_l), 'R': 9.80665*intp_r(nt_r)},
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

class Ax6Calib(BaseProcess):
    """
    This is a class that takes calibration cwa files and stores
    offset values.
    """
    def __init__(self, Lfilename, Rfilename):
        super().__init__(Lfilename=Lfilename, Rfilename=Rfilename)
        reader = ReadCwa()
        l_skdh = reader.predict(Lfilename)
        r_skdh = reader.predict(Rfilename)

        # determine x, y, z axis
        lgs = np.apply_along_axis(sorted, 0, arr=l_skdh['accel'])
        rgs = np.apply_along_axis(sorted, 0, arr=r_skdh['accel'])

        # left side, measured g's
        # Each orientation is measured for 10 seconds: 250 data points in total
        # We get the average of middle 100 ~ 150 points to estimate
        # axis/orientation specific g measurement.
        # The points should have standard deviation less than 0.01
        # (this is already too high)
        def get_gs(arr, ia, ib):
            """ A function to estimate the measured 1g along each axis """
            while True:
                if any(arr[ia:ib,].std(axis=0) > 0.01):
                    ia+=10
                    ib-=10
                else:
                    gs = arr[ia:ib,].mean(axis=0)
                    break
            return gs
        lpos, lneg, rpos, rneg = map(get_gs,
                [lgs, lgs, rgs, rgs],
                [-200, 50, -200, 50],
                [-50, 200, -50, 200])
        self.ground_gs = [[lneg[0],lpos[0],lneg[1],lpos[1],lneg[2],lpos[2]],
                [rneg[0],rpos[0],rneg[1],rpos[1],rneg[2],rpos[2]]]
