from datetime import timedelta
import numpy as np
from skdh.io import ReadCwa
from base import BaseProcess, cycle_filt, time_asleep, get_axis_offsets

class Ax6(BaseProcess):
    """
    This is a class that contains preprocessed info of cwa files.
    """
    def __init__(self, Lfilename, Rfilename, ground_gs, study_tz,
            **kwargs):
        """
            Parameters:
                ground_gs: list
                    list of offsets (left/right sensor)
                study_tz: str
                    timezone of the study site (ex. US/Pacific)
                **kwargs
                intp_option: str
                    'none', 'linear' or 'cubic' (spline)
                re_fs: int
                    target [re]sampling frequency
                fs: int
                    sampling frequency
                filter: boolean
                    apply filter (True) or not (False)
                fc: int
                    cut-off frequency for low-pass filtering
        """
        super().__init__(Lfilename=Lfilename, Rfilename=Rfilename)
        reader = ReadCwa()
        l_skdh = reader.predict(Lfilename)
        r_skdh = reader.predict(Rfilename)
        self.info.timezone = study_tz   # prioritize this step

        # get axis/orientation-wise offsets
        offset_rm = map(self.remove_offset,
                [l_skdh['accel'], r_skdh['accel']],
                get_axis_offsets(ground_gs))

        # This is a feature originally prepared for OPAL sensor processing
        # For Axivity sensor, you use the entire recording, so rowidx = None
        # Or should we also put in the time????
        rowidx = None

        if 'intp_option' in kwargs:
            try:
                intp_acc = map(self.resample_to,
                        [l_skdh['time'], r_skdh['time']],
                        offset_rm,
                        [kwargs.get('intp_option'), kwargs.get('intp_option')],
                        [kwargs.get('re_fs'), kwargs.get('re_fs')])
                intp_vel = map(self.resample_to,
                        [l_skdh['time'], r_skdh['time']],
                        [l_skdh['gyro'], r_skdh['gyro']],
                        [kwargs.get('intp_option'), kwargs.get('intp_option')],
                        [kwargs.get('re_fs'), kwargs.get('re_fs')])
                accmags = self._get_mag(
                        {x:9.80665*y for x, y in zip(['L', 'R'], intp_acc)}
                        )
                velmags = self._get_mag(
                        {x:y for x, y in zip(['L', 'R'], intp_vel)})
            except:
                raise ValueError("re_fs should be provided if you're downsampling")
        else:
            accmags = self._get_mag(
                    {x:9.80665*y for x, y in zip(['L', 'R'], offset_rm)},
                    rowidx)
            velmags = self._get_mag(
                    {'L': l_skdh['gyro'], 'R': r_skdh['gyro']},
                    rowidx)
        if 'filter' in kwargs:
            try:
                accmags_f = {
                        'lmag': self.low_pass(kwargs.get('fc'),     # cut-off (fc), hamming window
                            kwargs.get('fs'),                       # first-order FIR low-pass
                            accmags['lmag']),
                        'rmag': self.low_pass(kwargs.get('fc'),
                            kwargs.get('fs'),
                            accmags['rmag'])}
            except:
                raise ValueError("fc and fs should be provided if you're filtering the data")
        else:
            accmags_f = accmags

        thresholds = self._get_ind_acc_threshold(accmags_f)

        rts = {'L': list(map(self.local_to_utc,
            [l_skdh['time'][0], l_skdh['time'][-1]],
            [study_tz, study_tz])),
            'R': list(map(self.local_to_utc,
                [r_skdh['time'][0], r_skdh['time'][-1]],
                [study_tz, study_tz]))}

        # Axivity sensors are werid... times are somewhat weird...
        utcoff = rts['L'][0].utcoffset().total_seconds()
        rts['L'] = list(map(lambda x: x-timedelta(seconds = utcoff),
            [rts['L'][0], rts['L'][1]]))
        rts['R'] = list(map(lambda x: x-timedelta(seconds = utcoff),
            [rts['R'][0], rts['R'][1]]))

        # update relevant information
        self.info.fname = [Lfilename, Rfilename]
        self.info.record_times = rts
        self.info.rowidx = rowidx
        if 'fs' in kwargs:
            self.info.fs = kwargs.get('fs')
        else:
            self.info.fs = 25   # Ax6 default fs is 25Hz
        self.info.recordlen = {'L': accmags_f['lmag'].shape[0],
                'R': accmags_f['rmag'].shape[0]}

        self.measures.accmags = accmags_f
        self.measures.velmags = velmags
        self.measures.thresholds = thresholds
        self.measures.__post_init__() # update fields

class Ax6Calib(BaseProcess):
    """
    This is a class that takes calibration cwa file(s) and stores
    offset values.
    """
    def __init__(self, calib1, **kwargs):
        super().__init__()
        reader = ReadCwa()
        l_skdh = reader.predict(calib1)

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
        # determine x, y, z axis
        # left side, measured g's
        lgs = np.apply_along_axis(sorted, 0, arr=l_skdh['accel'])
        if 'calib2' in kwargs:
            r_skdh = reader.predict(kwargs.get('calib2'))
            rgs = np.apply_along_axis(sorted, 0, arr=r_skdh['accel'])
        # Each orientation is measured for 10 seconds: 250 data points in total
        # We get the average of middle 100 ~ 150 points to estimate
        # axis/orientation specific g measurement.
        # The points should have standard deviation less than 0.01
        # (this is already too high)
            lpos, lneg, rpos, rneg = map(get_gs,
                    [lgs, lgs, rgs, rgs],
                    [-200, 50, -200, 50],
                    [-50, 200, -50, 200])
            self.ground_gs = [[lneg[0],lpos[0],lneg[1],lpos[1],lneg[2],lpos[2]],
                    [rneg[0],rpos[0],rneg[1],rpos[1],rneg[2],rpos[2]]]
        else:
            lpos, lneg = map(get_gs, [lgs, lgs], [-200, 50], [-50, 200])
            self.ground_gs = [lneg[0],lpos[0],lneg[1],lpos[1],lneg[2],lpos[2]]

class Ax6Single(BaseProcess):
    """
    Single Ax6 sensor processing
    """
    def __init__(self, filename, ground_gs, study_tz, **kwargs):
        super().__init__()
        reader = ReadCwa()
        l_skdh = reader.predict(filename)
        self.info.timezone = study_tz   # prioritize this step

        # get axis/orientation-wise offsets
        offset_rm = self.remove_offset(l_skdh['accel'],
                get_axis_offsets(ground_gs))

        # This is a feature originally prepared for OPAL sensor processing
        # For Axivity sensor, you use the entire recording, so rowidx = None
        # Or should we also put in the time????
        rowidx = None

        if 'intp_option' in kwargs:
            try:
                intp_acc = self.resample_to(l_skdh['time'],
                        offset_rm,
                        kwargs.get('intp_option'),
                        kwargs.get('re_fs'))
                intp_vel = self.resample_to(l_skdh['time'],
                        l_skdh['gyro'],
                        kwargs.get('intp_option'),
                        kwargs.get('re_fs'))
                accmags = self._get_mag(
                        {'L':9.80665*intp_acc})
                velmags = self._get_mag(
                        {'L':intp_vel})
            except:
                raise ValueError("re_fs should be provided if you're downsampling")
        else:
            accmags = self._get_mag(
                    {'L':9.80665*offset_rm},
                    rowidx)
            velmags = self._get_mag(
                    {'L': l_skdh['gyro']},
                    rowidx)
        if 'filter' in kwargs:
            try:
                accmags_f = {
                        'umag': self.low_pass(kwargs.get('fc'),
                            kwargs.get('fs'),
                            accmags['umag'])}
            except:
                raise ValueError("fc and fs should be provided if you're filtering the data")
        else:
            accmags_f = accmags

        thresholds = self._get_ind_acc_threshold(accmags_f)

        rts = {'U': list(map(self.local_to_utc,
            [l_skdh['time'][0], l_skdh['time'][-1]],
            [study_tz, study_tz]))}

        # Axivity sensors are werid... times are somewhat weird...
        utcoff = rts['U'][0].utcoffset().total_seconds()
        rts['U'] = list(map(lambda x: x-timedelta(seconds = utcoff),
            [rts['U'][0], rts['U'][1]]))

        # update relevant information
        self.info.fname = filename
        self.info.record_times = rts
        self.info.rowidx = rowidx
        if 'fs' in kwargs:
            self.info.fs = kwargs.get('fs')
        else:
            self.info.fs = 25   # Ax6 default fs is 25Hz
        self.info.recordlen = {'U': accmags_f['umag'].shape[0]}
        self.measures.accmags = accmags_f
        self.measures.velmags = velmags
        self.measures.thresholds = thresholds
        self.measures.__post_init__() # update fields
