#from datetime import timedelta
import numpy as np
from skdh.io import ReadCwa
from base import BaseProcess, cycle_filt,\
        time_asleep, get_axis_offsets, get_axis_offsets_v2, rate_calc

class Ax6(BaseProcess):
    """
    This is a class that contains preprocessed info of cwa files.
    """
    def __init__(self, Lfilename, Rfilename, ground_gs, study_tz, **kwargs):
        """
            Parameters:
                ground_gs: list
                    list of offsets (left/right sensor)
                study_tz: str
                    timezone of the study site (ex. US/Pacific)
                **kwargs
                misalign: dict(list)
                    a dictionary whose items are left and right
                    sensors' axis misalignment error amounts
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

        # So before adjusting for gain and offset error,
        # you need to correct the misalignment of axes.
        if 'offset' in kwargs:
            l_aligned = l_skdh['accel'] - kwargs.get('offset')['L']
            r_aligned = r_skdh['accel'] - kwargs.get('offset')['R']
        else:
            l_aligned = l_skdh['accel']
            r_aligned = r_skdh['accel']

        # get axis/orientation-wise offsets
        offset_rm = map(self.remove_offset,
                        [l_aligned, r_aligned],
                        get_axis_offsets_v2(ground_gs))

        # This is a feature originally prepared for OPAL sensor processing
        # For Axivity sensor, you use the entire recording, so rowidx = None
        # Or should we also put in the time????
        rowidx = None

        if 'intp_option' in kwargs:
            try:
                intp_acc = map(self.resample_to,
                               [l_skdh['time'], r_skdh['time']],
                               offset_rm,
                               [kwargs.get('intp_option'),
                                kwargs.get('intp_option')],
                               [kwargs.get('re_fs'), kwargs.get('re_fs')])
                intp_vel = map(self.resample_to,
                               [l_skdh['time'], r_skdh['time']],
                               [l_skdh['gyro'], r_skdh['gyro']],
                               [kwargs.get('intp_option'),
                                kwargs.get('intp_option')],
                               [kwargs.get('re_fs'), kwargs.get('re_fs')])
                accmags = self._get_mag(
                        {x: 9.80665*y for x, y in zip(['L', 'R'], intp_acc)}
                        )
                velmags = self._get_mag(
                        dict(zip(['L', 'R'], intp_vel)))
            except ValueError:
                print("Missing: re_fs for downsampling the data")
        else:
            accmags = self._get_mag(
                    {x: 9.80665*y for x, y in zip(['L', 'R'], offset_rm)},
                    rowidx)
            velmags = self._get_mag(
                    {'L': l_skdh['gyro'], 'R': r_skdh['gyro']},
                    rowidx)
        # convert here?
        velmags['lmag'] = 0.017453*velmags['lmag']
        velmags['rmag'] = 0.017453*velmags['rmag']

        if 'filter' in kwargs:
            try:
                accmags_f = {
                        # cut-off (fc), hamming window
                        # first-order FIR low-pass
                        'lmag': self.low_pass(kwargs.get('fc'),
                                              kwargs.get('fs'),
                                              accmags['lmag']),
                        'rmag': self.low_pass(kwargs.get('fc'),
                                              kwargs.get('fs'),
                                              accmags['rmag'])}
            except ValueError:
                print("Missing: fc and fs for filtering the data")
        else:
            accmags_f = accmags

        thresholds = self._get_ind_acc_threshold(accmags_f)

        rts = {'L': list(map(self._calc_datetime,
                             [l_skdh['time'][0], l_skdh['time'][-1]])),
               'R': list(map(self._calc_datetime,
                             [r_skdh['time'][0], r_skdh['time'][-1]]))}

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
        self.measures.__post_init__()  # update fields

class Ax6Calib(BaseProcess):
    """
    This is a class that takes calibration cwa file(s) and stores
    offset values.
    """
    def __init__(self, calib1, absolute=False, **kwargs):
        super().__init__()
        reader = ReadCwa()
        ss_l = reader.predict(calib1)['accel']
        self.absolute = absolute

        def find_window_both(arr):
            """ This is a wrapper to use find_window on a dictionary """
            sns = find_sns(arr)
            return dict(zip(['p', 'n'],     # p: positive; n: negative
                            map(find_window,
                                [arr, arr],
                                list(sns.values()))))

        def find_window(arr, sampnums):
            """
            A function to find sample numbers of THE window (6 seconds)
            The window should have adjacent points, and the std of the points
            should be no greater than 0.02.
            """
            ia = 0
            ib = 150
            while True:
                # Continuity should be kept!
                # 0.02 is experimental - could adjust later (6/22/23)
                if all((all(np.diff(sampnums[ia:ib]) == 1),
                        np.std(arr[sampnums[ia:ib]]) < 0.01)):
                    break   # else unnecessary after break!
                ia += 10
                ib += 10
            return sampnums[ia:ib]

        def find_sns(arr):
            """ A function to find SampleNumberS """
            # Axivity sensors suck....
            idp = np.where(np.logical_and(arr < 1.1, arr > 0.9))[0]
            idn = np.where(np.logical_and(arr < -0.9, arr > -1.1))[0]
            # p = positive; n = negative
            return {'p': idp, 'n': idn}

        def get_gs(ndarr, absolute=self.absolute):
            """ A function to calculate gain, offset, and misalignment """
            # dict; keys are 'pos' and 'neg'
            x_w, y_w, z_w = map(find_window_both,
                                [ndarr[:, 0], ndarr[:, 1], ndarr[:, 2]])
            # non-g sample numbers
            x_png = sorted(np.concatenate((y_w['p'], z_w['p'])))
            x_nng = sorted(np.concatenate((y_w['n'], z_w['n'])))
            y_png = sorted(np.concatenate((x_w['p'], z_w['p'])))
            y_nng = sorted(np.concatenate((x_w['n'], z_w['n'])))
            z_png = sorted(np.concatenate((x_w['p'], y_w['p'])))
            z_nng = sorted(np.concatenate((x_w['n'], y_w['n'])))
            x_non_g = sorted(np.concatenate((x_png, x_nng)))
            y_non_g = sorted(np.concatenate((y_png, y_nng)))
            z_non_g = sorted(np.concatenate((z_png, z_nng)))

            # Correct method of calculating offset - 6/25/23 (GEL)
            offset = list(map(np.mean,
                              [ndarr[x_non_g, 0],
                               ndarr[y_non_g, 1],
                               ndarr[z_non_g, 2]]))
            # measured g values, offset NOT removed
            gs_orig = list(map(np.mean,
                               [ndarr[x_w['n'], 0],
                                ndarr[x_w['p'], 0],
                                ndarr[y_w['n'], 1],
                                ndarr[y_w['p'], 1],
                                ndarr[z_w['n'], 2],
                                ndarr[z_w['p'], 2]]))
            # Remove offsets
            arr2 = ndarr - np.array(offset)
            # Then check for the amount of misalignment
            # Use only the positive valules... doesn't matter much
            if absolute:
                xm, ym, zm = map(lambda x: np.mean(abs(x)),
                                 [arr2[x_png, 0], arr2[y_png, 1], arr2[z_png, 2]])
            else:
                xm, ym, zm = map(np.mean,
                                 [arr2[x_png, 0], arr2[y_png, 1], arr2[z_png, 2]])
            # measured g values, offset removed
            gs = list(map(np.mean,
                          [arr2[x_w['n'], 0],
                           arr2[x_w['p'], 0],
                           arr2[y_w['n'], 1],
                           arr2[y_w['p'], 1],
                           arr2[z_w['n'], 2],
                           arr2[z_w['p'], 2]]))
            # misaglinment reflected (percentage boost)
            gs_boost = [gs[0]*(1+xm), gs[1]*(1+xm),
                        gs[2]*(1+ym), gs[3]*(1+ym),
                        gs[4]*(1+zm), gs[5]*(1+zm)]

            return {'offset': offset,
                    'misalign': [xm, ym, zm],
                    'processed': arr2,
                    'gs_orig': gs_orig,
                    'gs': gs,
                    'gs_boost': gs_boost,
                    'samp_num': {'x': x_w, 'y': y_w, 'z': z_w}}

        calib1_vals = get_gs(ss_l)

        if 'calib2' in kwargs:
            ss_r = reader.predict(kwargs.get('calib2'))['accel']
            calib2_vals = get_gs(ss_r)

            self.offset = {'L': calib1_vals['offset'],
                           'R': calib2_vals['offset']}
            self.misalign = {'L': calib1_vals['misalign'],
                             'R': calib2_vals['misalign']}
            self.original = {'L': ss_l, 'R': ss_r}
            self.processed = {'L': calib1_vals['processed'],
                              'R': calib2_vals['processed']}
            self.raw_gs = {'L': calib1_vals['gs_orig'],
                           'R': calib2_vals['gs_orig']}
            self.off_gs = {'L': calib1_vals['gs'],
                           'R': calib2_vals['gs']}
            self.boost_gs = {'L': calib1_vals['gs_boost'],
                             'R': calib2_vals['gs_boost']}
            self.samp_num = {'L': calib1_vals['samp_num'],
                             'R': calib2_vals['samp_num']}
        else:
            self.offset = calib1_vals['offset']
            # this is just for the consistency across classes...
            self.misalign = {'L': calib1_vals['misalign']}
            self.original = ss_l
            self.processed = calib1_vals['processed']
            self.raw_gs = calib1_vals['gs_orig']
            self.off_gs = calib1_vals['gs']
            self.boost_gs = calib1_vals['gs_boost']
            self.samp_num = calib1_vals['samp_num']

class Ax6Single(BaseProcess):
    """
    Single Ax6 sensor processing
    """
    def __init__(self, filename, study_tz, ground_gs=None, **kwargs):
        super().__init__()
        reader = ReadCwa()
        l_skdh = reader.predict(filename)
        self.info.timezone = study_tz   # prioritize this step

        if 'offset' in kwargs:
            l_aligned = l_skdh['accel'] - kwargs.get('offset')
        else:
            l_aligned = l_skdh['accel']

        # get axis/orientation-wise offsets
        if ground_gs is not None:
            offset_rm = self.remove_offset(l_aligned,
                                           get_axis_offsets_v2(ground_gs))
        else:
            offset_rm = l_aligned

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
                        {'L': 9.80665*intp_acc})
                velmags = self._get_mag(
                        {'L': intp_vel})
            except ValueError:
                print("Missing: re_fs for downsampling the data")
                # raise ValueError("Missing: re_fs for downsampling")
        else:
            accmags = self._get_mag(
                    {'L': 9.80665*offset_rm},
                    rowidx)
            velmags = self._get_mag(
                    {'L': l_skdh['gyro']},
                    rowidx)
        # convert here?
        velmags['umag'] = 0.017453*velmags['umag']

        if 'filter' in kwargs:
            try:
                accmags_f = {
                        'umag': self.low_pass(kwargs.get('fc'),
                                              kwargs.get('fs'),
                                              accmags['umag'])}
            except ValueError:
                print("Missing: fc and fs for filtering the data")
        else:
            accmags_f = accmags

        thresholds = self._get_ind_acc_threshold(accmags_f)

        rts = {'U': list(map(self._calc_datetime,
                             [l_skdh['time'][0], l_skdh['time'][-1]]))}

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
        self.measures.__post_init__()  # update fields
