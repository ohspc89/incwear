import json
import numpy as np
from datetime import datetime, timedelta
from base import (BaseProcess, CalibProcess, cycle_filt, time_asleep,
                  get_axis_offsets_v2, rate_calc)

# Movesense Active sensors have a buffer of 4 data points.
# We need a function to 'debuffer'
def unbuffer(arr, keyword, rec_type='imu'):
    buff = [arr[i][rec_type][keyword] for i in range(len(arr))]
    N = len(buff)
    buffN = 4
    X = [buff[i][j]['x'] for i in range(N) for j in range(buffN)]
    Y = [buff[i][j]['y'] for i in range(N) for j in range(buffN)]
    Z = [buff[i][j]['z'] for i in range(N) for j in range(buffN)]
    return np.column_stack((X, Y, Z))

class Active(BaseProcess):
    """
    This is a class that contains preprocessed info of json files
    - developmental stage / single sensor only
    """
    def __init__(self, filename, study_tz, offset=None, gs=None,
                 rec_type='imu', **kwargs):
        super().__init__()
        self.info.timezone = study_tz
        f = open(filename)
        data = json.load(f)

        tempaccmat = unbuffer(data['data'], 'ArrayAcc')
        tempvelmat = unbuffer(data['data'], 'ArrayGyro')

        if offset is not None:
            l_aligned = tempaccmat - np.array(offset)
        else:
            l_aligned = tempaccmat

        if gs is not None:
            offset_rm = 9.80665*self.correct_gain(l_aligned,
                                                  get_axis_offsets_v2(gs))
        else:
            offset_rm = l_aligned

        rowidx = None

        accmags = self._get_mag({'L': offset_rm}, rowidx)
        # convert: deg/s -> rad/s
        velmags = self._get_mag({'L': 0.017453*tempvelmat}, rowidx)
        # Up to this point it works
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
        # right now I can't find a good way to estimate exact time
        # from movesense sensors...
        # Addressing the issue temporarily by making use of the filename
        dt_text = filename.split('/')[-1].split('.json')[0].split('Z_')[0]
        dt_obj = datetime.strptime(dt_text, '%Y%m%dT%H%M%S')
        # timestamp
        ts = [data['data'][i][rec_type]['Timestamp']
              for i in range(len(data['data']))]

        rts = {'U': [dt_obj+timedelta(seconds=int(ts[0])/1e6),
                     dt_obj + timedelta(seconds=int(ts[-1])/1e6)]}

        self.info.fname = filename
        self.info.record_times = rts
        self.info.rowidx = rowidx
        if 'fs' in kwargs:
            self.info.fs = kwargs.get('fs')
        else:
            self.info.fs = 52  # Movesense default fs is 52Hz
        self.info.recordlen = {'U': accmags_f['umag'].shape[0]}
        self.measures.accmags = accmags_f
        self.measures.velmags = velmags
        self.measures.thresholds = thresholds
        self.measures.__post_init__()

class ActiveCalib(CalibProcess):
    """
    This is a class that takes calibration json file(s) and stores
    offset values.
    """
    def __init__(self, calib1, g_thresholds=[9.5, 10.1],
                 rec_type='imu', **kwargs):
        super().__init__()
        self.g_thresholds = g_thresholds
        self.info.fs = 52
        f = open(calib1)
        data = json.load(f)

        tempaccmat = unbuffer(data['data'], 'ArrayAcc', rec_type=rec_type)

        calib1_vals = self.get_gs(tempaccmat)

        if 'calib2' in kwargs:
            self.info.fname = [calib1, kwargs.get('calib2')]
            f2 = open(kwargs.get('calib2'))
            data2 = json.load(f2)
            tempaccmat2 = unbuffer(data2['data'], 'ArrayAcc', rec_type=rec_type)
            calib2_vals = self.get_gs(tempaccmat2)

            self.offset = {'L': calib1_vals['offset'],
                           'R': calib2_vals['offset']}
            self.misalign = {'L': calib1_vals['misalign'],
                             'R': calib2_vals['misalign']}
            self.original = {'L': tempaccmat, 'R': tempaccmat2}
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
            self.info.fname = [calib1]
            self.offset = calib1_vals['offset']
            self.misalign = {'L': calib1_vals['misalign']}
            self.original = tempaccmat
            self.processed = calib1_vals['processed']
            self.raw_gs = calib1_vals['gs_orig']
            self.off_gs = calib1_vals['gs']
            self.boost_gs = calib1_vals['gs_boost']
            self.samp_num = calib1_vals['samp_num']
