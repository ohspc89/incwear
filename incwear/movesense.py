import json
import numpy as np
from datetime import datetime, timedelta
from base import (BaseProcess, cycle_filt, time_asleep,
                  get_axis_offsets_v2, rate_calc)

# Movesense Active sensors have a buffer of 4 data points.
# We need a function to 'debuffer'
def unbuffer(arr, keyword):
    buff = [arr[i]['imu'][keyword] for i in range(len(arr))]
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
    def __init__(self, filename, ground_gs, study_tz,
                 **kwargs):
        super().__init__()
        self.info.timezone = study_tz
        f = open(filename)
        data = json.load(f)

        tempaccmat = unbuffer(data['data'], 'ArrayAcc')
        tempvelmat = unbuffer(data['data'], 'ArrayGyro')

        if 'misalign' in kwargs:
            l_aligned = tempaccmat - kwargs.get('misalign')['L']
        else:
            l_aligned = tempaccmat

        if ground_gs is not None:
            offset_rm = 9.80665*self.remove_offset(l_aligned,\
                    get_axis_offsets_v2(ground_gs))
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
        # right now I can't find a good way to estimate exact time from movesense sensors...
        # Addressing the issue temporarily by making use of the filename
        dt_text = filename.split('/')[-1].split('.json')[0].split('Z_')[0]
        dt_obj = datetime.strptime(dt_text, '%Y%m%dT%H%M%S')
        # timestamp
        ts = [data['data'][i]['imu']['Timestamp'] for i in range(len(data['data']))]

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

class ActiveCalib(BaseProcess):
    """
    This is a class that takes calibration json file(s) and stores
    offset values.
    """
    def __init__(self, calib1, absolute=False, **kwargs):
        super().__init__()
        self.absolute = absolute
        f = open(calib1)
        data = json.load(f)

        tempaccmat = unbuffer(data['data'], 'ArrayAcc')

        def find_window_both(arr):
            """ This is a wrapper to use find_window on a dictionary """
            sns = find_sns(arr)
            return dict(zip(['p', 'n'],
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
            ib = 240  # 52Hz x 6 = 312
            while True:
                # Continuity should be kept!
                # 0.02 is experimental - could adjust later (6/22/23)
                if all((all(np.diff(sampnums[ia:ib]) == 1),
                        np.std(arr[sampnums[ia:ib]]) < 0.02)):
                    break   # else unnecessary after break!
                ia += 10
                ib += 10
            return sampnums[ia:ib]

        def find_sns(arr):
            """ A function to find SampleNumberS """
            # Movesense measures in m/s^2
            idp = np.where(np.logical_and(arr < 10.1, arr > 9.5))[0]
            idn = np.where(np.logical_and(arr < -9.5, arr > -10.1))[0]
            return {'p': idp, 'n': idn}

        def get_gs(ndarr, absolute=self.absolute):
            """ A function to calculate gs and misalignment degrees """
            x_w, y_w, z_w = map(find_window_both,
                                [ndarr[:, 0], ndarr[:, 1], ndarr[:, 2]])
            # sample numbers, non-gravity axes
            x_png = sorted(np.concatenate((y_w['p'], z_w['p'])))
            x_nng = sorted(np.concatenate((y_w['n'], z_w['n'])))
            y_png = sorted(np.concatenate((x_w['p'], z_w['p'])))
            y_nng = sorted(np.concatenate((x_w['n'], z_w['n'])))
            z_png = sorted(np.concatenate((x_w['p'], y_w['p'])))
            z_nng = sorted(np.concatenate((x_w['n'], y_w['n'])))
            x_non_g = sorted(np.concatenate((x_png, x_nng)))
            y_non_g = sorted(np.concatenate((y_png, y_nng)))
            z_non_g = sorted(np.concatenate((z_png, z_nng)))

            offset = list(map(np.mean,
                              [ndarr[x_non_g, 0],
                               ndarr[y_non_g, 1],
                               ndarr[z_non_g, 2]]))
            # measured g values, without adjusting for misalignment
            gs_orig = list(map(np.mean,
                               [ndarr[x_w['n'], 0],
                                ndarr[x_w['p'], 0],
                                ndarr[y_w['n'], 1],
                                ndarr[y_w['p'], 1],
                                ndarr[z_w['n'], 2],
                                ndarr[z_w['p'], 2]]))
            # misalignment removed
            arr2 = ndarr - np.array(offset)

            if absolute:
                xm, ym, zm = map(lambda x: np.mean(abs(x)),\
                        [arr2[x_png, 0], arr2[y_png, 1], arr2[z_png, 2]])
            else:
                xm, ym, zm = map(np.mean,\
                        [arr2[x_png, 0], arr2[y_png, 1], arr2[z_png, 2]])

            gs = list(map(np.mean,
                          [arr2[x_w['n'], 0],
                           arr2[x_w['p'], 0],
                           arr2[y_w['n'], 1],
                           arr2[y_w['p'], 1],
                           arr2[z_w['n'], 2],
                           arr2[z_w['p'], 2]]))
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

        calib1_vals = get_gs(tempaccmat)

        if 'calib2' in kwargs:
            f2 = open(kwargs.get('calib2'))
            data2 = json.load(f2)
            tempaccmat2 = unbuffer(data2['data'], 'ArrayAcc')
            calib2_vals = get_gs(tempaccmat2)

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
            self.offset = calib1_vals['offset']
            self.misalign = {'L': calib1_vals['misalign']}
            self.original = tempaccmat
            self.processed = calib1_vals['processed']
            self.raw_gs = calib1_vals['gs_orig']
            self.off_gs = calib1_vals['gs']
            self.boost_gs = calib1_vals['gs_boost']
            self.samp_num = calib1_vals['samp_num']
