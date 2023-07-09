from datetime import timedelta
import numpy as np
import h5py
from base import BaseProcess, make_start_end_datetime,\
        cycle_filt, time_asleep, rate_calc, get_axis_offsets_v2

class SensorMissingError(Exception):
    """ If a sensor recording is missing, raise an error """
    pass

class OpalV2(BaseProcess):
    """
    A class that will store (preliminarily) processed data
        recorded in the OPAL V2sensors
    """
    def __init__(self, filename, in_en_dts, label_r, **kwargs):
        """
        Parameters:
            filename: str
                *.h5 data file's name
            in_en_dts: list
                list of datetime objects
            label_r: str
                string that identifies the "r"ight side
            **kwargs
            misalign: dict(list)
                a dictionary whose items are left and right
                sensors' axis misalignment error
            fs: int
                sampling frequency
            filter: boolean
                apply filter (True) or not (False)
            fc: int
                cut-off frequency for low-pass filtering
        """
        super().__init__(filename=filename,
                in_en_dts=in_en_dts,
                label_r=label_r)

        with h5py.File(filename, 'r') as h5file:
            # Sensors has TWO members.
            # Each subgroup has a name with the format: XI-XXXXXX
            # The number after the dash is the Case ID
            sensors = h5file['Sensors']
            # This should always be of length 2
            sids = list(sensors.keys())
            if len(sids) == 1:
                raise SensorMissingError("Only one sensor recording exists")
            elif len(sids) == 0:
                raise SensorMissingError("No sensor recording exists")

            # We need to find out which sensor was attached to which leg.
            # First, we read the label of the second Case ID (sensorlabel)
            sensorlabel = sensors[sids[1]]\
                    ['Configuration'].attrs["Label 0"].decode()
            # Second, we compare sensorlabel with the user provided label
            # for the "right" (ex. 'right', 'R', 'Right_leg', 'derecho'...)
            # If the match is True, then ridx = 1, or the second Case ID.
            # If the match if False, then ridx = 0, the first Case ID.
            ridx = label_r.lower() in sensorlabel.lower()
            sensordict = {'L': sensors[sids[not ridx]],
                          'R': sensors[sids[ridx]]}

            # Index with the recording start time(in) and the end time(en)
            # Note that in_en_dts are given in UTC as well
            rowidx = self._prep_row_idx(sensordict['L'], in_en_dts)

            # Accelerometer and Gyrocscope norms / this part takes some time
            accmags = self._get_mag(
                    {x: sensordict[x]['Accelerometer'] for x in ['L', 'R']},
                    rowidx)
            velmags = self._get_mag(
                    {x: sensordict[x]['Gyroscope'] for x in ['L', 'R']},
                    rowidx)
            
            if 'filter' in kwargs:
                try:
                    accmags_f = {
                            'lmag': self.low_pass(kwargs.get('fc'),
                                kwargs.get('fs'),
                                accmags['lmag']),
                            'rmag': self.low_pass(kwargs.get('fc'),
                                kwargs.get('fs'),
                                accmags['rmag'])}
                except:
                    raise ValueError("fc and fs should be provided if you're filtering the data")
            else:
                accmags_f = accmags

            # Storing the acceleration threshold values:
            #   [laccth, lnaccth, raccth, rnaccth]
            thresholds = self._get_ind_acc_threshold(accmags_f)

            if rowidx is not None:
                rts = [self._calc_datetime(sensors[sids[1]]['Time'][0]) +\
                        timedelta(seconds=rowidx[0]*0.05),
                        self._calc_datetime(sensors[sids[1]]['Time'][0])+\
                                timedelta(seconds=rowidx[-1]*0.05)]
            else:
                rts = list(map(self._calc_datetime,
                    [sensors[sids[1]]['Time'][0],
                        sensors[sids[1]]['Time'][-1]]))

            self.info.fname = [filename]
            self.info.record_times = {'L': rts, 'R': rts}
            if 'fs' in kwargs:
                self.info.fs = kwargs.get('fs')
            else:
                self.info.fs = 20   # default: 20 S/s
            self.info.label_r = label_r
            self.info.rowidx = rowidx
            self.info.recordlen= {'L': accmags_f['lmag'].shape[0],
                    'R': accmags_f['rmag'].shape[0]}

            self.measures.accmags = accmags_f
            self.measures.velmags = velmags
            self.measures.thresholds = thresholds
            self.measures.__post_init__() # update files

class OpalV1(BaseProcess):
    """
    A class that will store (preliminarily) processed data
        recorded in a pair of Opal V1 sensors
    """
    def __init__(self, filename, in_en_dts, **kwargs):
        """
        **kwargs:
        lsensorlist: list
            a list of sensor id's used to measure left leg movement
        """
        super().__init__(filename = filename,
                in_en_dts = in_en_dts)

        with h5py.File(filename, 'r') as sensors:
            # Opal V1 does not have 'R' or 'L' label...
            # Sensors have two members
            # Each subgroup has a nmae with the format: XI-XXXXXX
            sids = list(sensors.keys())[1:3]
            if len(sids) == 1:
                raise SensorMissingError("Only one sensor recording exists")
            elif len(sids) == 0:
                raise SensorMissingError("No sensor recording exists")

            # Determine based on the log: which sensor ids were used to measure
            # left leg movements?
            if 'lsensorlist' in kwargs:
                is_l = sids[1] in kwargs.get('lsensorlist')
            # If no log is provided, the first id corresponds to the left
            else:
                is_l = False
            print('LEFT SENSOR ID: ', sids[is_l])
            sensordict = {'L': sensors[sids[is_l]],
                    'R': sensors[sids[~is_l]]}
            # We don't know which of the two sensors' button was pressed...
            try:
                indexed1 = np.where(sensordict['L']['ButtonStatus'][:]==1)[0][0]
                print('Left Button Pressed: ', indexed1)
                rowidx = self._prep_row_idx(sensordict['L'], in_en_dts)
            except:
                try:
                    indexed1 = np.where(sensordict['R']['ButtonStatus'][:]==1)[0][0]
                    print('Right Button Pressed: ', indexed1)
                except:
                    indexed1 = [0][0]
                    print('No Button Press ', indexed1)
                rowidx = self._prep_row_idx(sensordict['R'], in_en_dts)

            print('Range of the data points: ', rowidx[0], rowidx[-1])

            accmags = self._get_mag(
                    {x:sensordict[x]['Calibrated']['Accelerometers'] for x in ['L', 'R']}, rowidx)
            velmags = self._get_mag(
                    {x:sensordict[x]['Calibrated']['Gyroscopes'] for x in ['L', 'R']}, rowidx)

            if 'filter' in kwargs:
                try:
                    accmags_f = {
                            'lmag': self.low_pass(kwargs.get('fc'),
                                kwargs.get('fs'),
                                accmags['lmag']),
                            'rmag': self.low_pass(kwargs.get('fc'),
                                kwargs.get('fs'),
                                accmags['rmag'])}
                except:
                    raise ValueError("fc and fs should be provided if you're filtering the data")
            else:
                accmags_f = accmags

            thresholds = self._get_ind_acc_threshold(accmags_f)

            if rowidx is not None:
                rts = [self._calc_datetime(sensors[sids[1]]['Time'][0])+\
                        timedelta(seconds=rowidx[0]*0.05),
                        self._calc_datetime(sensors[sids[1]]['Time'][0])+\
                                timedelta(seconds=rowidx[-1]*0.05)]
            else:
                # make use of the button press information
                rts = list(map(self._calc_datetime,
                    [sensors[sids[1]]['Time'][indexed1],
                        sensors[sids[1]]['Time'][-1]]))

            self.info.fname = [filename]
            self.info.record_times = {'L': rts, 'R': rts}
            self.info.fs = 20
            self.info.label_r = 'N/A'
            self.info.rowidx = rowidx
            self.info.recordlen = {'L': accmags_f['lmag'].shape[0],
                    'R': accmags_f['rmag'].shape[0]}
            self.measures.accmags = accmags_f
            self.measures.velmags = velmags
            self.measures.thresholds = thresholds
            self.measures.__post_init__()

class OpalV2Single(BaseProcess):
    """ Single Opal V2 Sensor """
    def __init__(self, filename, in_en_dts, ground_gs=None, **kwargs):
        super().__init__(filename=filename, in_en_dts=in_en_dts)

        with h5py.File(filename, 'r') as h5file:
            sensors = h5file['Sensors']
            sids = list(sensors.keys())[0]

            #sensordict = {'L': sensors[sids],
            #        'R': sensors[sids]}
            sensordict = {'L': sensors[sids]}
            rowidx = self._prep_row_idx(sensordict['L'], in_en_dts)

            #accmags = self._get_mag(
            #        {x:sensordict[x]['Accelerometer'] for x in ['L', 'R']},
            #        rowidx)
            #velmags = self._get_mag(
            #        {x:sensordict[x]['Gyroscope'] for x in ['L', 'R']},
            #        rowidx)
            if 'offset' in kwargs:
                l_aligned = sensordict['L']['Accelerometer'] -\
                        np.array(kwargs.get('offset'))
            else:
                l_aligned = sensordict['L']['Accelerometer']

            if ground_gs is not None:
                offset_rm = self.remove_offset(l_aligned,
                                               get_axis_offsets_v2(ground_gs))
                # If Gain/Offset corrected, 9.80665 should be multiplied again
                offset_rm = 9.80665*offset_rm
            else:
                offset_rm = l_aligned

            if 'filter' in kwargs:
                try:
                    filt_x = self.low_pass(kwargs.get('fc'),
                                           kwargs.get('fs'),
                                           offset_rm[:,0])
                    filt_y = self.low_pass(kwargs.get('fc'),
                                           kwargs.get('fs'),
                                           offset_rm[:,1])
                    filt_z = self.low_pass(kwargs.get('fc'),
                                           kwargs.get('fs'),
                                           offset_rm[:,2])
                    self.filtered = np.column_stack((filt_x, filt_y, filt_z))
                    accmags_f = self._get_mag({'L': self.filtered})
                except:
                    raise ValueError("fc and fs should be provided if you're filtering the data")
            else:
                accmags_f = self._get_mag({'L': offset_rm})
            velmags = self._get_mag(
                    {'L': sensordict['L']['Gyroscope']})

            """
            accmags = self._get_mag({'L': offset_rm})
            velmags = self._get_mag(
                    {'L': sensordict['L']['Gyroscope']})

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
            """
            thresholds = self._get_ind_acc_threshold(accmags_f)

            if rowidx is not None:
                rts = [self._calc_datetime(sensors[sids]['Time'][0])+\
                        timedelta(seconds=rowidx[0]*0.05),
                        self._calc_datetime(sensors[sids]['Time'][0])+\
                                timedelta(seconds=rowidx[-1]*0.05)]
            else:
                rts = list(map(self._calc_datetime,
                    [sensors[sids]['Time'][0],
                        sensors[sids]['Time'][-1]]))

            self.info.fname = filename
            self.info.record_times = {'U': rts}
            if 'fs' in kwargs:
                self.info.fs = kwargs.get('fs')
            else:
                self.info.fs = 20   # default: 20 S/s
            self.info.label_r = 'N/A'
            self.info.rowidx = rowidx
            self.info.recordlen = {'U': accmags_f['umag'].shape[0]}
            self.measures.accmags = accmags_f
            self.measures.velmags = velmags
            self.measures.thresholds = thresholds
            self.measures.__post_init__()

class OpalV2Calib(BaseProcess):
    def __init__(self, calib1, absolute=False, **kwargs):
        super().__init__()
        sensors = h5py.File(calib1)['Sensors']
        id_list = list(sensors.keys())  # length is 1 or 2
        ss_l = sensors[id_list[0]]['Accelerometer']
        self.absolute = absolute

        def find_window_both(arr):
            """ This is a wrapper to use find_window on a dictionary """
            sns = find_sns(arr)
            return dict(zip(['p', 'n'],
                            map(find_window,
                                [arr, arr],
                                list(sns.values()))))

        def find_window(arr, sampnums):
            """
            A function to find sample numbers of THE window (7 seconds)
            The window should have adjacent points, and the std of the points
            should be no greater than 0.02.
            """
            ia = 0
            ib = 140
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
            idp = np.where(np.logical_and(arr < 9.9, arr > 9.7))[0]
            idn = np.where(np.logical_and(arr < -9.7, arr > -9.9))[0]
            return {'p': idp, 'n': idn}

        def get_gs(ndarr, absolute=self.absolute):
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

            offset = list(map(np.mean,
                              [ss_l[x_non_g, 0],
                               ss_l[y_non_g, 1],
                               ss_l[z_non_g, 2]]))
            # measured g values, without adjusting for misalignment
            gs_orig = list(map(np.mean,
                               [ss_l[x_w['n'], 0],
                                ss_l[x_w['p'], 0],
                                ss_l[y_w['n'], 1],
                                ss_l[y_w['p'], 1],
                                ss_l[z_w['n'], 2],
                                ss_l[z_w['p'], 2]]))
            # offset removed
            arr2 = ss_l - np.array(offset)
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
            # misalignment reflected
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

        # If 2 sensors...
        if len(id_list) == 2:
            ss_r = sensors[id_list[1]]['Accelerometer']
            calib2_vals = get_gs(ss_r)

            self.offset = {'L': calib1_vals['offset'],
                           'R': calib2_vals['offset']}
            self.misalign = {'L': calib1_vals['misalign'],
                             'R': calib2_vals['misalign']}
            self.processed = {'L': calib1_vals['processed'],
                              'R': calib2_vals['processed']}
            self.original = {'L': ss_l, 'R': ss_r}
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
            self.misalign = {'L': calib1_vals['misalign'],
                             'IDS': {'L': id_list[0]}}
            self.original = ss_l
            self.processed = calib1_vals['processed']
            self.raw_gs = calib1_vals['gs_orig']
            self.off_gs = calib1_vals['gs']
            self.boost_gs = calib1_vals['gs_boost']
            self.samp_num = calib1_vals['samp_num']
