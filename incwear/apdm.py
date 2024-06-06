from datetime import timedelta
import numpy as np
import h5py
from base import (BaseProcess, CalibProcess, make_start_end_datetime,
                  cycle_filt, time_asleep, rate_calc, get_axis_offsets_v2)


class SensorMissingError(Exception):
    """ If a sensor recording is missing, raise an error """
    pass

class OpalV2(BaseProcess):
    """
    A class that will store (preliminarily) processed data
        recorded in the OPAL V2sensors
    """
    def __init__(self, filename, label_r=None, in_en_dts=None,
                 offset=None, gs=None, thr_method='orig', **kwargs):
        """
        Parameters
        ----------
        filename: str
            *.h5 data file's name
        label_r: str | None
            string that identifies the "r"ight side
            If None, try all sensor keys and randomly choose
            left or right
        in_en_dts: list | None
            list of datetime objects to trim the full recording
            If None, then the entire recording will be used for
            further processing
        offset: dict(list)
            a dictionary whose items are left and right
            sensors' axis misalignment error
        gs: list
            a list (or list of lists) of gain errors
        thr_method: str
            'orig'inal or 'new'
        **kwargs
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
        # Update fs!
        if 'fs' in kwargs:
            self.info.fs = kwargs.get('fs')
        else:
            self.info.fs = 20       # default: 20Hz

        with h5py.File(filename, 'r') as h5file:
            # Sensors has TWO members.
            # Each subgroup has a name with the format: XI-XXXXXX
            # The number after the dash is the Case ID
            sensors = h5file['Sensors']
            # This should always be of length 2
            sids = list(sensors.keys())
            if len(sids) == 1:
                raise SensorMissingError("Only one sensor recording exists")
            if len(sids) == 0:
                raise SensorMissingError("No sensor recording exists")

            # Usually there are 2 sensors in a .h5 file
            if len(sids) == 2:
                # We need to find out which sensor was attached to which leg
                # First, we read the label of the second Case ID (sensorlabel)
                sensorlabel = sensors[sids[1]]\
                        ['Configuration'].attrs["Label 0"].decode()
                # Second, we compare sensorlabel with the user provided label
                # for the "right" (ex. 'right', 'R', 'Right_leg', 'derecho'...)
                # If the match is True, then ridx = 1, or the second Case ID.
                # If the match if False, then ridx = 0, the first Case ID.
                if label_r is not None:
                    ridx = label_r.lower() in sensorlabel.lower()
                else:
                    print("No right leg identifier string provided.")
                    ridx = 0
                sensordict = {'L': sensors[sids[not ridx]],
                              'R': sensors[sids[ridx]]}
            # This is for a testing purpose, and not a typically expected case.
            # The case of FOUR sensor data in one h5 file
            else:
                # vibrantly moved sensors should display greater sd values
                # non-moving sensors are assumed to be mostly silent.
                gvars = {sid: np.std(sensors[sid]['Accelerometer'][:, 2])
                         for sid in sids}
                # lsid: a tuple - (sensorid: sd of measured g values)
                # sort `gvar` in a descending order, and take the first two
                lsid, rsid = sorted(gvars.items(),
                                    key=lambda x: x[1],
                                    reverse=True)[0:2]
                sensordict = {'L': sensors[sids[sids.index(lsid[0])]],
                              'R': sensors[sids[sids.index(rsid[0])]]}

            # Index with the recording start time(in) and the end time(en)
            # Note that in_en_dts are given in UTC as well
            rowidx = self._prep_row_idx(sensordict['L'], in_en_dts)

            # Remove offset
            if offset is not None:
                lagn = sensordict['L']['Accelerometer'] - np.array(offset['L'])
                ragn = sensordict['R']['Accelerometer'] - np.array(offset['R'])
            else:
                l_aligned = sensordict['L']['Accelerometer']
                r_aligned = sensordict['R']['Accelerometer']

            # Correct the gain error
            if gs is not None:
                offset_rm_l = self.correct_gain(lagn,
                                                get_axis_offsets_v2(gs["L"]))
                offset_rm_r = self.correct_gain(ragn,
                                                get_axis_offsets_v2(gs["R"]))
                # If Gain/Offset corrected, 9.80665 should be multiplied again
                offset_rm_l = 9.80665*offset_rm_l
                offset_rm_r = 9.80665*offset_rm_r
            else:
                offset_rm_l = l_aligned
                offset_rm_r = r_aligned

            # Accelerometer and Gyrocscope norms / this part takes some time
            accmags = self._get_mag(
                    {x: sensordict[x]['Accelerometer'] for x in ['L', 'R']},
                    rowidx)
            vmags = self._get_mag(
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
            if thr_method == 'new':
                thresholds = self._get_ind_acc_threshold2(accmags_f)
            else:
                thresholds = self._get_ind_acc_threshold(accmags_f)

            if rowidx is not None:
                rts = [self._calc_datetime(sensors[sids[1]]['Time'][0]) +
                       timedelta(seconds=rowidx[0]*0.05),
                       self._calc_datetime(sensors[sids[1]]['Time'][0]) +
                       timedelta(seconds=rowidx[-1]*0.05)]
            else:
                rts = list(map(self._calc_datetime,
                               [sensors[sids[1]]['Time'][0],
                                sensors[sids[1]]['Time'][-1]]))

            self.info.fname = [filename]
            self.info.record_times = {'L': rts, 'R': rts}
            self.info.label_r = label_r
            self.info.rowidx = rowidx
            self.info.recordlen = {'L': accmags_f['lmag'].shape[0],
                                   'R': accmags_f['rmag'].shape[0]}

            self.measures.accmags = accmags_f
            self.measures.velmags = vmags
            self.measures.thresholds = thresholds
            self.measures.__post_init__()    # update files

class OpalV1(BaseProcess):
    """
    A class that will store (preliminarily) processed data
        recorded in a pair of Opal V1 sensors
    This may need more work (7/12/2023)
    """
    def __init__(self, filename, in_en_dts, btn_ignore=False, **kwargs):
        """
        **kwargs:
        lsensorlist: list
            a list of sensor id's used to measure left leg movement
        """
        super().__init__(filename=filename, in_en_dts=in_en_dts)

        # Update fs!
        if 'fs' in kwargs:
            self.info.fs = kwargs.get('fs')
        else:
            self.info.fs = 20       # default: 20Hz

        with h5py.File(filename, 'r') as sensors:
            # Opal V1 does not have 'R' or 'L' label...
            # Sensors have two members
            # Each subgroup has a nmae with the format: XI-XXXXXX
            sids = list(sensors.keys())[1:3]
            if len(sids) == 1:
                raise SensorMissingError("Only one sensor recording exists")
            if len(sids) == 0:
                raise SensorMissingError("No sensor recording exists")

            # Determine based on the log: which sensor ids were used to measure
            # left leg movements?
            if 'lsensorlist' in kwargs:
                is_l = sids[1] in kwargs.get('lsensorlist')
            # If no log is provided, the first id corresponds to the left
            elif len(sensors.attrs['MonitorLabelList']) > 0:
                first_label = sensors.attrs['MonitorLabelList'][0].decode()
                print(f'First label found: {first_label}')
                if 'left' in first_label.lower().split():
                    print('The label has the keyword - left')
                    is_l = 0
                else:
                    is_l = 1
            else:
                is_l = False
            print('LEFT SENSOR ID: ', sids[is_l])
            sensordict = {'L': sensors[sids[is_l]],
                          'R': sensors[sids[~is_l]]}
            # We don't know which of the two sensors' button was pressed...
            # ... but skip it if a user wants to ignore and just want to see the full data
            if btn_ignore:
                rowidx = self._prep_row_idx(sensordict['L']['Time'],
                                            in_en_dts)
                indexed1 = 0
            else:
                try:
                    indexed1 = np.where(sensordict['L']['ButtonStatus'][:]==1)[0][0]
                    print('Left Button Pressed: ', indexed1)
                    rowidx = self._prep_row_idx(sensordict['L']['Time'],
                                                in_en_dts,
                                                btnstatus=indexed1)
                except:
                    try:
                        indexed1 = np.where(sensordict['R']['ButtonStatus'][:]==1)[0][0]
                        print('Right Button Pressed: ', indexed1)
                    except:
                        indexed1 = 0
                        print('No Button Press ', indexed1)
                    rowidx = self._prep_row_idx(sensordict['R']['Time'],
                                                in_en_dts,
                                                btnstatus=indexed1)

                print('Range of the data points: ', rowidx[0], rowidx[-1])

            accmags = self._get_mag(
                    {x: sensordict[x]['Calibrated']['Accelerometers']
                     for x in ['L', 'R']}, rowidx)
            velmags = self._get_mag(
                    {x: sensordict[x]['Calibrated']['Gyroscopes']
                     for x in ['L', 'R']}, rowidx)

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
                rts = [self._calc_datetime(sensors[sids[1]]['Time'][0]) +
                       timedelta(seconds=rowidx[0]*0.05),
                       self._calc_datetime(sensors[sids[1]]['Time'][0]) +
                       timedelta(seconds=rowidx[-1]*0.05)]
            else:
                # make use of the button press information
                rts = list(map(self._calc_datetime,
                               [sensors[sids[1]]['Time'][indexed1],
                                sensors[sids[1]]['Time'][-1]]))

            self.info.fname = [filename]
            self.info.record_times = {'L': rts, 'R': rts}
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
    def __init__(self, filename, in_en_dts, offset=None, gs=None, **kwargs):
        super().__init__(filename=filename, in_en_dts=in_en_dts)

        if 'fs' in kwargs:
            self.info.fs = kwargs.get('fs')
        else:
            self.info.fs = 20       # default: 20Hz

        with h5py.File(filename, 'r') as h5file:
            sensors = h5file['Sensors']
            if 'sid' in kwargs:
                sids = kwargs.get('sid')
            else:
                sids = list(sensors.keys())[0]

            sensordict = {'L': sensors[sids]}
            rowidx = self._prep_row_idx(sensordict['L'], in_en_dts)

            if offset is not None:
                l_aligned = sensordict['L']['Accelerometer'] - np.array(offset)
            else:
                l_aligned = sensordict['L']['Accelerometer']

            if gs is not None:
                offset_rm = self.correct_gain(l_aligned,
                                              get_axis_offsets_v2(gs))
                # If Gain/Offset corrected, 9.80665 should be multiplied again
                offset_rm = 9.80665*offset_rm
            else:
                offset_rm = l_aligned

            if 'filter' in kwargs:
                try:
                    filt_x = self.low_pass(kwargs.get('fc'),
                                           kwargs.get('fs'),
                                           offset_rm[:, 0])
                    filt_y = self.low_pass(kwargs.get('fc'),
                                           kwargs.get('fs'),
                                           offset_rm[:, 1])
                    filt_z = self.low_pass(kwargs.get('fc'),
                                           kwargs.get('fs'),
                                           offset_rm[:, 2])
                    self.filtered = np.column_stack((filt_x, filt_y, filt_z))
                    accmags_f = self._get_mag({'L': self.filtered})
                except:
                    raise ValueError("fc and fs should be provided if you're filtering the data")
            else:
                accmags_f = self._get_mag({'L': offset_rm}, rowidx)
            vmags = self._get_mag({'L': sensordict['L']['Gyroscope']}, rowidx)

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
                rts = [self._calc_datetime(sensors[sids]['Time'][0]) +
                       timedelta(seconds=rowidx[0]*0.05),
                       self._calc_datetime(sensors[sids]['Time'][0]) +
                       timedelta(seconds=rowidx[-1]*0.05)]
            else:
                rts = list(map(self._calc_datetime,
                               [sensors[sids]['Time'][0],
                                sensors[sids]['Time'][-1]]))

            self.info.fname = filename
            self.info.record_times = {'U': rts}
            self.info.label_r = 'N/A'
            self.info.rowidx = rowidx
            self.info.recordlen = {'U': accmags_f['umag'].shape[0]}
            self.measures.accmags = accmags_f
            self.measures.velmags = vmags
            self.measures.thresholds = thresholds
            self.measures.__post_init__()


class OpalV2Calib(CalibProcess):
    def __init__(self, calib1, g_thresholds=[9.7, 9.9], **kwargs):
        super().__init__()   # absolute, winlen, stdcut inherited
        sensors = h5py.File(calib1)['Sensors']
        id_list = list(sensors.keys())  # length is 1 or 2
        # If 'sid' is provided, handle only that one.
        # If provided sid is not in id_list, then throw an error.
        errmsg = "sid not in the data. Please check your argument again."
        if 'sid' in kwargs:
            temp = kwargs.get('sid')
            if temp not in id_list:
                raise ValueError(errmsg)
            id_list = [kwargs.get('sid')]
        self.g_thresholds = g_thresholds
        self.info.fs = 20
        ss_l = sensors[id_list[0]]['Accelerometer']

        calib1_vals = self.get_gs(ss_l)
        # Opal sensor has two files in one sensor recording...
        self.info.fname = [calib1]

        # If 2 sensors...
        if len(id_list) == 2:
            ss_r = sensors[id_list[1]]['Accelerometer']
            calib2_vals = self.get_gs(ss_r)

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
