from datetime import datetime, timezone, timedelta
import numpy as np
import h5py
from base import BaseProcess, make_start_end_datetime, cycle_filt, time_asleep

class SensorMissingError(Exception):
    """ If a sensor recording is missing, raise an error """
    pass

class OpalV2(BaseProcess):
    """
    A class that will store (preliminarily) processed data
        recorded in the OPAL V2sensors
    """
    def __init__(self, filename, in_en_dts, label_r):
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
                    {x:sensordict[x]['Accelerometer'] for x in ['L', 'R']},
                    rowidx)
            velmags = self._get_mag(
                    {x:sensordict[x]['Gyroscope'] for x in ['L', 'R']}, 
                    rowidx)

            # Storing the acceleration threshold values:
            #   [laccth, lnaccth, raccth, rnaccth]
            thresholds = self._get_ind_acc_threshold(accmags)

            if rowidx is not None:
                rts = [self._calc_datetime(sensors[sids[1]]['Time'][0]) +\
                        timedelta(seconds=rowidx[0]*0.05), in_en_dts[1]]
            else:
                rts = list(map(self._calc_datetime,
                    [sensors[sids[1]]['Time'][0],
                        sensors[sids[1]]['Time'][-1]]))

            self.info.fname = [filename]
            self.info.record_times = {'L': rts, 'R': rts}
            self.info.fs = 20
            self.info.label_r = label_r
            self.info.rowidx = rowidx
            self.info.recordlen= {'L': accmags['lmag'].shape[0],
                    'R': accmags['rmag'].shape[0]}

            self.measures.accmags = accmags
            self.measures.velmags = velmags
            self.measures.thresholds = thresholds
            self.measures.__potst_init__() # update files

class OpalV1(BaseProcess):
    """
    A class that will store (preliminarily) processed data
        recorded in a pair of Opal V1 sensors
    """
    def __init__(self, filename, in_en_dts):
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

            # This should be checked (4/3/23)
            sensordict = {'L': sensors[sids[0]],
                    'R': sensors[sids[1]]}
            rowidx = self._prep_row_idx(sensordict['L'], in_en_dts)

            accmags = self._get_mag(
                    {x:sensordict[x]['Calibrated']['Accelerometers'] for x in ['L', 'R']}, rowidx)
            velmags = self._get_mag(
                    {x:sensordict[x]['Calibrated']['Gyroscopes'] for x in ['L', 'R']}, rowidx)

            thresholds = self._get_ind_acc_threshold(accmags)

            if rowidx is not None:
                rts = [self._calc_datetime(sensors[sids[1]]['Time'][0])+\
                        timedelta(seconds=rowidx[0]*0.05), in_en_dts[1]]
            else:
                rts = list(map(self._calc_datetime,
                    [sensors[sids[1]]['Time'][0],
                        sensors[sids[1]]['Time'][-1]]))

            self.info.fname = [filename]
            self.info.record_times = {'L': rts, 'R': rts}
            self.info.fs = 20
            self.info.label_r = 'N/A'
            self.info.rowidx = rowidx
            self.info.recordlen = {'L': accmags['lmag'].shape[0],
                    'R': accmags['rmag'].shape[0]}
            self.measures.accmags = accmags
            self.measures.velmags = velmags
            self.measures.thresholds = thresholds
            self.measures.__post__init__()

class OpalV2Single(BaseProcess):
    """ Single Opal V2 Sensor """
    def __init__(self, filename, in_en_dts):
        super().__init__(filename = filename,
                in_en_dts = in_en_dts)

        with h5py.File(filename, 'r') as h5file:
            sensors = h5file['Sensors']
            sids = list(sensors.keys())[0]

            sensordict = {'L': sensors[sids],
                    'R': sensors[sids]}
            rowidx = self._prep_row_idx(sensordict['L'], in_en_dts)

            accmags = self._get_mag(
                    {x:sensordict[x]['Accelerometer'] for x in ['L', 'R']},
                    rowidx)
            velmags = self._get_mag(
                    {x:sensordict[x]['Gyroscope'] for x in ['L', 'R']},
                    rowidx)
            thresholds = self._get_ind_acc_threshold(accmags)

            if rowidx is not None:
                rts = [self._calc_datetime(sensors[sids]['Time'][0])+\
                        timedelta(seconds=rowidx[0]*0.05), in_en_dts[1]]
            else:
                rts = list(map(self._calc_datetime,
                    [sensors[sids]['Time'][0],
                        sensors[sids['Time'][-1]]]))

            self.info.fname = [filename]
            self.info.record_times = {'L': rts, 'R': rts}
            self.info.fs = 20
            self.info.label_r = 'N/A'
            self.info.rowidx = rowidx
            self.info.recordlen = {'L': accmags['lmag'].shape[0],
                    'R': accmags['rmag'].shape[0]}
            self.measures.accmags = accmags
            self.measures.velmags = velmags
            self.measures.thresholds = thresholds
            self.measures.__post_init__()

