'''
HDF stands for [Hierarchical Data Format]. Every object in an HDF5 file has a name,
and they're arranged in a POSIX-style hierarchy with / separators
'''

import h5py
import numpy as np
import pandas as pd
from functools import partial
import matplotlib.pyplot as plt
import time
from scipy.signal import find_peaks, detrend
from datetime import datetime, timedelta

'''
Written by Jinseok Oh, Ph.D.
2022/9/13

ReadSensorLog.py is a python script porting MakeDataStructure_v2.m,
a file prepared to extract data from APDM OPAL V2 sensors.

A data file in h5 format would have three Level-1 names:
    - Annotations
    - Processed
    - Sensors

Original script exports the following values:
    - x [N x 9, Acc, Gyro, Mag each three columns; N = total number of samples]
    - iButtonPress [NULL]
    - bButton [Time from the 'Annotations' of a sensor]
    - sampleRate [sampling rate, getting from the attributes of 'Sensors']
    - dateNumbers [date of data collection]
    - label [read from 'Configuration' of 'Sensors']:
        'Pie izquierdo' is left foot (L) / 'pie derecho 2' is right foot (R)
    - caseId [NULL]
    - nSamples [NULL]
    - q [Orientation data from 'Processed']
    - id ['Sensors' - 'Name']
    - iButtonPressed ['Annotations']

Class Subject will try to have all those values, particularly those used in the actual analysis.
However, not in the exact format. For example, if 'Annotations' is saved as an attribute: 'annots',
then a user can access its sub-attribute, such as caseID [well but it's not used anyways...]
'''
filename = './MATLAB/h5files/20200217-082740_106v1.h5'
class Subject():

    def __init__(self, filename):
        # Read the HDF5 file
        f = h5py.File(filename, 'r')

        # Save the file name
        self.filename = filename

        # 'Annotations' comprises three columns: Time, Case ID, annotation in spanish
        # Time in epoch microseconds... what does this mean?
        self.annots = f['Annotations']
        
        # 'Sensors' has TWO members.
        # Each subgroup will have a name in the format of XI-XXXXXX. The number after the dash is the Case ID.
        # This study used two sensors during a recording, so there will be two subgrops.
        self.sensors = f['Sensors']
        self.sensorids = list(self.sensors.keys())

        # To determine which sensor was attached to the left foot
        leftidx = 0
        whichfoot = self.sensors[self.sensorids[leftidx]]['Configuration']['Config Strings'][0][2].decode()
        if 'derecho' in whichfoot:
            rightidx = 0
            leftidx = 1
        else:
            rightidx = 1

        # Acc, Gyro, Magnetometer data stored.
        # Example:
        # test = Subject('Users/joh/Downloads/xxx.h5')
        # test.left.Accelerometer
        self.left = self.sensors[self.sensorids[leftidx]]
        self.right = self.sensors[self.sensorids[rightidx]]

        # How many datapoints?
        self.N = self.left['Accelerometer'].shape[0]

        # 1 data point every 50000μs or 0.05s
        self.T = 0.05
        # sfreq will be equal to Sample rate.
        self.sfreq = int(1/self.T)

        # Recorded timepoints - may not use, but just in case
        self.rtime = map(self.calc_datetime, self.left['Time'])

        # 'Processed' - the orientation in quarternion
        self.procs = f['Processed']
        self.ori_left = self.procs[self.sensorids[leftidx]]['Orientation']
        self.ori_right = self.procs[self.sensorids[rightidx]]['Orientation']

        # Time in seconds unit - starts from 0 and the step size is 0.05s
        # This may be used later for plotting
        self.time = (self.left['Time'] - self.left['Time'][0])/1e6

    # This function will calculate the actual date and time from the time recorded in the sensor
    def calc_datetime(self, x):
        tOffset = -6                            # Guatemala: UTC-6 (No Daylight Savings Time)
        div = 24*3600*1e6
        div2 = 3600*1e6
        d_days = x // div                       # How many days have passed since (1970, 1, 1, UTC)
        rem1 = x % div                          # Remainder in microseconds
        d_hrs = (rem1 // div2) + tOffset        # Hours
        rem2 = rem1 % div2
        d_mins = rem2 // (60*1e6)               # Minutes
        rem3 = rem2 % (60*1e6)
        d_secs = rem3 // 1e6                    # Seconds

        # So whatever value from the sensor is the time passed from Jan 1, 1970 UTC in microseconds
        # This function returns the datetime up to seconds
        record_start_time = datetime(1970, 1, 1, 0, 0, 0) + timedelta(days = d_days, seconds = d_secs,\
                minutes = d_mins, hours = d_hrs, microseconds = rem3 % 1e6)
        return (record_start_time)

    # dtype argument should be either 'Accelerometer' or 'Gyroscope'
    # row_idx should be a list of two indices: user-specified array start and end
    # det_option: how are you going to detrend the data?
    def _get_mag(self, dtype, row_idx=None, det_option='median'):
        if row_idx is None:
            row_idx = [x for x in range(self.N)]

        # Use np.linalg.norm... amazing!
        magdata_l = np.linalg.norm(self.left[dtype][row_idx], axis=1)
        magdata_r = np.linalg.norm(self.right[dtype][row_idx], axis=1)
        mags = pd.DataFrame(data = {'lmag':magdata_l, 'rmag':magdata_r})

        # MATLAB's detrend function is not used, so we can consider that the default option
        # for detrending the data is subtracting the median
        if det_option == 'median':
            return(mags - mags.median(axis=0))
        elif det_option == 'customfunc':
            return(mags.apply(detrend, axis=0))
        else:
            print("Your option is either [median] or [customfunc]")

    # winsize in second unit
    def _mov_avg_filt(self, winsize, pdSeries):
        L = int(self.sfreq * winsize)
        return(pdSeries.rolling(L).mean())

    def _local_maxima(self, values, height):
        locmax, maxprop = find_peaks(values, height = height)
        return([locmax, maxprop])

    def _calc_ind_threshold(self, maxprop):
        return(np.mean(maxprop['peak_heights']) - 0.5*np.std(maxprop['peak_heights']))

    # All relevant parameter values coming from Trujillo-Priego et al,, (2017)
    def get_ind_acc_threshold_ivan(self, row_idx, reject_th = [-1.02, 1.32], winsize = 0.5, height=1.0):
        errmsg = "The rejection threshold for the accelerometer data should be a list of \
                a positive and a negative threshold"
        if type(reject_th) is not list:
            print(errmsg)
        elif len(reject_th) != 2:
            print(errmsg)
        elif sum(np.sign(reject_th)) != 0:
            print(errmsg)
        else:
            mags2 = self._get_mag('Accelerometer', row_idx, 'median')
            mags2['rectlmag'] = mags2['lmag'].apply(lambda x: abs(x) if x > max(reject_th) or x < min(reject_th) else 0)
            mags2['rectrmag'] = mags2['rmag'].apply(lambda x: abs(x) if x > max(reject_th) or x < min(reject_th) else 0)

            mags2['avglmag'] = self._mov_avg_filt(winsize, mags2['rectlmag'])
            mags2['avgrmag'] = self._mov_avg_filt(winsize, mags2['rectrmag'])

            self.accmag = mags2

            self.laccth = self._calc_ind_threshold(self._local_maxima(mags2.avglmag.values, height=height)[1])
            self.raccth = self._calc_ind_threshold(self._local_maxima(mags2.avgrmag.values, height=height)[1])

            # Now these values will be used for both positive and negative borders
            return([self.laccth, self.raccth])

    # This is equivalent to the MATLAB script
    def get_ind_acc_threshold(self, row_idx, reject_th = 3.2501, winsize = 0.5, height = 1.0):
        mags = self._get_mag('Accelerometer', row_idx, 'median')
        mags2['lposthcand'] = mags2['lmag'].apply(lambda x: x if x > 0 and x < reject_th else 0)
        mags2['lnegthcand'] = mags2['lmag'].apply(lambda x: x if x < 0 and x > reject_th else 0)
        mags2['rposthcand'] = mags2['rmag'].apply(lambda x: x if x > 0 and x < reject_th else 0)
        mags2['rnegthcand'] = mags2['rmag'].apply(lambda x: x if x < 0 and x > reject_th else 0)

        self.left_pos_peaks = self._local_maxima(mags2.lposthcand.values, height = height)
        self.left_neg_peaks = self._local_maxima(mags2.lnegthcand.values, height = height)
        
    def get_ind_angvel_threshold(self, reject_th = 0.32, winsize = 0.5, height=0.01):
        errmsg = "The rejection threshold for the gyroscope data should be a positive value"
        if reject_th < 0:
            print(errmsg)
        else:
            mags2 = self._get_mag('Gyroscope', 'customfunc')
            mags2['rectlmag'] = mags2['lmag'].apply(lambda x: x if x > reject_th else 0)
            mags2['rectrmag'] = mags2['rmag'].apply(lambda x: x if x > reject_th else 0)

            mags2['avglmag'] = self._mov_avg_filt(winsize, mags2['rectlmag'])
            mags2['avgrmag'] = self._mov_avg_filt(winsize, mags2['rectrmag'])

            self.avmag = mags2

            self.lavth = self._calc_ind_threshold(self._local_maxima(mags2.avglmag.values, height=height)[1])
            self.ravth = self._calc_ind_threshold(self._local_maxima(mags2.avgrmag.values, height=height)[1])

            return([self.lavth, self.ravth])

    # 

