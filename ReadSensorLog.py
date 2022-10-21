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
#filename = './MATLAB/h5files/20200217-082740_106v1.h5'
class Subject():

    def __init__(self, filename, in_en_dts):
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
        self.rtime = map(self._calc_datetime, self.left['Time'])

        # 'Processed' - the orientation in quarternion
        self.procs = f['Processed']
        self.ori_left = self.procs[self.sensorids[leftidx]]['Orientation']
        self.ori_right = self.procs[self.sensorids[rightidx]]['Orientation']

        # Time in seconds unit - starts from 0 and the step size is 0.05s
        # This may be used later for plotting
        self.time = (self.left['Time'] - self.left['Time'][0])/1e6

        # Let's put in the index with the recording start time(in) and the end time(en) compensated
        self.row_idx = self._prep_row_idx(in_en_dts)

    # This function will calculate the actual date and time from the time recorded in the sensor
    def _calc_datetime(self, x):
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

    def _calc_ind_threshold(self, maxprop, ivan=False):
        return(np.mean(maxprop['peak_heights']) - (1-ivan + 0.5*ivan)*np.std(maxprop['peak_heights']))

    # All relevant parameter values coming from Trujillo-Priego et al, (2017)
    # This needs further development for sure...
    def get_ind_acc_threshold_ivan(self, reject_th=[-1.02, 1.32], winsize=0.5, height=1.0):
        errmsg = "The rejection threshold for the accelerometer data should be a list of \
                a positive and a negative threshold"
        if type(reject_th) is not list:
            print(errmsg)
        elif len(reject_th) != 2:
            print(errmsg)
        elif sum(np.sign(reject_th)) != 0:
            print(errmsg)
        else:
            mags2 = self._get_mag('Accelerometer', self.row_idx, 'median')
            mags2['rectlmag'] = mags2['lmag'].apply(lambda x: abs(x) if x > max(reject_th) or x < min(reject_th) else 0)
            mags2['rectrmag'] = mags2['rmag'].apply(lambda x: abs(x) if x > max(reject_th) or x < min(reject_th) else 0)

            mags2['avglmag'] = self._mov_avg_filt(winsize, mags2['rectlmag'])
            mags2['avgrmag'] = self._mov_avg_filt(winsize, mags2['rectrmag'])


            self.laccth = self._calc_ind_threshold(self._local_maxima(mags2.avglmag.values, height=height)[1])
            self.raccth = self._calc_ind_threshold(self._local_maxima(mags2.avgrmag.values, height=height)[1])

            # Now these values will be used for both positive and negative borders
            return([self.laccth, self.raccth])

    # This is equivalent to the MATLAB script
    # You can provide in_en_dts and/or det_option ('median' or 'customfunc')
    def get_ind_acc_threshold(self, det_option='median'):

        reject_th = 3.2501
        height = 1.0
        
        mags2 = self._get_mag('Accelerometer', self.row_idx, det_option)
        mags2['lposthcand'] = mags2['lmag'].apply(lambda x: x if x > 0 and x < reject_th else 0)
        mags2['lnegthcand'] = mags2['lmag'].apply(lambda x: abs(x) if x < 0 and x > -reject_th else 0)
        mags2['rposthcand'] = mags2['rmag'].apply(lambda x: x if x > 0 and x < reject_th else 0)
        mags2['rnegthcand'] = mags2['rmag'].apply(lambda x: abs(x) if x < 0 and x > -reject_th else 0)

        # Let's make this mags2 an attribute of this class object
        self.accmags = mags2[['lmag', 'rmag']].copy()

        left_pos_peaks  = self._local_maxima(mags2.lposthcand.values, height = height)[1]
        left_neg_peaks  = self._local_maxima(mags2.lnegthcand.values, height = height)[1]
        right_pos_peaks = self._local_maxima(mags2.rposthcand.values, height = height)[1]
        right_neg_peaks = self._local_maxima(mags2.rnegthcand.values, height = height)[1]

        self.laccth  = self._calc_ind_threshold(left_pos_peaks)
        self.lnaccth = self._calc_ind_threshold(left_neg_peaks)*-1
        self.raccth  = self._calc_ind_threshold(right_pos_peaks)
        self.rnaccth = self._calc_ind_threshold(right_neg_peaks)*-1

        return ([self.laccth, self.lnaccth, self.raccth, self.rnaccth])

    def _prep_row_idx(self, in_en_dts):
        if in_en_dts is not None:
            # Getting the time differences:
            # 1) start_datetime - donned_datetime
            # 2) start_datetime - doffed_datetime
            diffs_in_microsec = map(lambda x: x - self._calc_datetime(self.left['Time'][0]), in_en_dts)

            # This list will include indices of the recording start time (in) and the recording end time (en).
            indices = list(map(lambda x: round((x.seconds*1e6 + x.microseconds)/50000), diffs_in_microsec))

            # This will be one input to self._get_mag
            row_idx = [x for x in range(indices[0], indices[1])]
        else:
            row_idx = None
            print("No recording start and end time provided. Analysis done on the entire recording")
        return(row_idx)

    # This is also the feature of Trujillo-Priego et al. (2017)    
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

    def _get_count(self):
        acounts = self.accmags.apply(np.sign)     # self.accmags is a DataFrame with two columns: lmag, rmag
        angvels = self._get_mag('Gyroscope', self.row_idx, 'median')      # angvels is another DataFrame with two columns
        acounts[angvels.le(0)] = 0                          # When angular velocity value is 0, count is 0.
        return(acounts)

    # Let's continue working on it over the weekend or so (10/19/22)
    def _get_Tcount(self):
        # _get_count does not seem to take too much of the time. We may use it.
        acounts = self._get_count()
        negcntidx = (acounts == -1)
        poscntidx = (acounts == 1)

        # Let's start with collecting all acc values that went over the threshold
        for_left_pos    = np.where(self.accmags['lmag'][poscntidx] > self.laccth)
        for_right_pos   = np.where(self.accmags['rmag'][poscntidx] > self.raccth)
        for_left_neg    = np.where(self.accmags['lmag'][negcntidx] < self.lnaccth)
        for_right_neg   = np.where(self.accmags['rmag'][negcntidx] < self.rnaccth)

        # parameters = indices of data points that are over the threshold (over_th_array) and
        #               acounts['lmags'] or acounts['rmags']
        def mark_Tcount(over_th_array, acounts, pos=True):
            if pos:
                corrsign = 1
            else:
                corrsign = -1
            N = len(over_th_array)
            Tcount = np.zeros(N)
            for i, j in enumerate(over_th_array):
                if (Tcount[j] == corrsign) or (Tcount[over_th_array[i+1]] == corrsign):
                    continue
                else:
                    # If three consecutive data are 1 or -1, the third data point's Tcount would be 1 or -1
                    if np.all(acounts[j:j+3] == corrsign):
                        Tcount[j+2] = corrsign
                    else:
                        Tcount[j] = corrsign

            nonzeroTC = np.where(Tcount != 0)

            # Remove duplicates
            for i, j in enumerate(nonzeroTC):
                if nonzeroTC[i+1] == (j+1):
                    Tcount[j] = 0

            return(Tcount)

        # keep failing this attempt...
        #self.Tcounts = pd.DataFrame(data = dict(zip(['L', 'R'], map(partial(mark_Tcount, side=['L', 'R']), [for_left, for_right]))))
        ltcounts  = mark_Tcount(for_left_pos, acounts, True)
        lntcounts = mark_Tcount(for_left_neg, acounts, False)
        rtcounts  = mark_Tcount(for_right_pos, acounts, True)
        rntcounts = mark_Tcount(for_right_neg, acounts, False)
        self.Tcounts = pd.DataFrame(data = {'L':ltcounts + lntcounts, 'R':rtcounts+rntcounts})
        
        return(self.Tcounts)

# This is a function that extracts times when the sensor was eonned or doffed     
def find_timepts_from_redcap(redcap_csv, full_id):
    # full_id will be in the format: "@@@v$" where @@@ is the infant id and $ is the number of visit
    # times will be the sensor donned and doffed times, based on the full_id
    infant_id, visit = full_id.split('v')
    times = redcap_csv[(redcap_csv.study_id == int(infant_id)) & (redcap_csv.visit == int(visit))][['time_donned', 'time_doffed']]
    return(times.values[0])

# Prepare the datetime objects based on 
#   donned and doffed times provided as an input
def make_start_end_datetime(don_and_doff, filename):
    # Getting year, month, and day from the filename
    temp = filename.split('/')[-1].split('-')[0]
    year    = int(temp[0:4])
    month   = int(temp[4:6])
    day     = int(temp[6:8])
    # Split hour and minute
    donned_h, donned_m = don_and_doff[0].split(":")
    doffed_h, doffed_m = don_and_doff[1].split(":")
    # donned_dt could be the earliest point that matches donned_h
    donned_dt = datetime(year, month, day, int(donned_h), int(donned_m), 0, 0)
    # Let's be lenient, and give 30 seconds of datapoints more
    if int(doffed_h) < 13 & int(doffed_h) > int(donned_h):
        doffed_dt = datetime(year, month, day+1, int(doffed_h), int(doffed_m), 30, 0)
    else:
        doffed_dt = datetime(year, month, day, int(doffed_h), int(doffed_m), 30, 0)
    return ([donned_dt, doffed_dt])

