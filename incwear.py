'''
HDF stands for [Hierarchical Data Format]. Every object in an HDF5 file has a name,
and they're arranged in a POSIX-style hierarchy with / separators
'''
import re
import numpy as np
import pandas as pd
import h5py
from functools import partial
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, detrend
from datetime import datetime, timedelta, timezone
import pytz
'''
Written by Jinseok Oh, Ph.D.
2022/9/13 - present (as of 2022/12/6)

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

Class Subject will try to have all those values,
    particularly those used in the actual analysis.
However, not in the exact format.
For example, if 'Annotations' is saved as an attribute: 'annots',
    then a user can access its sub-attribute, such as caseID
[well but it's not used anyways...]
'''
#filename = '/Users/joh/Downloads/20220322-084542_LL020_M2_D2.h5'
class Subject():

    # Make the default detrend option to median. Another option is 'customfunc'.
    def __init__(self, filename, in_en_dts, det_option = 'median'):

        f = h5py.File(filename, 'r')

        # Save the file name
        self.filename = filename

        # Annotations comprises three columns: Time, Case ID, annotation
        # Time in epoch microseconds... what does this mean?
        self.annots = f['Annotations']
      
        '''
        'Sensors' has TWO members.
        Each subgroup will have a name in the format of XI-XXXXXX.
        The number after the dash is the Case ID.
        This study used two sensors during a recording,
            so there will be two subgrops.
        '''
        self.sensors = f['Sensors']
        self.sensorids = list(self.sensors.keys())

        # To determine which sensor was attached to the left foot
        leftidx = 0
        whichfoot = self.sensors[self.sensorids[leftidx]]\
                ['Configuration']['Config Strings'][0][2].decode()
        if ('derecho' in whichfoot) or ('right' in whichfoot.lower()):
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

        self.N = self.left['Accelerometer'].shape[0] # Num of datapoints

        self.T = 0.05 # 1 data point every 50000μs or 0.05s
        self.sfreq = int(1/self.T) # sampling frequency.

        # Recorded timepoints in UTC!
        self.rtime = map(self._calc_datetime, self.left['Time'])

        self.procs = f['Processed'] # the orientation in quarternion
        self.ori_left = self.procs[self.sensorids[leftidx]]['Orientation']
        self.ori_right = self.procs[self.sensorids[rightidx]]['Orientation']

        # Time in seconds unit - starts from 0 and the step size is 0.05s
        # This may be used later for plotting
        self.time = (self.left['Time'] - self.left['Time'][0])/1e6

        # Index with the recording start time(in) and the end time(en)
        # Note that in_en_dts are given in UTC as well
        self.row_idx = self._prep_row_idx(in_en_dts)

        # acceleration and gyrocscope norms
        self.accmags = self._get_mag('Accelerometer',
                                     self.row_idx,
                                     det_option)

        self.count = self._get_count()

        # Storing the acceleration threshold values: [laccth, lnaccth, raccth, rnaccth]
        self.laccth, self.lnaccth, self.raccth, self.rnaccth = self._get_ind_acc_threshold()

        # 'combined' are acc norms and count values
        lcombined = pd.merge(self.accmags.lmag,
                             self.count['lcount'],
                             right_index = True,
                             left_index = True)
        rcombined = pd.merge(self.accmags.rmag,
                             self.count['rcount'],
                             right_index = True,
                             left_index = True)
        # 'count' could not be the column name, because that's already taken
        #  for an attribute of a dataframe.
        self.lcombined = lcombined.rename(
                columns = {'lmag':'accmag', 'lcount':'counts'})
        self.rcombined = rcombined.rename(
                columns = {'rmag':'accmag', 'rcount':'counts'})

        # mark 1 where the acceleration norm vector element crossed
        #  either a positive or negative threshold
        self.ValLthr_pos = (self.lcombined.counts ==  1.0) & (self.lcombined.accmag > self.laccth)
        self.ValLthr_neg = (self.lcombined.counts == -1.0) & (self.lcombined.accmag < self.lnaccth)
        self.ValRthr_pos = (self.rcombined.counts ==  1.0) & (self.rcombined.accmag > self.raccth)
        self.ValRthr_neg = (self.rcombined.counts == -1.0) & (self.rcombined.accmag < self.rnaccth)

        # 'thresholded' attributes would be 
        #  1: over the positive threshold 
        # -1: under the negative threshold
        #  0: otherwise
        self.lthresholded = self.ValLthr_pos + self.ValLthr_neg * -1
        self.rthresholded = self.ValRthr_pos + self.ValRthr_neg * -1

        #self.Tcount = self._get_Tcount()
        # Tmov will be 1 if acceleration value 
        #  crossed thresholds (neg or pos) two times.
        self.Tmov = self._get_mov()

        # This is a dictionary with 2 keys ("Lkinematics", "Rkinematics")
        # Each key will have a corresponding dictionary as its value
        # 1) LMovstart : A vector storing indices of the start of a bout
        # 2) LMovend   : A vector storing indices of the end of a bout
        # 3) LMovLength: A vector that stores the length of each bout. Non-zero indices will be equal to those of Tmov
        # 4) LavepMov  : A vector that stores the average of the acceleration norm per bout.
        # 5) LpeakpMov : A vector that sotres the maximum of the acceleration norm per bout
        # 6) RMovLength, RavepMov, RpeakpMov: Right equivalents of the first three columns
        self.kinematics = self._raw_mov_kinematics()

    # This function will calculate the actual date and time from the time recorded in the sensor
    def _calc_datetime(self, x):
        '''
        [x] is a timestamp in units of microseconds
            since 0:00, Jan 1, 1970 UTC (source: APDM_DevelopersGuide)
        '''
        record_start_time = datetime.fromtimestamp(x/1e6, timezone.utc)
        return record_start_time

    # dtype argument should be either 'Accelerometer' or 'Gyroscope'
    # row_idx should be a list of two indices: user-specified array start and end
    # det_option: how are you going to [det]rend the data?
    def _get_mag(self, dtype, row_idx=None, det_option='median'):
        if det_option not in ['median', 'customfunc']:
            det_option='median'
            print("Wrong detrending option provided - setting it to [median]")

        if row_idx is None:
            row_idx = list(range(self.N))

        # Use np.linalg.norm... amazing!
        magdata_l = np.linalg.norm(self.left[dtype][row_idx], axis=1)
        magdata_r = np.linalg.norm(self.right[dtype][row_idx], axis=1)
        mags = pd.DataFrame(data = {'lmag':magdata_l, 'rmag':magdata_r})

        # MATLAB's detrend function is not used, so we can consider that the default option
        # for detrending the data is subtracting the median
        if det_option == 'median':
            out = mags - mags.median(axis=0)
        else:
            out = mags.apply(detrend, axis=0)   # else condition will be ['customfunc']
        return out

    # winsize in second unit
    def _mov_avg_filt(self, winsize, pdSeries):
        L = int(self.sfreq * winsize)
        return(pdSeries.rolling(L).mean())

    def _local_maxima(self, values, height):
        locmax, maxprop = find_peaks(values, height = height)
        return([locmax, maxprop])

    def _calc_ind_threshold(self, maxprop, ivan=False):
        return(np.mean(maxprop['peak_heights']) 
               - (1-ivan + 0.5*ivan)*np.std(maxprop['peak_heights']))

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
            #mags2 = self._get_mag('Accelerometer', self.row_idx, 'median')
            mags2 = self.accmags.copy()
            mags2['rectlmag'] = mags2.lmag.apply(
                    lambda x: abs(x) if x > max(reject_th) or x < min(reject_th) else 0)
            mags2['rectrmag'] = mags2.rmag.apply(
                    lambda x: abs(x) if x > max(reject_th) or x < min(reject_th) else 0)

            mags2['avglmag'] = self._mov_avg_filt(winsize, mags2['rectlmag'])
            mags2['avgrmag'] = self._mov_avg_filt(winsize, mags2['rectrmag'])


            laccth = self._calc_ind_threshold(
                    self._local_maxima(mags2.avglmag.values, height=height)[1])
            raccth = self._calc_ind_threshold(
                    self._local_maxima(mags2.avgrmag.values, height=height)[1])

            # Now these values will be used for both positive and negative borders
            return([laccth, raccth])

    # This is equivalent to the MATLAB script
    def _get_ind_acc_threshold(self):

        reject_th = 3.2501
        height = 1.0

        #mags2 = self._get_mag('Accelerometer', self.row_idx, det_option)
        mags2 = self.accmags.copy()
        mags2['lposthcand'] = mags2.lmag.apply(lambda x: x if x > 0 and x < reject_th else 0)
        mags2['lnegthcand'] = mags2.lmag.apply(lambda x: abs(x) if x < 0 and x > -reject_th else 0)
        mags2['rposthcand'] = mags2.rmag.apply(lambda x: x if x > 0 and x < reject_th else 0)
        mags2['rnegthcand'] = mags2.rmag.apply(lambda x: abs(x) if x < 0 and x > -reject_th else 0)

        left_pos_peaks  = self._local_maxima(mags2.lposthcand.values, height = height)[1]
        left_neg_peaks  = self._local_maxima(mags2.lnegthcand.values, height = height)[1]
        right_pos_peaks = self._local_maxima(mags2.rposthcand.values, height = height)[1]
        right_neg_peaks = self._local_maxima(mags2.rnegthcand.values, height = height)[1]

        laccth  = self._calc_ind_threshold(left_pos_peaks)
        lnaccth = self._calc_ind_threshold(left_neg_peaks)*-1
        raccth  = self._calc_ind_threshold(right_pos_peaks)
        rnaccth = self._calc_ind_threshold(right_neg_peaks)*-1

        return ([laccth, lnaccth, raccth, rnaccth])

    def _prep_row_idx(self, in_en_dts):

        if in_en_dts is not None:
            d_in_microsec = list(map(
                lambda x: x - self._calc_datetime(self.left['Time'][0]), in_en_dts))
            '''
            d_in_microsec is the list of TWO datetime.timedelta objects
            The first element of this list shows the time difference between
                the start of the time recorded in sensors and don_t.
            The second element is the analogous for doff_t.
            The sampling frequency of the APDM OPAL sensor is 20Hz,
                so each data point is 1/20 seconds or 5e4 microseconds.
            Therefore, if the time difference is represented in the
                microseconds unit and divided by 50000 (with a bit of
                rounding) you get how many data points don_t and doff_t
                are away from the index 0.
            Consequently, two indices will be searched:
                1) startidx = data index that corresponds to the don_t
                2) endidx = data index that corresponds to the doff_t.
                    For this one, there are occasions where the
                    REDCap reports are 'inaccurate', meaning that the
                    sensors were turned off long before the reported
                    doff_t and this exception needs to be handled by
                    simply taking the last value of the time series
            '''
            C = 1e6 # conversion constant: 1 second = 1e6 microseconds
            N = self.T * 1e6    # 50000.0
            poten = round((d_in_microsec[1].seconds*C
                           + d_in_microsec[1].microseconds)/N)
            if poten < self.left['Time'].shape[0]:
                indices = list(map(
                    lambda x: round((x.seconds*C + x.microseconds)/N), d_in_microsec))
            else: 
                indices = list()
                indices.extend([round((d_in_microsec.seconds*C
                                      + d_in_microsec.microseconds)/N),
                                self.left['Time'].shape[0]-1])

            # This will be one input to self._get_mag
            row_idx = list(range(indices[0], indices[1]))
        else:
            row_idx = None
            print("No recording start and end time provided. \
                    Analysis done on the entire recording")
        return row_idx

    # This is also the feature of Trujillo-Priego et al. (2017)
    def get_ind_angvel_threshold(self, reject_th = 0.32, winsize = 0.5, height=0.01):
        errmsg = "The rejection threshold for the gyroscope \
                data should be a positive value"
        if reject_th < 0:
            print(errmsg)
        else:
            mags2 = self._get_mag('Gyroscope', self.row_idx, 'median')
            mags2['rectlmag'] = mags2.lmag.apply(lambda x: x if x > reject_th else 0)
            mags2['rectrmag'] = mags2.rmag.apply(lambda x: x if x > reject_th else 0)

            mags2['avglmag'] = self._mov_avg_filt(winsize, mags2['rectlmag'])
            mags2['avgrmag'] = self._mov_avg_filt(winsize, mags2['rectrmag'])

            self.avmag = mags2

            self.lavth = self._calc_ind_threshold(
                    self._local_maxima(mags2.avglmag.values, height=height)[1])
            self.ravth = self._calc_ind_threshold(
                    self._local_maxima(mags2.avgrmag.values, height=height)[1])

            return([self.lavth, self.ravth])

    def _get_count(self):
        acounts = self.accmags.copy().apply(np.sign)     # self.accmags is a DataFrame with two columns: lmag, rmag
        angvels = self._get_mag('Gyroscope', self.row_idx, 'median')      # angvels is another DataFrame with two columns
        acounts2 = acounts.rename(columns={'lmag':'lcount', 'rmag':'rcount'})
        acounts2.lcount[angvels.lmag.le(0)] = 0                          # When angular velocity value is 0, count is 0.
        acounts2.rcount[angvels.rmag.le(0)] = 0
        return acounts2

    # Let's continue working on it over the weekend or so (10/19/22)
    def _get_Tcount(self):

        # Let's start with collecting all acc values that went over the threshold
        # The output of np.where would be a tuple - so take the first value
        # I do this to reduce the preprocessing time...
        left_over_posth  = np.where(self.ValLthr_pos)[0]
        right_over_posth = np.where(self.ValRthr_pos)[0]
        left_under_negth = np.where(self.ValLthr_neg)[0]
        right_under_negth = np.where(self.ValRthr_neg)[0]

        # parameters = indices of data points that are over the threshold (over_th_array) and
        #               acounts['lmags'] or acounts['rmags']
        def mark_Tcount(over_th_array, acounts_series, pos=True):
            if pos:
                corrsign = 1
            else:
                corrsign = -1
            N = len(acounts_series)
            M = len(over_th_array)
            Tcount = np.zeros(N)
            for i, j in enumerate(over_th_array):
                '''
                "over_th_array" has the indices of the acceleration values 
                    that are over a threshold (left or right).
                [j] gives one of those indices while [i] indicates 
                    the order of [j] in "over_th_array".
                It could be that the Tcount value at the current index 
                    may have been set by the previous index 
                    (ex. [j-2] satisfied the "else" condition so Tcount[j] = 1 or -1)
                If so, skip to the next index.
                Also, if the next index is not zero, then the current 
                    Tcount[j] is considered as a redundant count and 
                    marked off (this is from the original MATLAB code). 
                    So we can skip such indices here.
                '''
                if (Tcount[j] == corrsign) or ((i <= (M-2)) and (Tcount[j+1] == corrsign)):
                    continue
                else:
                    # If three consecutive data are 1 or -1, 
                    #   the third data point's Tcount would be 1 or -1
                    if np.all(acounts_series[j:j+3] == corrsign):
                        Tcount[j+2] = corrsign
                    else:
                        Tcount[j] = corrsign

            nonzeroTC = np.where(Tcount != 0)[0]
            L = len(nonzeroTC)

            # Remove duplicates
            for i, j in enumerate(nonzeroTC):
                if (i <= (L-2)) and (nonzeroTC[i+1] == (j+1)):
                    Tcount[j] = 0
                elif (i == (L-1)) and (nonzeroTC[i-1] == (j-1)):
                    Tcount[j-1] = 0

            return Tcount

        # keep failing this attempt...
        #self.Tcounts = pd.DataFrame(data = dict(zip(['L', 'R'], map(partial(mark_Tcount, side=['L', 'R']), [for_left, for_right]))))
        ltcounts  = mark_Tcount(left_over_posth, self.count['lcount'], True)
        lntcounts = mark_Tcount(left_under_negth, self.count['lcount'], False)
        rtcounts  = mark_Tcount(right_over_posth, self.count['rcount'], True)
        rntcounts = mark_Tcount(right_under_negth, self.count['rcount'], False)
        Tcounts = pd.DataFrame(data = {'L':ltcounts + lntcounts, 
                                       'R':rtcounts+rntcounts})

        return Tcounts

    def _get_mov(self):
        Tcounts = self._get_Tcount()

        def tmov(temp):
            Tmov = np.zeros(len(temp))
            # Among non-zero Tcount (-1 or 1) values...
            nonzeroTC = np.where(temp!=0)[0]
            for i, j in enumerate(nonzeroTC[:-1]):
            # If the difference between the current point and 
            #   its subsequent one is greater than 8 data points, 
            #   skip the current point.
                if (np.diff([j, nonzeroTC[i+1]])[0] > 8) or (Tmov[j] == 1):
                    continue
                # If not, if the sign of the two adjacent points differ 
                #   (-1 vs. 1 or vice versa; 0 cannot be included because 
                #   points attempt nonzeroTC cannot be 0)
                else:
                    if np.sign(temp[j]) != np.sign(temp[nonzeroTC[i+1]]):
                        Tmov[nonzeroTC[i+1]] = 1
                    else:
                        continue
            return Tmov

        Tmov = Tcounts.apply(tmov, axis=0)
        return Tmov

    def _raw_mov_kinematics(self):

        def start_to_end(Tmov_series, thr_array, combined, side='L'):
            Tmovidx = np.nonzero(Tmov_series.values)[0]        # Indices where Tmov == 1
            N = len(Tmovidx)                            # This number is what we need
            lenMov  = np.zeros(N)       # How many data points between the start and the end of a bout?
            avepMov = np.zeros(N)       # Average acceleration per Movement
            peakpMov = np.zeros(N)      # Peak acceleration per Movement
            movstart = np.zeros(N)      # Will store indices of the starts of bouts
            movend   = np.zeros(N)      # Will store indices of the ends of bouts
            if side == 'R':
                colnames = ['RMovIdx', 
                            'RMovStart', 
                            'RMovEnd', 
                            'RMovLength', 
                            'RavepMov', 
                            'RpeakpMov']
            else:
                colnames = ['LMovIdx', 
                            'LMovStart', 
                            'LMovEnd', 
                            'LMovLength', 
                            'LavepMov', 
                            'LpeakpMov']

            # j is the index of Tmov's. 
            # At any j, count will be -1 or 1 and not 0.
            for i, j in enumerate(Tmovidx):
                k = -1              # Moving backward...
                while True:
                    # ... to find the data point that crossed the threshold value 
                    #   whose sign is opposite to the count at j. This means that 
                    #   the baseline (accmag = 0) is crossed once.
                    if np.sign(thr_array[j+k]) == -np.sign(combined['counts'][j]):
                        movstartidx = int(j+k)      # The start of a bout
                        break
                    else:
                        k -= 1      # Keep going backwards if a previous attempt failed
                m = 1               # Moving forward...
                while True:
                    # ... to find the data point that either touches 
                    # or crosses the baseline one more time.
                    # This time the sign of the data point at the
                    # index [j+l] could be 0 or opposite to that of
                    # the datapoint at the index [j]
                    if np.sign(combined['counts'][j+m]) != np.sign(combined['counts'][m]):
                        movendidx = int(j+m)       # The end of a bout
                        break
                    else:
                        m += 1
                movstart[i] = movstartidx
                movend[i] = movendidx
                lenMov[i] = movendidx - movstartidx + 1
                avepMov[i] = sum(abs(combined.accmag[movstartidx:(movendidx+1)])) / lenMov[i]
                peakpMov[i] = max(abs(combined.accmag[movstartidx:(movendidx+1)]))

            return(dict(zip(colnames, [Tmovidx, movstart, movend, lenMov, avepMov, peakpMov])))

        Lkinematics = start_to_end(self.Tmov.L,
                                   self.lthresholded.copy(),
                                   self.lcombined.copy(),
                                   'L')
        Rkinematics = start_to_end(self.Tmov.R,
                                   self.rthresholded.copy(),
                                   self.rcombined.copy(),
                                   'R')
        # Lkinematics and Rkinematics could differ in length - so better return a dictionary.
        kinematics  = {'Lkinematics':Lkinematics, 'Rkinematics':Rkinematics}

        return kinematics

    def _mark_simultaneous(self):

        '''
        We need to re-evaluate how we operationalize simultaneity of bouts.
        Let's suppose that at the index j, Tmov.L[j] = 1
        If the following three conditions are all true for another index k:
          1) Tmov.R[k] = 1              (right leg bout)
          2) RestMov[(k-n):(k+m+1)] = 1 (duration of that bout)
          3) k-n =< j <= k+m            (left leg bout located inside
                                         the duration of the right leg bout)
        we can argue that the bout of the left leg at the index j occurred
        in the vicinity of the bout of the right leg at the index k.
        Consequently, we could label these movements as "simultaneous".
        Simultaneous will be further specified to two cases.
            1) Bilateral Synchronous: starting points of RestMov and LestMov are exactly the same
            2) Bilateral Asynchronous: starting points don't match, but they overlap
        '''
        r_dict = self.kinematics['Rkinematics'].copy()
        l_dict = self.kinematics['Lkinematics'].copy()

        # sole_r and sole_l will each show the indices of nonzero RTmov and LTmov elements
        # In other words, locations of bouts
        sole_r = pd.DataFrame(data = {'MovIdx':r_dict['RMovIdx'],
                                      'Start':r_dict['RMovStart'],
                                      'End': r_dict['RMovEnd']})
        sole_l = pd.DataFrame(data = {'MovIdx':l_dict['LMovIdx'],
                                      'Start':l_dict['LMovStart'],
                                      'End': l_dict['LMovEnd']})

        # If RStart is equal to Lstart, the corresponding RMovIdx is
        # bilateral synchronous to the LMovIdx associated with the LStart
        # If RStart is not equal, but RMovIdx is in between LMovStart and LMovEnd,
        # then it's bilateral asynchronous

        def match_idx(sole_1, sole_2, refside = 'R'):
            bilatSyncidx = {}
            bilatAsyncidx = {}
            bilatTotal = {}
            if refside == 'L':
                keys = ['LStart', 'RStart']
            else:
                keys = ['RStart', 'LStart']
            for i, mov in enumerate(sole_1.MovIdx):
                for j in range(sole_2.shape[0]):
                    if (mov > sole_2.Start[j]) and (mov < sole_2.End[j]):
                        bilatTotal[mov] = i
                        if sole_1.Start[i] == sole_2.Start[j]:
                            # You're storing the row indices of LStart and RStart
                            # This is for the ease of later processing of sole_r and sole_l
                            bilatSyncidx[mov] = dict(zip(keys, [i,j]))
                            break
                        else:
                            bilatAsyncidx[mov] = dict(zip(keys, [i,j]))
                            break
            return({'Sync':bilatSyncidx, 'Async':bilatAsyncidx, 'Total':bilatTotal})

        # still takes quite a long time...need to make it more efficient
        bilat_r = match_idx(sole_r, sole_l)
        bilat_l = match_idx(sole_l, sole_r, 'L')

        # Return information about simultaneous moves
        return({'RSim':bilat_r, 'LSim':bilat_l})

def make_start_end_datetime(redcap_csv, filename, site):
    '''
    A filename (if we're living in an ideal world) is in the format:
        '/Some path/YYYYmmdd-HHMMSS_[participant identifier].h5'
    Use the .h5 filename -> search times sensors were donned and doffed
    A redcap_csv will be in the following format (example below)

    id    |             filename              | don_t | doff_t
    ----------------------------------------------------------
    LL001 | 20201223-083057_LL001_M1_Day_1.h5 | 9:20  | 17:30

    Split the filename with '/' and take the first (path_removed).
    Then we concatenate them with the time searched from the REDCap
        export file to generate don_dt and doff_dt (dt: datetime object)

    You're also requested to provide the 'site' where the data
        were collected. This eventually will be provided as the
        drop-down menu whose values are from pytz.all_timezones
    '''
    path_removed = filename.split('/')[-1]

    def match_any(x, filename):
        '''
        Highly likely, entries of the column: filename in redcap_csv
            could be erroneous. Therefore, to be on the safe side,
            split REDCap entry and check if more than half the splitted items
            are included in the filename.
        '''
        if type(x) == str:
            ls = re.split('[-:_.]', x)
            lowered = list(map(lambda x:x.lower(), ls))
            if np.mean([x in filename.lower() for x in lowered]) > 0.5:
                return True
            else:
                return False
        else:
            return False

    idx = redcap_csv.filename.apply(lambda x: match_any(x, filename))

    don_n_doff = redcap_csv.loc[np.where(idx)[0][0], ['don_t', 'doff_t']]
    #don_n_doff = times.values[0]  # times is a Pandas Series

    temp = path_removed.split('-')[0]
    don_h, _ = don_n_doff[0].split(":")
    doff_h, _ = don_n_doff[1].split(":")

    don_dt = datetime.strptime('-'.join([temp, don_n_doff[0]]),
                                 '%Y%m%d-%H:%M') + timedelta(minutes=1)
    doff_temp = datetime.strptime('-'.join([temp, don_n_doff[1]]),
                                    '%Y%m%d-%H:%M')
    '''
    There are cases where the REDCap values are spurious at best.
    Sometimes the first time point recorded in sensors could be later
        than what's listed as the time_donned in the REDCap file by
        few microseconds.
    This would case the problem later, as the method of the Subject class
        [_prep_row_idx] will 'assume' that the opposite is always true,
        and calculate: REDCap time_donned - sensor initial time.
    Of course this will be a negative value, ruining everything; so add
        a minute to what's reported on the REDCap time donned.
    If the time sensors were doffed was early in the morning (ex. 2 AM)
        you know that a day has passed.
    The condition below, however, may not catch every possible exception.
    Let's hope for the best that everyone removed the sensors
        before 2 PM the next day.
    '''
    if int(doff_h) < 14 & (abs(int(don_h) - int(doff_h)) < 10):
        doff_dt = doff_temp + timedelta(days=1)
    else:
        doff_dt = doff_temp

    def convert_to_utc(dt, site):
        local = pytz.timezone(site)
        local_dt = local.localize(dt, is_dst = None)
        utc_dt = local_dt.astimezone(pytz.utc)
        return utc_dt

    # site-specific don/doff times are converted to UTC time
    utc_don_doff = list(map(lambda lst: convert_to_utc(lst, site), [don_dt, doff_dt]))
    return utc_don_doff
