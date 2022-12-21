"""
Written by Jinseok Oh, Ph.D.
2022/9/13 - present (as of 2022/12/6)

HDF stands for [Hierarchical Data Format]. Every object in an HDF5 file has a name,
and they're arranged in a POSIX-style hierarchy with / separators

incwear.py is a python script porting MakeDataStructure_v2.m,
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
"""
import re
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, detrend
import pytz
#filename = '/Users/joh/Downloads/20220322-084542_LL020_M2_D2.h5'

@dataclass
class tderivs:
    """ Storing measures derived from thresholds """
    accmags: pd.DataFrame
    thresholds: list    # [laccth, lnaccth, raccth, rnaccth]
    over_accth: dict = field(init=False)
    under_naccth: dict = field(init=False)
    # 'th_crossed' attributes would be
    #  1: over the positive threshold
    # -1: under the negative threshold
    #  0: otherwise
    th_crossed: dict = field(init=False)
    def __post_init__(self):
        self.over_accth = {'L': self.accmags.lmag > self.thresholds[0],
                           'R': self.accmags.rmag > self.thresholds[2]}
        self.under_naccth = {'L': self.accmags.lmag < self.thresholds[1],
                             'R': self.accmags.rmag < self.thresholds[3]}
        self.th_crossed = {
                'L': self.over_accth['L'] + self.under_naccth['L'] * -1,
                'R': self.over_accth['R'] + self.under_naccth['R'] * -1}

class subject:
    """
    An object that will store values of kinematic variables
        derived from the OPAL V2 recordings
    """
    def __init__(self, filename, in_en_dts, label_r):

        with h5py.File(filename, 'r') as h5file:
            # Sensors has TWO members.
            # Each subgroup has a name with the format: XI-XXXXXX
            # The number after the dash is the Case ID
            sensors = h5file['Sensors']
            sensorids = list(sensors.keys())

            # We need to find out which sensor was attached to which leg.
            # First, we read the label of the first Case ID (sensorlabel)
            sensorlabel = sensors[sensorids[0]]\
                    ['Configuration']['Config Strings'][0][2].decode()
            # Second, we compare sensorlabel with the user provided label
            # for the "right" (ex. 'right', 'R', 'Right_leg', 'derecho'...)
            # If the two matches, then the first Case ID corresponds to
            # the sensor attached to the right leg.
            if label_r.lower() in sensorlabel.lower():
                self.sensors = {'L':sensors[sensorids[1]],
                                'R':sensors[sensorids[0]]}
            # If the two do not match, then the first Case ID corresponds
            # to the sensor attached to the left leg.
            else:
                self.sensors = {'L':sensors[sensorids[0]],
                                'R':sensors[sensorids[1]]}

            # Recorded timepoints in UTC!
            # self.rtime = map(self._calc_datetime, self.left['Time'])

            # Time in seconds unit - starts from 0 and the step size is 0.05s
            # This may be used later for plotting
            # self.time = (self.left['Time'] - self.left['Time'][0])/1e6

            # Index with the recording start time(in) and the end time(en)
            # Note that in_en_dts are given in UTC as well
            row_idx = self._prep_row_idx(self.sensors['L'], in_en_dts)

            # acceleration and gyrocscope norms
            accmags = self._get_mag(self.sensors, 'acc', row_idx)
            angvels = self._get_mag(self.sensors, 'gyro', row_idx)

            acounts = self._get_count(accmags, angvels)

            # Storing the acceleration threshold values:
            #   [laccth, lnaccth, raccth, rnaccth]
            thresholds = self._get_ind_acc_threshold(accmags)
            self.tderivs = tderivs(accmags = accmags,
                                   thresholds = thresholds)

            # Tmov will be 1 if acceleration value
            #  crossed thresholds (neg or pos) two times.
            # Why have it as an instance attribute? For plotting, of course.
            tcounts = self._get_tcount(acounts)
            self.Tmov = self._get_mov(tcounts)

            # This is a dictionary with 2 keys ("Lkinematics", "Rkinematics")
            # Each key will have a corresponding dictionary as its value
            #   [MovIdx, MovLength, MovStart, MovEnd, avepMov, peakpMov]
            self.kinematics = self._raw_mov_kinematics(accmags, acounts)

    # This function will calculate the actual date and time from the time recorded in the sensor
    def _calc_datetime(self, time_stamp):
        """
        A function to convert a timestamp into a datetime instance

        Parameters:
            time_stamp: int
                a timestamp in units of microseconds since 1970-1-1-0:00 UTC

        Returns:
            a datetime instance whose value is converted from the time_stamp
        """
        record_start_time = datetime.fromtimestamp(time_stamp/1e6,
                                                   timezone.utc)
        return record_start_time

    def _get_mag(self, sensors, device, row_idx=None, det_option='median'):
        """
        A function to calculate the magnitude of a provided dtype.

        Parameters:
            sensors: dictionary of h5file['sensor'] members
                OPAL V2 Sensor recordings
            device: {'acc', 'gyro'}
                Accelerometer or Gyroscope data
            row_idx: list of ints
                Recording start and end index
            det_option: {'median', 'customfunc'}
                Subtract median or use some custom method to detrend

        Returns:
            detrended magnitude of acceleration or angular velocity
        """
        # check if dtype_label is correctly provided.
        assert(device in ['acc', 'gyro']),\
                "device is either 'acc' or 'gyro'"
        if device == 'acc':
            dtype = 'Accelerometer'
        else:
            dtype = 'Gyroscope'

        # sensors['L'][dtype] will be a N x 3 matrix whose entries are
        # either accelerations or angular velocities along x, y, z axes.
        # nrow is equal to N
        nrow = sensors['L'][dtype].shape[0]

        if det_option not in ['median', 'customfunc']:
            det_option='median'
            print("Unknown detrending option - setting it to [median]")

        if row_idx is None:
            row_idx = list(range(nrow))    # targeting the entire dataset

        # Use np.linalg.norm... amazing!
        lmag, rmag = map(lambda x: np.linalg.norm(x[row_idx], axis=1),
                         [sensors['L'][dtype], sensors['R'][dtype]])
        #magdata_l = partial(map(np.linalg.norm(arr_l[row_idx], axis=1)
        #magdata_r = np.linalg.norm(arr_r[row_idx], axis=1)
        mags = pd.DataFrame(data = {'lmag':lmag, 'rmag':rmag})

        # MATLAB's detrend function is not used,
        #   so we can consider that the default option
        #   for detrending the data is subtracting the median
        if det_option == 'median':
            out = mags - mags.median(axis=0)
        else:
            out = mags.apply(detrend, axis=0)   # else condition: [customfunc]
        return out

    # winsize in second unit
    def _mov_avg_filt(self, winsize, pd_series):
        win_len = int(20 * winsize)   # sfreq = 20Hz
        return(pd_series.rolling(win_len).mean())

    def _calc_ind_threshold(self, maxprop, ivan=False):
        return(np.mean(maxprop['peak_heights'])
               - (1-ivan + 0.5*ivan)*np.std(maxprop['peak_heights']))

    def get_ind_acc_threshold_ivan(self, accmags, winsize=0.5, height=1.0):
        """
        All relevant parameter values coming from Trujillo-Priego et al., (2017)
        Needs further development
        """
        reject = [-1.02, 1.32]

        mags2 = accmags.copy()
        mags2['rectlmag'] = mags2.lmag.apply(
                lambda x: abs(x) if x > max(reject) or x < min(reject) else 0)
        mags2['rectrmag'] = mags2.rmag.apply(
                lambda x: abs(x) if x > max(reject) or x < min(reject) else 0)

        mags2['avglmag'] = self._mov_avg_filt(winsize, mags2['rectlmag'])
        mags2['avgrmag'] = self._mov_avg_filt(winsize, mags2['rectrmag'])

        laccth = self._calc_ind_threshold(
                find_peaks(mags2.avglmag.values, height=height)[1])
        raccth = self._calc_ind_threshold(
                find_peaks(mags2.avgrmag.values, height=height)[1])

        # used for both positive and negative borders
        return([laccth, raccth])

    # This is equivalent to the MATLAB script
    def _get_ind_acc_threshold(self, accmags, reject = 3.2501, height = 1.0):
        """
        A function to find individual thresholds

        Parameters:
            accmags: pd.DataFrame
                detrended acceleration norms (L/R)
            reject: float, optional
                a cut-off; values below this number will be included
            height: float, optional
                minimal height that defines a peak

        Returns:
            a list of thresholds
                laccth: left positive threshold
                lnaccth: left negative threshold
                raccth: right positive threshold
                rnaccth: right negative threshold
        """
        mags2 = accmags.copy()
        mags2['lpospks'] = mags2.lmag.apply(
                lambda x: x if 0 < x < reject else 0)
        mags2['lnegpks'] = mags2.lmag.apply(
                lambda x: abs(x) if -reject < x < 0 else 0)
        mags2['rpospks'] = mags2.rmag.apply(
                lambda x: x if 0 < x < reject else 0)
        mags2['rnegpks'] = mags2.rmag.apply(
                lambda x: abs(x) if -reject < x < 0 else 0)

        pnpks = mags2.iloc[:, ['lpospks',
                               'lnegpks',
                               'rpospks',
                               'rnegpks']].apply(
                                       lambda x: find_peaks(x,
                                                            height=height)[1])

        laccth  = self._calc_ind_threshold(pnpks['lpospks'])
        lnaccth = self._calc_ind_threshold(pnpks['lnegpks']) * -1
        raccth  = self._calc_ind_threshold(pnpks['rpospks'])
        rnaccth = self._calc_ind_threshold(pnpks['rnegpks']) * -1

        return ([laccth, lnaccth, raccth, rnaccth])

    def _prep_row_idx(self, sensorobj, in_en_dts):
        """
        A function to return two datetime instances that correspond to
            the start and the end of recording.

        Parameters:
            sensorobj: HDF group
                either sensor['L'] or sensor['R']
            in_en_dts: list
                times sensors were donned (don_t) and doffed (doff_t)

        Returns:
            row_idx: list or None
                If donned and doffed times are found from the REDCap export,
                return a list of two indices of datapoints that each
                corresponds to the start and the end of the recording.
                If no time is found, return None.
        """
        if in_en_dts is not None:
            # d_in_micro is the list of TWO datetime.timedelta objects
            # The first element of this list shows the time difference
            #   between the start of the time recorded in sensors and don_t.
            # The second element is the analogous for doff_t.
            # The sampling frequency of the APDM OPAL sensor is 20Hz,
            #   so each data point is 1/20 seconds or 5e4 microseconds.
            # Therefore, if the time difference is represented in the
            #   microseconds unit and divided by 50000 (with a bit of
            #   rounding) you get how many data points don_t and doff_t
            #   are away from the index 0.
            # Consequently, two indices will be searched:
            #   1) startidx = data index that corresponds to the don_t
            #   2) endidx = data index that corresponds to the doff_t.
            #       For this one, there are occasions where the
            #       REDCap reports are 'inaccurate', meaning that the
            #       sensors were turned off long before the reported
            #       doff_t and this exception needs to be handled by
            #       simply taking the last value of the time series
            d_in_micro = list(map(
                lambda x: x - self._calc_datetime(sensorobj['Time'][0]), in_en_dts))
            convert = 1e6    # conversion constant: 1 second = 1e6 microseconds
            microlen = 0.05 * convert    # duration of a data point in microseconds
            poten = round((d_in_micro[1].seconds*convert
                           + d_in_micro[1].microseconds)/microlen)
            if poten < sensorobj['Time'].shape[0]:
                indices = list(map(
                    lambda x: round(
                        (x.seconds*convert + x.microseconds)/microlen),
                    d_in_micro))
            else:
                indices = []
                indices.extend([round((d_in_micro[0].seconds*convert
                                      + d_in_micro[0].microseconds)/microlen),\
                                sensorobj['Time'].shape[0]-1])

            # This will be one input to self._get_mag
            row_idx = list(range(indices[0], indices[1]))
        else:
            row_idx = None
            print("No recording start and end time provided. \
                    Analysis done on the entire recording")
        return row_idx

    def get_ind_angvel_threshold(self, accmags, reject = 0.32, winsize = 0.5, height=0.01):
        """ feature of Trujillo-Priego et al. (2017), needs development """
        errmsg = "The rejection threshold for the gyroscope \
                data should be a positive value"
        assert (reject > 0), errmsg

        mags2 = accmags.copy()
        mags2['rectlmag'] = mags2.lmag.apply(lambda x: x if x > reject else 0)
        mags2['rectrmag'] = mags2.rmag.apply(lambda x: x if x > reject else 0)

        mags2['avglmag'] = self._mov_avg_filt(winsize, mags2['rectlmag'])
        mags2['avgrmag'] = self._mov_avg_filt(winsize, mags2['rectrmag'])

        self.avmag = mags2

        self.lavth = self._calc_ind_threshold(
                find_peaks(mags2.avglmag.values, height=height)[1])
        self.ravth = self._calc_ind_threshold(
                find_peaks(mags2.avgrmag.values, height=height)[1])

        return([self.lavth, self.ravth])

    def _get_count(self, accmags, angvels):
        """
        A function to get counts. The count of a data point is
            the sign of the acceleration norm at the data point.
        If the angular velocity norm at the data point is less than 0,
            then the count is also 0.

        Arguments:
            accmags: pd.DataFrame
                detrended acceleration norms (L/R)

            angvels: pd.DataFrame
                detrended angular velocity norms (L/R)

        Returns:
            acounts2: pd.DataFrame
                count values (L/R)
        """
        acounts = accmags.copy().apply(np.sign)
        acounts2 = acounts.rename(columns={'lmag':'lcount', 'rmag':'rcount'})
        acounts2.lcount[angvels.lmag.le(0)] = 0   # angvel == 0 -> count = 0.
        acounts2.rcount[angvels.rmag.le(0)] = 0
        return acounts2

    def _get_tcount(self, acounts):
        """
        A function to get tcounts

        Parameters:
            acounts: pd.DataFrame
                output of self._get_count(accmags, angvels)

        Returns:
            tcounts: pd.DataFrame
                tcounts of left and right sensors.
        """

        # Let's start with collecting all acc values that went over the threshold
        # The output of np.where would be a tuple - so take the first value
        # I do this to reduce the preprocessing time...
        over_posth_l = np.where(self.tderivs.over_accth['L'])[0]
        over_posth_r = np.where(self.tderivs.over_accth['R'])[0]
        under_negth_l = np.where(self.tderivs.under_naccth['L'])[0]
        under_negth_r = np.where(self.tderivs.under_naccth['R'])[0]

        def mark_tcount(over_th_arr, acounts, pos=True):
            """
            Parameters:
                over_th_arr: np.array
                    indices of data points over a threshold
                    (ex: over_posth_l)
                acounts: pd.DataFrame
                    output of self._get_count(accmags, angvels)

            Returns:
                tcount: np.array
                    nonzero counts that crossed
                    a positive or negative threshold
            """
            corrsign = -1
            if pos:
                corrsign = 1
            arr_len = len(over_th_arr)
            t_count = np.zeros(len(acounts))
            for i, j in enumerate(over_th_arr):
                # "over_th_array" has the indices of the acceleration values
                #   that are over a threshold (left or right).
                # [j] gives one of those indices while [i] indicates
                #   the order of [j] in "over_th_array".
                # It could be that the Tcount value at the current index
                #   may have been set by the previous index
                #   (ex. [j-2] satisfied the "else" condition so Tcount[j] = 1 or -1)
                # If so, skip to the next index.
                # Also, if the next index is not zero, then the current
                #   Tcount[j] is considered as a redundant count and
                #   marked off (this is from the original MATLAB code).
                #   So we can skip such indices here.
                if (t_count[j] == corrsign) or\
                        ((i <= (arr_len-2)) and (t_count[j+1] == corrsign)):
                    continue
                else:
                    # If three consecutive data are 1 or -1,
                    #   the third data point's Tcount would be 1 or -1
                    if np.all(acounts[j:j+3] == corrsign):
                        t_count[j+2] = corrsign
                    else:
                        t_count[j] = corrsign

            nz_tcount = np.where(t_count != 0)[0]    # non-zero Tcounts
            nztc_len = len(nz_tcount)

            # Remove duplicates
            for i, j in enumerate(nz_tcount):
                if (i <= (nztc_len-2)) and (nz_tcount[i+1] == (j+1)):
                    t_count[j] = 0
                elif (i == (nztc_len-1)) and (nz_tcount[i-1] == (j-1)):
                    t_count[j-1] = 0

            return t_count

        ltcounts  = mark_tcount(over_posth_l, acounts['lcount'], pos=True)
        lntcounts = mark_tcount(under_negth_l, acounts['lcount'], pos=False)
        rtcounts  = mark_tcount(over_posth_r, acounts['rcount'], pos=True)
        rntcounts = mark_tcount(under_negth_r, acounts['rcount'], pos=False)
        tcounts = pd.DataFrame(data = {'L':ltcounts + lntcounts,
                                       'R':rtcounts + rntcounts})

        return tcounts

    def _get_mov(self, tcounts):
        """
        A function to get movement counts

        Parameters:
            tcounts: pd.DataFrame
                the output of self._get_tcounts(accmags)

        Returns:
            tmov: pd.DataFrame
                movement counts (L/R)
        """

        def mark_tmov(temp):
            tmov = np.zeros(len(temp))
            # Among non-zero Tcount (-1 or 1) values...
            nz_tcount = np.nonzero(temp)[0]
            for i, j in enumerate(nz_tcount[:-1]):
            # If the difference between the current point and
            #   its subsequent one is greater than 8 data points,
            #   skip the current point.
                if (np.diff([j, nz_tcount[i+1]])[0] > 8) or (tmov[j] == 1):
                    continue
                # If not, if the sign of the two adjacent points differ
                #   (-1 vs. 1 or vice versa; 0 cannot be included because
                #   points attempt nonzeroTC cannot be 0)
                else:
                    if np.sign(temp[j]) != np.sign(temp[nz_tcount[i+1]]):
                        tmov[nz_tcount[i+1]] = 1
                    else:
                        continue
            return tmov

        tmov = tcounts.apply(mark_tmov, axis=0)
        return tmov

    def _raw_mov_kinematics(self, accmags, acounts):
        """
        A function to return kinematic variables

        Parameters:
            accmags: pd.DataFrame
            acounts: pd.DataFrame

        Return:
            kinematics: dict
        """
        lcombined = pd.merge(accmags.lmag,
                             acounts['lcount'],
                             right_index = True,
                             left_index = True)
        rcombined = pd.merge(accmags.rmag,
                             acounts['rcount'],
                             right_index = True,
                             left_index = True)

        lcombined.rename(columns = {'lmag':'accmag', 'lcount':'counts'},
                         inplace=True)
        rcombined.rename(columns = {'rmag':'accmag', 'rcount':'counts'},
                         inplace=True)

        def start_to_end(tmov_series, th_arr, mags_n_cnts):
            """
            This function will provide six pieces of
            movement related information

            Parameters:
                tmov_series: pd.Series
                    output of _get_mov()
                th_arr: list
                    If a accmag value of a datapoint crossed
                    its corresponding threshold, the value is 1
                    Otherwise, 0.
                    ex) self.tderivs.th_crossed['L'].copy()
                mags_n_cnts: pd.DataFrame
                    a data frame that has accmag and counts values

            Returns:
                kinematics: dict
                    a dict of pieces of movement related information
                    - index: second crossing of accmag threshold
                    - how long was a movement
                    - where was the start of a movement
                    - where was the end of a movement
                    - what was the average acceleration per movement
                    - what was the peak acceleration per movement
            """
            tmovidx = np.nonzero(tmov_series.values)[0]
            movinfo = np.zeros((len(tmovidx), 5))   # colnames follow
            colnames = ['MovIdx',
                        'MovLength',
                        'MovStart',
                        'MovEnd',
                        'avepMov',
                        'peakpMov']

            # j is the index of Tmov's.
            # At any j, count will be -1 or 1 and not 0.
            for i, j in enumerate(tmovidx):
                k = -1              # Moving backward...
                while True:
                    # ... to find the data point that crossed the threshold value
                    #   whose sign is opposite to the count at j. This means that
                    #   the baseline (accmag = 0) is crossed once.
                    if np.sign(th_arr[j+k]) ==\
                            -np.sign(mags_n_cnts['counts'][j]):
                        movsidx = int(j+k)  # "MOV"ement "S"tart "INDEX"
                        break
                    k -= 1      # Keep going backwards if a previous attempt failed
                idx2 = 1               # Moving forward...
                while True:
                    # ... to find the data point that either touches
                    # or crosses the baseline one more time.
                    # This time the sign of the data point at the
                    # index [j+l] could be 0 or opposite to that of
                    # the datapoint at the index [j]
                    if np.sign(mags_n_cnts['counts'][j+idx2]) !=\
                            np.sign(mags_n_cnts['counts'][idx2]):
                        moveidx = int(j+idx2)  # "MOV"ement "E"nd "INDEX"
                        break
                    idx2 += 1
                movinfo[i, :] = [moveidx - movsidx + 1,
                                 movsidx,
                                 moveidx,
                                 np.mean(
                                     abs(mags_n_cnts.accmag[movsidx:(moveidx+1)])),
                                 max(abs(mags_n_cnts.accmag[movsidx:(moveidx+1)]))]
            return(dict(zip(colnames, [tmovidx,
                                       movinfo[:,0],
                                       movinfo[:,1],
                                       movinfo[:,2],
                                       movinfo[:,3],
                                       movinfo[:,4]])))

        kinematics_l = start_to_end(self.Tmov.L,
                                    self.tderivs.th_crossed['L'].copy(),
                                    lcombined.copy())
        kinematics_r = start_to_end(self.Tmov.R,
                                    self.tderivs.th_crossed['R'].copy(),
                                    rcombined.copy())
        # Lkinematics and Rkinematics could differ in length
        #   so better return a dictionary.
        kinematics  = {'Lkinematics':kinematics_l, 'Rkinematics':kinematics_r}

        return kinematics

    def _mark_simultaneous(self):
        """
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
        """
        r_dict = self.kinematics['Rkinematics'].copy()
        l_dict = self.kinematics['Lkinematics'].copy()

        # sole_r and sole_l will each show the indices of nonzero RTmov and LTmov elements
        # In other words, locations of bouts
        sole_r = pd.DataFrame(data = {'MovIdx':r_dict['MovIdx'],
                                      'Start':r_dict['MovStart'],
                                      'End': r_dict['MovEnd']})
        sole_l = pd.DataFrame(data = {'MovIdx':l_dict['MovIdx'],
                                      'Start':l_dict['MovStart'],
                                      'End': l_dict['MovEnd']})

        # If RStart is equal to Lstart, the corresponding RMovIdx is
        # bilateral synchronous to the LMovIdx associated with the LStart
        # If RStart is not equal, but RMovIdx is in between LMovStart and LMovEnd,
        # then it's bilateral asynchronous

        def match_idx(sole_1, sole_2, refside = 'R'):
            bilat_syncidx = {}
            bilat_asyncidx = {}
            bilat_total = {}
            if refside == 'L':
                keys = ['LStart', 'RStart']
            else:
                keys = ['RStart', 'LStart']
            for i, mov in enumerate(sole_1.MovIdx):
                for j in range(sole_2.shape[0]):
                    if sole_2.Start[j] < mov < sole_2.End[j]:
                        bilat_total[mov] = i
                        if sole_1.Start[i] == sole_2.Start[j]:
                            # You're storing the row indices of LStart and RStart
                            # This is for the ease of later processing of sole_r and sole_l
                            bilat_syncidx[mov] = dict(zip(keys, [i,j]))
                        else:
                            bilat_asyncidx[mov] = dict(zip(keys, [i,j]))
                        break
            return({'Sync':bilat_syncidx, 'Async':bilat_asyncidx, 'Total':bilat_total})

        # still takes quite a long time...need to make it more efficient
        bilat_r = match_idx(sole_r, sole_l)
        bilat_l = match_idx(sole_l, sole_r, 'L')

        # Return information about simultaneous moves
        return({'RSim':bilat_r, 'LSim':bilat_l})

def make_start_end_datetime(redcap_csv, filename, site):
    """
    A function to make two datetime objects based on the
        entries from the REDCap export

    Parameters:
        redcap_csv: pd.DataFrame
        filename: string
            format: '/Some path/YYYYmmdd-HHMMSS_[identifier].h5'
        site: string
            where the data were collected (ex. America/Guatemala)

    Returns:
        utc_don_doff: list
            site-specific don/doff times converted to UTC time
    """

    # Use the .h5 filename -> search times sensors were donned and doffed
    # A redcap_csv will be in the following format (example below)
    #
    # id    |             filename              | don_t | doff_t
    # ----------------------------------------------------------
    # LL001 | 20201223-083057_LL001_M1_Day_1.h5 | 9:20  | 17:30
    #
    # Split the filename with '/' and take the first (path_removed).
    # Then we concatenate them with the time searched from the REDCap
    #   export file to generate don_dt and doff_dt (dt: datetime object)

    path_removed = filename.split('/')[-1]

    def match_any(string, filename):
        """
        Highly likely, entries of the column: filename in redcap_csv
            could be erroneous. Therefore, to be on the safe side,
            split REDCap entry and check if more than half the splitted items
            are included in the filename.
        """
        result = False
        if isinstance(string, str):
            splitted = re.split('[-:_.]', string)
            lowered = list(map(lambda x:x.lower(), splitted))
            result = np.mean([x in filename.lower() for x in lowered]) > 0.5

        return result

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
    # There are cases where the REDCap values are spurious at best.
    # Sometimes the first time point recorded in sensors could be later
    #   than what's listed as the time_donned in the REDCap file by
    #   few microseconds.
    # This would case the problem later, as the method of the Subject class
    #   [_prep_row_idx] will 'assume' that the opposite is always true,
    #   and calculate: REDCap time_donned - sensor initial time.
    # Of course this will be a negative value, ruining everything; so add
    #   a minute to what's reported on the REDCap time donned.
    # If the time sensors were doffed was early in the morning (ex. 2 AM)
    #   you know that a day has passed.
    # The condition below, however, may not catch every possible exception.
    # Let's hope for the best that everyone removed the sensors
    #   before 2 PM the next day.

    if int(doff_h) < 14 & (abs(int(don_h) - int(doff_h)) < 10):
        doff_dt = doff_temp + timedelta(days=1)
    else:
        doff_dt = doff_temp

    def convert_to_utc(datetime_obj, site):
        local = pytz.timezone(site)
        local_dt = local.localize(datetime_obj, is_dst = None)
        utc_dt = local_dt.astimezone(pytz.utc)
        return utc_dt

    # site-specific don/doff times are converted to UTC time
    utc_don_doff = list(map(lambda lst: convert_to_utc(lst, site), [don_dt, doff_dt]))
    return utc_don_doff
