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
from dataclasses import dataclass, field
import re
from datetime import datetime, timedelta, timezone
from itertools import chain, repeat
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, detrend, firwin, lfilter
from scipy.interpolate import interp1d, CubicSpline
import pytz


@dataclass
class Processed:
    """ Storing measures derived from thresholds """
    accmags: dict
    velmags: dict
    thresholds: dict    # laccth, lnaccth, raccth, rnaccth
    over_accth: dict = field(init=False)
    under_naccth: dict = field(init=False)
    # 'th_crossed' attributes would be
    #  1: over the positive threshold
    # -1: under the negative threshold
    #  0: otherwise
    th_crossed: dict = field(init=False)

    def __post_init__(self):
        if len(self.accmags.keys()) == 2:
            lpos_temp = self.accmags['lmag'] > self.thresholds['laccth']
            lneg_temp = self.accmags['lmag'] < self.thresholds['lnaccth']
            rpos_temp = self.accmags['rmag'] > self.thresholds['raccth']
            rneg_temp = self.accmags['rmag'] < self.thresholds['rnaccth']

            self.over_accth = {'L': lpos_temp, 'R': rpos_temp}
            self.under_naccth = {'L': lneg_temp, 'R': rneg_temp}

            self.th_crossed = {
                    'L': lpos_temp + lneg_temp * -1,
                    'R': rpos_temp + rneg_temp * -1}
        else:
            lpos_temp = self.accmags['umag'] > self.thresholds['accth']
            lneg_temp = self.accmags['umag'] < self.thresholds['naccth']
            self.over_accth = {'U': lpos_temp}  # L: it's actually unidentified
            self.under_naccth = {'U': lneg_temp}
            self.th_crossed = {'U': lpos_temp + lneg_temp * -1}


@dataclass
class SubjectInfo:
    """ Storing miscellaneous information """
    fname: list
    record_times: dict
    fs: int
    timezone: str
    label_r: str | None
    rowidx: list | None
    recordlen: dict


@dataclass
class CalibFileInfo:
    """ Storing miscellaneous information """
    fname: list
    fs: int

class BaseProcess:
    """ Inherited by classes: axivity.Ax6, apdm.OpalV2 """
    def __str__(self):
        return self._name

    def __repr__(self):
        ret = f"{self._name}("
        for k in self._kw:
            ret += f"{k}={self._kw[k]!r}, "
        if ret[-1] != "(":
            ret = ret[:-2]
        ret += ")"
        return ret

    def __eq__(self, other):
        if isinstance(other, type(self)):
            return self._kw == other._kw
        else:
            return False

    def __init__(self, **kwargs):
        """ BaseProcess will be inherited """
        self._name = self.__class__.__name__
        self._kw = kwargs

        # Placehodler - will be updated
        self.info = SubjectInfo(
            fname = ['no_filename'],
            record_times = {'L': None, 'R': None},
            fs = 0,
            timezone = '',
            label_r = None,
            rowidx = None,
            recordlen = {'L': 0, 'R': 0})

        self.measures = Processed(
            accmags = {'lmag': np.array([0,0,0,0,0]), 
                'rmag': np.array([0,0,0,0,0])},
            velmags = {'lmag': np.array([0,0,0,0,0]),
                'rmag': np.array([0,0,0,0,0])},
            thresholds = {'laccth': 1,
                'lnaccth': -1,
                'raccth': 1,
                'rnaccth': -1})

    # This function will calculate the actual date and time from the time recorded in the sensor
    def _calc_datetime(self, timestamp):
        """
        A function to convert a timestamp into a datetime instance

        Parameters:
            timestamp: int
                a timestamp in units of microseconds since 1970-1-1-0:00 UTC
                For Opal sensors, you can directly transform this to a datetime object
                For Axivity sensors, dtype is numpy.datetime64.

        Returns:
            datetime_utc: datetime
                a datetime instance whose value is converted from the time_stamp
        """
        if self._name in ['OpalV2', 'OpalV1', 'OpalV2Single']:
            ts = timestamp/1e6
            datetime_utc = datetime.fromtimestamp(ts, timezone.utc)
        elif self._name in ['Ax6', 'Ax6Single']:
            ts = np.array(timestamp*1e3, dtype='datetime64[ms]')
            datetime_local = pd.to_datetime(ts).tz_localize(self.info.timezone)
            datetime_utc = datetime_local.astimezone(pytz.utc)
        return datetime_utc

    def _get_mag(self, sensors, row_idx=None, det_option='median'):
        """
        A function to calculate the magnitude of a provided dtype.

        Parameters:
            sensors: dict
                itmes are three-column arrays of acc/gyro data
            row_idx: list | None
                Recording start and end indices (each int, for OpalV2)
            det_option: str
                'median' or 'customfunc'
                Subtract median or use some custom method to detrend

        Returns:
            out: dict
                detrended acceleration or angular velocity norm
                keys = ['lmag', 'rmag']
        """
        if det_option not in ['median', 'customfunc']:
            det_option='median'
            print("Unknown detrending option - setting it to [median]")

        # Axivity sensors differ in length.
        # Let's create another function and use map()
        def linalg_norm(arr, row_idx):
            nrow = arr.shape[0]

            if row_idx is None:
                row_idx = list(range(nrow)) # targeting the entire dataset
            mag = np.linalg.norm(arr[row_idx], axis=1)
            return mag

        # Use np.linalg.norm... amazing!
        mags = map(linalg_norm,
                list(sensors.values()),
                [row_idx, row_idx])

        # MATLAB's detrend function is not used,
        #   so we can consider that the default option
        #   for detrending the data is subtracting the median
        if det_option == 'median':
            out = map(lambda x: x - np.median(x), mags)
        else:
            # detrend every 1000 seconds (20000 data points)
            # Now you have to convert the map objects to lists
            lmaglist, rmaglist = list(mags)
            tempx, tempy = [], []
            binsize = np.ceil(len(lmaglist)/20000)
            for i in range(int(binsize)):
                tempx.extend(detrend(lmaglist[(20000*i):20000*(i+1)]))
                tempy.extend(detrend(rmaglist[(20000*i):20000*(i+1)]))

            out = [np.array(tempx), np.array(tempy)]

        # if only one sensor is processed
        if len(sensors.keys())-1:
            return dict(zip(['lmag', 'rmag'], out))
        else:
            return({'umag':list(out)[0]})

    @staticmethod
    def remove_offset(arr, offsets):
        """
        a function that removes offsets from measurements

        Parameters:
            arr: numpy.array
                raw accelerometer data
            offsets: numpy.array
                [xgain, xoffset, ygain, yoffset, zgain, zoffset]
                list of axis/orientation specific offsets

        Returns:
            an array - offset removed accelerometer data
        """
        """
        xoffs = np.array([offsets[0], 0, offsets[1]])
        yoffs = np.array([offsets[2], 0, offsets[3]])
        zoffs = np.array([offsets[4], 0, offsets[5]])
        xidx = np.sign(arr[:,0]).astype(int)+1
        yidx = np.sign(arr[:,1]).astype(int)+1
        zidx = np.sign(arr[:,2]).astype(int)+1
        newx = [x + xoffs[xi] for x, xi in zip(arr[:,0], xidx)]
        newy = [y + yoffs[yi] for y, yi in zip(arr[:,1], yidx)]
        newz = [z + zoffs[zi] for z, zi in zip(arr[:,2], zidx)]
        """
        """
        newx = [(x - offsets[1])/offsets[0] for x in arr[:,0]]
        newy = [(y - offsets[3])/offsets[2] for y in arr[:,1]]
        newz = [(z - offsets[5])/offsets[4] for z in arr[:,2]]
        """
        newx = [x/offsets[0] for x in arr[:, 0]]
        newy = [y/offsets[1] for y in arr[:, 1]]
        newz = [z/offsets[2] for z in arr[:, 2]]

        #return np.vstack((newx, newy, newz)).T
        return np.column_stack((newx, newy, newz))

    @staticmethod
    def low_pass(fc, fs, arr, window='hamming'):
        """
        low-pass filter the data, using the provided
        cut-off frequency (fc).
        As it's a simple first-order low-pass filter,
        limit the window choice to hamming.
        """
        h = firwin(2, cutoff=fc, window=window, fs=fs)
        return lfilter(h, 1, arr)

    @staticmethod
    def local_to_utc(timestamp, study_tz):
        local = datetime.fromtimestamp(timestamp,
                pytz.timezone(study_tz))
        return local.astimezone(pytz.utc)

    @staticmethod
    def resample_to(timearr, arr, intp_option, re_fs):
        """
        Downsample the data to a resampling frequency (re_fs).
        If rsf = 20(Hz), it's the sampling frequency of
        Opal V2. Interpolation option (intp_option) is provided
        """
        assert(intp_option in ['cubic', 'linear']),\
                "intp_option should be cubic or linear"
        if intp_option == 'cubic':
            intp = CubicSpline(timearr, arr, axis=0)
        else:
            intp = interp1d(timearr, arr, axis=0)

        nt = np.round(np.arange(timearr[0], timearr[-1], 1/re_fs), 2)
        return intp(nt)

    def _calc_ind_threshold(self, maxprop):
        """
        ind_threshold = mean(peak_heights) - C * std(peak_heights)

        C = 1       Smith et al. (2015)
            0.5     Trujillo-Priego et al. (2017), ivan=True
        """
        return np.mean(maxprop['peak_heights'])\
                - np.std(maxprop['peak_heights'])

    # This is equivalent to the MATLAB script
    def _get_ind_acc_threshold(self, accmags, reject = 3.2501, height = 1.0):
        """
        A function to find individual thresholds

        Parameters:
            accmags: dict
                detrended acceleration norms (L/R)
            reject: float, optional
                a cut-off; values below this number will be included
            height: float, optional
                minimal height that defines a peak

        Returns:
            thresholds: dict
                laccth: left positive threshold
                lnaccth: left negative threshold
                raccth: right positive threshold
                rnaccth: right negative threshold
        """
        mags2 = accmags.copy()
        posvals = map(lambda x: [u if 0 < u < reject else 0 for u in x],
                      mags2.values())
        negvals = map(lambda x: [abs(u) if -reject < u < 0 else 0 for u in x],
                      mags2.values())

        pnpks = map(lambda x: [find_peaks(u, height=height)[1] for u in x],
                    [posvals, negvals])
        accths = map(self._calc_ind_threshold, chain(*pnpks))

        if len(accmags.keys())-1:
            tkeys = ['laccth', 'raccth', 'lnaccth', 'rnaccth']
            signvec = [1,1,-1,-1]
        else:
            tkeys = ['accth', 'naccth']
            signvec = [1,-1]
        return dict(zip(tkeys, np.multiply(signvec, list(accths))))

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
        # index where button is first pressed
        # This is the option for Opal V1 only.
        try:
            indexed1 = np.where(sensorobj['ButtonStatus'][:]==1)[0][0]
        except:
            indexed1 = 0
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
            if indexed1:
                # overwrite whatever's given as the start of the recording
                in_en_dts[0] = self._calc_datetime(sensorobj['Time'][indexed1])
            d_in_micro = list(map(
                lambda x: x - self._calc_datetime(sensorobj['Time'][0]), in_en_dts))
            convert = 1e6    # conversion constant: 1 second = 1e6 microseconds
            microlen = 0.05 * convert    # duration of a data point in microseconds
            poten = np.ceil((d_in_micro[1].days*86400*convert +    # 86400 = 24*3600
                d_in_micro[1].seconds*convert +
                d_in_micro[1].microseconds)/microlen)\
                                   .astype('int')
            if poten < sensorobj['Time'].shape[0]:
                indices = list(map(
                    lambda x: round(
                        (x.days*86400*convert +
                            x.seconds*convert + x.microseconds)/microlen),
                    d_in_micro))
            else:
                indices = []
                indices.extend([round((d_in_micro[0].seconds*convert
                                      + d_in_micro[0].microseconds)/microlen),\
                                sensorobj['Time'].shape[0]-1])

            # This will be one input to self._get_mag
            row_idx = list(range(indices[0], indices[1]))
        else:
            if indexed1:
                print("No recording start and end time provided. \
                        Data starts from the first click of the button.")
                row_idx = list(range(indexed1, sensorobj['Time'].shape[0]-1))
            else:
                row_idx = None
                print("No recording start and end time provided. \
                        Analysis done on the entire recording")
        return row_idx

    def _get_count(self):
        """
        A function to get counts. The count of a data point is
            the sign of the acceleration norm at the data point.
        If the angular velocity norm at the data point is less than 0,
            then the count is also 0.

        Parameters:
            None

        Returns:
            acounts2: list
                count values (L/U/R)
        """
        raw_counts = map(np.sign, self.measures.accmags.values())
        acounts = [np.multiply(x, y) for x, y in\
                zip(raw_counts,
                    map(lambda x: np.greater(x, 0),
                        self.measures.velmags.values()))]
        return acounts

    def _get_cntc(self, side='L'):
        """
        A function to get counts and tcounts

        Parameters:
            None

        Returns:
            tcounts: dict
                keys: count, tcount
        """
        acounts = self._get_count()    # accmags or angvels

        # Let's start with collecting all acc values that went over the threshold
        # The output of np.where would be a tuple - so take the first value
        # I do this to reduce the preprocessing time...
        #temp_posth_l = np.where(self.measures.over_accth['L'])[0]
        #temp_posth_r = np.where(self.measures.over_accth['R'])[0]
        #temp_negth_l = np.where(self.measures.under_naccth['L'])[0]
        #temp_negth_r = np.where(self.measures.under_naccth['R'])[0]

        temp = map(lambda x: np.where(x)[0],
                [self.measures.over_accth[side],
                    self.measures.under_naccth[side]])

        #temp_r = map(lambda x: np.where(x)[0],
        #        [self.measures.over_accth['R'],
        #            self.measures.under_naccth['R']])

        # angular velocity should be taken into account...
        # let's just make sure that the detrended angvel[i] > 0
        if side == 'R':
            angvel_gt = np.nonzero(acounts[1])[0]
        else:
            angvel_gt = np.nonzero(acounts[0])[0]

        over_posth, under_negth = map(lambda x:
                np.intersect1d(x, angvel_gt), temp)
        #over_posth_r, under_negth_r = map(lambda x:
        #        np.intersect1d(x, angvel_gt_r), temp_r)

        def mark_tcount(over_th_arr, acounts, pos=True):
            """
            Parameters:
                over_th_arr: np.array
                    indices of data points over a threshold
                    (ex: over_posth_l)
                acounts: dict
                    output of self._get_count(accmags, angvels)

            Returns:
                tcount: np.array
                    nonzero counts that crossed
                    a positive or negative threshold
            """
            corrsign = -1
            if pos:
                corrsign = 1
            arr_len = len(acounts)
            t_count = np.zeros(arr_len)
            for j in over_th_arr:
                # "over_th_arr" has the indices of the acceleration values
                #   that are over a threshold (left or right).
                # [j] gives one of those indices.
                # It could be that the Tcount value at the current index
                #   may have been set by the previous index
                #   (ex. [j-2] satisfied the "else" condition so Tcount[j] = 1 or -1)
                # If so, skip to the next index.
                # Also, if the next index is not zero, then the current
                #   Tcount[j] is considered as a redundant count and
                #   marked off (this is from the original MATLAB code).
                #   So we can skip such indices here.
                if all((j < (arr_len-2), all(t_count[j:j+2] != corrsign))):
                    if all(acounts[j:j+3] == corrsign):
                        t_count[j+2] = corrsign
                    else:
                        t_count[j] = corrsign

            #nz_tcount = np.where(t_count != 0)[0]    # non-zero Tcounts
            nz_tcount = np.nonzero(t_count)[0]
            nztc_len = len(nz_tcount)

            # Remove duplicates
            for i, j in enumerate(nz_tcount):
                if (i <= (nztc_len-2)) and (nz_tcount[i+1] == (j+1)):
                    t_count[j] = 0
                elif (i == (nztc_len-1)) and (nz_tcount[i-1] == (j-1)):
                    t_count[j-1] = 0

            return t_count

        if side == 'R':
            usecount = acounts[1]
        else:
            usecount = acounts[0]
        ttcounts = mark_tcount(over_posth, usecount, pos=True)
        tntcounts = mark_tcount(under_negth, usecount, pos=False)
        #ltcounts  = mark_tcount(over_posth_l, acounts[0], pos=True)
        #lntcounts = mark_tcount(under_negth_l, acounts[0], pos=False)
        #rtcounts  = mark_tcount(over_posth_r, acounts[1], pos=True)
        #rntcounts = mark_tcount(under_negth_r, acounts[1], pos=False)

        #return dict(zip(['L', 'R'], [[acounts[0], ltcounts+lntcounts],
        #                             [acounts[1], rtcounts+rntcounts]]))
        return {side:[usecount, ttcounts + tntcounts]}

    def get_mov(self, side='L'):
        """
        A function to get movement counts

        Parameters:
            side: str
                'L'eft, 'R'ight, or 'U'nilateral

        Returns:
            tmov: list
                movement counts (L/R)
        """

        assert(side in ['L', 'R', 'U']), "side should be 'L', 'R', or 'U'"

        # tcounts is a dictionary (keys: 'L', 'R')
        # Each value is a list of TWO lists (counts, tcounts)
        tcounts = self._get_cntc(side)

        # index | count | tcount | th_crossed
        arr_a = np.column_stack((np.arange(len(tcounts[side][0])),
                                 tcounts[side][0],  # counts
                                 tcounts[side][1],  # tcounts
                                 self.measures.th_crossed[side])).astype(int)
        arr_b = arr_a[np.nonzero(arr_a[:,2])[0], :]
        movidx = np.zeros((arr_b.shape[0], 3), dtype=int)

        maxmov_dt = 1.5

        # arr_b[0,] would be the row with the first nonzero tcount.
        # Smith et al. (2015) has this quote:
        #   "The start of a movement was defined as simultaneous accceleration
        #    above a magnitude threshold and angular velocity greater than 0."
        # So I originally thought that the start of a movement should be
        #   a data point with tcount value 1 or -1
        # However, a typicaly movement's acceleration profile would rather be
        #   sinusoidal. Probably we could search backwards a little more and
        #   define the "start of a movement" as the data point that precedes
        #   a oint with the nonzero tcount (first_tc) in time and has the same
        #   count value to that of first_tc.
        for i in range(arr_b.shape[0]-1):
            pairdiff = np.diff(arr_b[i:i+2,:], axis=0).ravel()
            # Two Tcounts are different (-1 vs. 1)
            # Rolling back to the version: Dev 29, 2022
            # Apr 26, 2023:
            #   If Ax6 (sampling rate: 25 samples/sec) is used,
            #   the difference in data points between the two opposite-sign
            #   tcounts should be 8*25/20 = 10.
            #   Also, a mov duration is no longer than 1.5 sec.
            #   OpalV2 at 20 samples/sec: 30 points
            #   Ax6 at 25 samples/sec: 38 points (round up 37.5)
            tcount_diff = int(8*self.info.fs/20)
            if all((pairdiff[2] != 0, pairdiff[0] <= tcount_diff)):
                sidx = arr_b[i+1, 0]     # second threshold crossing
                if arr_b[i,3] == arr_b[i,2]:  # if th_cross == t_count
                    first_tc = arr_b[i,0]
                else:
                    if arr_a[arr_b[i,0]-1,3] == arr_b[i,2]:
                        first_tc = arr_b[i,0]-1
                    else:
                        first_tc = arr_b[i,0]-2
                # Feb 9, 2023: fstep = 15 - sidx + first_tc
                # Apr 26, 2023:
                #   15 (=30/2) was increased from 8.
                #   30 is 1.5*sampling rate (20 for OpalV2)
                fstep = int(maxmov_dt*self.info.fs/2 + 1) - sidx + first_tc
                if all((0 < fstep < first_tc,
                    arr_a[(first_tc-1),3] == arr_a[first_tc,3])):
                    # diffcnt is the index of the point whose count value is
                    # different from that of first_tc, implying that
                    # the baseline is 'crossed'. This should NOT happen.
                    # Therefore, find one point behind sidx and set that as
                    # the start of a movement
                    # (1/3/22) Let's revise so that diffcnt would be
                    # the index of the FIRST point whose cross_th value is
                    # the same as that of first_tc and is within k data points
                    # from the first_tc where k = max(3, fstep)
                    # k = min(3, fstep)
                    # (2/2/23) No. Use fstep, and run another while loop
                    k = 0
                    while (arr_a[(first_tc-k),3] == arr_a[first_tc,3]):
                        k += 1
                        if k > fstep:
                            break
                    movstart = first_tc - k + 1
                    #diffcnt = np.where(arr_a[(first_tc-fstep):first_tc,1] !=
                    #        arr_b[i,2])[0]
                    #if diffcnt.size:
                    #    movstart = first_tc - fstep + diffcnt[-1] + 1
                    # If you don't find a point whose count value is different,
                    # simply go back by the amount of step
                    #else:
                    #    movstart = first_tc - fstep
                else:
                    movstart = first_tc

                addi = int(movstart + np.ceil(maxmov_dt*self.info.fs))
                try:
                    # movend: the first point that has the "count" value
                    # whose sign is the opposite to that of sidx's 
                    # "tcount" value
                    movend = np.where(
                            arr_a[sidx:addi,1] == -arr_a[sidx,2])[0][0]
                    # If you move by movend from sidx and that point crossed
                    # a threshold (-1 or 1, nonzero),
                    # check one point further, and see if that point also crossed
                    # the same threshold. If not, the end of a movement
                    # should be one back. Otherwise, take that as the movend.
                    # The end of a movement should be one back
                    # (2/2/23) What if you forget about it, and just take it?
                    movidx[i] = [movstart, sidx, sidx + movend]
                except:
                    continue

        movidx_nz = movidx[np.nonzero(movidx[:,0])[0],:]

        for i in range(1, movidx_nz.shape[0]):
            if not any(movidx_nz[i-1,:]):
                j=i
                while j < movidx_nz.shape[0]:
                    if movidx_nz[j,0] < movidx_nz[i-2,2]:
                        movidx_nz[j,:] = np.zeros(3)
                        j+=1
                    else:
                        break
            elif movidx_nz[i,0] < movidx_nz[i-1,2]:
                movidx_nz[i,] = np.zeros(3)

        movidx_nz2 = movidx_nz[np.nonzero(movidx_nz[:,0])[0],]

        #tmov = tcounts.apply(mark_tmov, axis=0)
        #return list(map(mark_tmov, tcounts.values()))
        return movidx_nz2

    def acc_per_mov(self, side='L', movmat=None):
        """
        A function to calculate average acceleration per movement
            and the peak acceleration per movement

        Parameters:
            side: str
                'L' ,'R', or 'U'
            movmat: None | np.array
                matrix that stores movements' indices

        Returns:
            acc_arr: numpy ndarray
                [mov start | ave accmag for mov[i] | peak accmag for mov[i]]
        """
        assert(side in ['L', 'R', 'U']), "side should be 'L', 'R', or 'U'"

        if movmat is not None:
            movidx = movmat
        else:
            movidx = self.get_mov(side)

        if side == 'L':
            accvec = np.abs(self.measures.accmags['lmag'].copy())
        elif side == 'U':
            accvec = np.abs(self.measures.accmags['umag'].copy())
        else:
            accvec = np.abs(self.measures.accmags['rmag'].copy())

        acc_arr = np.array([[x, np.mean(accvec[x:y]), max(accvec[x:y])]\
                for x, y in zip(movidx[:,0], movidx[:,2]+1)])

        return acc_arr

    def plot_segment(self, time_passed, duration=20, side='L', movmat=None):
        """
        A function to let user visually check movement counts

        Parameters:
            time_passed: float
                time passed from the start of the recording in seconds

            duration: int
                duration in seconds, default is 20

            side: str
                'L', 'R', or 'U'

        Returns:
            a diagnostic figure to check movement counts
        """
        assert(side in ['L', 'R', 'U']), "side should be 'L', 'R', or 'U'"

        if movmat is not None:
            movidx = movmat
        else:
            movidx = self.get_mov(side)

        if side == 'L':
            labels = ['lmag', 'laccth', 'lnaccth']
        elif side == 'U':
            labels = ['umag', 'accth', 'naccth']
        else:
            labels = ['rmag', 'raccth', 'rnaccth']

        # Jan 31, 23 / WHY did I do this? (checking rowidx is None)
        # Feb 09, 23 / I think this can be remove
        #if self.info.rowidx is not None:
            #new_t = self.info.record_times[0]\
            #        + timedelta(seconds=time_passed)
            #end_t = self.info.record_times[1]
        startidx = int(time_passed * self.info.fs)
        endidx = startidx + int(duration  * self.info.fs) + 1
        mov_st = np.where(movidx[:,0] >= startidx)[0]
        mov_fi = np.where(movidx[:,2] <= endidx)[0]

        _, ax = plt.subplots(1)
        accline, = ax.plot(self.measures.accmags[labels[0]][startidx:endidx],
                marker='o', c='pink', label='acceleration')
        pthline = ax.axhline(y=self.measures.thresholds[labels[1]],
                c='k', linestyle='dotted', label='positive threshold')
        nthline = ax.axhline(y=self.measures.thresholds[labels[2]],
                c='k', linestyle='dashed', label='negative threshold')
        ax.axhline(y=0, c='r')  # baseline
        # If Ax6 or Active, convert from 1 deg/s to 0.017453 rad/s
        #rad_convert = 0.017453 if self._name in ['Ax6', 'Ax6Single'] else 1
        velline, = ax.plot(
                self.measures.velmags[labels[0]][startidx:endidx],
                c='deepskyblue', linestyle='dashdot', label='angular velocity')
        ax.legend(handles = [accline, pthline, nthline, velline])

        if all((mov_st.size, mov_fi.size)):
            if mov_st[0] == mov_fi[-1]:
                mov_lens = movidx[mov_st[0],2] - movidx[mov_st[0],0]
            else:
                fi2 = mov_fi[-1] + 1
                mov_lens = movidx[mov_st[0]:fi2,2] - movidx[mov_st[0]:fi2,0]
            if mov_lens.size:
                if mov_lens.size == 1:
                    hull = np.arange(movidx[mov_st[0],0],
                                     movidx[mov_st[0],0] + mov_lens + 1)
                    hl, = ax.plot(hull - startidx,
                            self.measures.accmags[labels[0]][hull],
                            c='g', linewidth=2, label='movement')
                    ax.legend(handles = [accline, pthline, nthline,
                                         velline, hl])
                else:
                    hull = [np.arange(x, (x+mov_lens[i]+1))\
                            for i, x in enumerate(movidx[mov_st[0]:fi2,0])]
                    hl, = ax.plot(hull[0] - startidx,
                            self.measures.accmags[labels[0]][hull[0]],
                            c='g', linewidth=2, label='movement')
                    ax.legend(handles = [accline, pthline, nthline,
                                         velline, hl])
                    for j in range(1, len(hull)):
                        ax.plot(hull[j]-startidx,
                                self.measures.accmags[labels[0]][hull[j]],
                                c='g', linewidth=2)
        title = f"{duration}s from "\
                f"{(self.info.record_times[side][0] + timedelta(seconds=time_passed)).ctime()}"\
                f" UTC\n(recording ended at {self.info.record_times[side][1].ctime()} UTC)"
        ax.set_title(title)
        ax.set_xlabel("Time since onset (sec)")
        ax.set_ylabel("Acc. magnitude (m/s^2)")
        # x tick labels to indicate "seconds"
        def numfmt(x, pos):
            s = '{}'.format(x/self.info.fs)
            return s
        ax.xaxis.set_major_formatter(numfmt)

        plt.show()


def get_axis_offsets(ground_gs):
    """
    A function to calculate axis/orientation specific biases

    Parameters:
        grounds_gs: list
            measured gravitational accelerations in the following order/
            x-, x+, y-, y+, z-, z+

    Returns:
        list of axis/orientation specific offsets
    """
    # If it is a list of TWO lists...
    if len(ground_gs) == 2:
        return list(map(lambda x: np.array([-1,1,-1,1,-1,1]) - x, ground_gs))
    else:
        return np.array([-1,1,-1,1,-1,1]) - ground_gs

def get_axis_offsets_v2(ground_gs):

    def calc_error(arr, idxlist):
        """ Return: a list of three gain values """
        # idx as a list...
        errs = list()
        for idx in idxlist:
            gain = np.diff(arr[idx[0]:idx[1]])[0]/2
            #offs = arr[idx[0]]+gain  # offset will be removed separately
            #errs.extend([gain, offs])
            errs.append(gain)
        return errs

    if len(ground_gs) == 2:
        # Left and Right
        lxyz, rxyz = map(calc_error,
                         ground_gs,
                         repeat([[0, 2], [2, 4], [4, 6]], 2))
        return np.array((lxyz, rxyz))
    else:
        lxyz = calc_error(ground_gs, [[0, 2], [2, 4], [4, 6]])
        return np.array(lxyz)

def time_asleep(movmat, recordlen, fs, t0=0, t1_user=None):
    """
    A function to approximate the time the infant was inactive

    Parameters:
        movmat: np.array
            indices of movements
        recordlen: int
            length of the recording
            ex) info.recordlen['R'] of a class object
        fs: int
            sampling rate
        t0: int
            initial time, default is 0
        t1_user: int|None
            user defined end time, default is None

    Returns:
        time_sleep: int
            estimated inactive times for left and right
            (unit: data point)
    """
    # time_asleep calculated based on the original MATLAB code...
    # If 6000 consequent data points (5 min) show:
    #   1) angular velocity less than 0.3
    #   2) acceleration norm less than 1.001
    # then this period is considered as the baby sleeping or inactive.
    # Technically, this is a bit weird... because you would 'ignore'
    # movements with negative peaks. The second condition should be applied
    # to the absolute values of the detrended acc magnitudes.

    # For Axivity sensor recording, there are just too many data points.
    # This approach would not work.

    # Another method to calculate [time asleep] is introduced in
    # Trujillo-Priego et al. (2017) - less than 3 movements in 5 min.
    # An approximation of this method could be:
    #
    #   a'  a  1   bc 2           d   3           4
    #   -----[-+-]--[-+--]---------[--+-]------[--+-]-
    #
    # Let's suppose that 1,2,3,4 are the movement counts (+).
    # Their starts and ends are marked by [ and ].
    # First, you define an anchor and a target.
    # The anchor and the target are the indices of movements separated by
    # no more than two movements.
    # From the image above, the first anchor is 1 and the first target is 2.
    # You then calculate the distance (D):
    #   one point before the start of a target (b) - start of an anchor (a)
    # If the anchor is 1, then instead of a, use a'.

    # i. If D > 6000, you increase [time asleep] by D
    #    Then the new anchor will be the previous target (2), and the new
    #    target will be target + 1 (3).
    #    D is newly defined: d-c (refer to the image above)
    #
    # ii. If FALSE, increase the target by 1 and check if the new distance
    #     b' - a' is greater than 6000 (refer to the imave below).
    #     If that's TRUE, do i. and move on.
    #     If that's FALSE, increase the target one more time do the same.
    #     (D = b'' - a'; D > 6000?)
    #
    #   a'  a  1   b  2           b'  3       b'' 4
    #   -----[-+-]--[-+--]---------[--+-]------[--+-]-
    #
    #     If you still see that D = b'' - a' < 6000, increase the anchor
    #     by 1, and repeat ii.
    #     If D > 6000, do i and move on.

    anchor = 0
    target = 1
    time_sleep = 0
    mvcnt = movmat.shape[0]
    len_5_min = 5*60*fs # 6000 for fs=20 samples/sec

    while target < (mvcnt-1):
        if target-anchor > 3:
            anchor += 1
            t0 = movmat[anchor][0]

        t1 = movmat[target][0]-1

        if t1-t0 > len_5_min:
            time_sleep += t1-t0
            anchor = target
            t0 = movmat[anchor][0]
            target += 1
        else:
            target += 1
    # When you break out from the loop, make t0 the end of the last move
    t0 = movmat[-1][2] + 1
    if t1_user is not None:
        t1 = t1_user-1
    else:
        t1 = recordlen-1
    if t1-t0 > len_5_min:
        time_sleep += t1-t0

    return time_sleep

def cycle_filt(movmat, threshold=4, fs=20):
    """
    A function to detect and filter out movements look highly cyclical

    Parameters:
        movmat: np.array
            an array of movement indices
        threshold: int
            difference between the end of one movement and the start of
            the next movement; default is 4
        fs: int
            sampling frequency

    Returns:
        movmat_del: np.array
            an array of movement indices / cyclical movements rejected
    """
    to_del = []
    i=0
    while i < (movmat.shape[0]-2):
        diff = movmat[i+1,0] - movmat[i,2]
        if diff <= threshold:
            j=i+1   # j could be the last mov idx
            while j < (movmat.shape[0]-1):
                if (movmat[j+1,0]-movmat[j,2]) <= threshold:
                    j+=1
                else:
                    break
            # If more than 8 'cycles' are observed, discard the movements
            # (should it be ten or more?)
            # Discussion with Dr. Smith (2/13/23) -> increasing the number
            # testing - doubling the number to 16? -> 40
            if j-i > 39:
                to_del.extend(range(i, j+1))
                # further remove 2 movements within 12 seconds of j'th movement
                # Why two? just...
                k=j+1
                counter = 0
                while all((k < (movmat.shape[0]-1), counter < 3)):
                    if (movmat[k,0]-movmat[j,2]) < 12*fs:
                        to_del.append(k)
                        counter+=1
                        k+=1
                    else:
                        break
                i=k
            else:
                i=j+1
        else:
            i+=1
    movmat_del = np.delete(movmat, to_del, 0)
    return movmat_del

def rate_calc(lmovmat, rmovmat, thr, recordlen, fs):
    """
    A function to calculate movement rate and return
    other relevant measures, such as sleep time

    Parameters:
        lmovmat, rmovmat: np.array
            movement matrices for the left/right sides
        thr: int
            equal to the threshold value in cycle_filt function.
            difference between the end of one movement and the start of
            the next movement; 4 for 20 S/s, 5 for 25 S/s (= 0.2 seconds)
        recordlen: int
            length of a recording; use '.info.recordlen.values()'
        fs: int
            Sampling frequency; if the movement matrices are downsampled,
            put in the frequency used for resampling.

    Returns:
        a dictionary with the following items:
            lrate, rrate: movement rates
            lrec_hr, rrec_hr: recording length in hours
            lsleep_hr, rsleep_hr: sleep hours
    """
    lmovs_del, rmovs_del = map(cycle_filt,
            [lmovmat, rmovmat],
            repeat(thr, 2))
    lsleep, rsleep = map(time_asleep,
            [lmovs_del, rmovs_del],
            recordlen,
            repeat(fs,2))
    lsleep_5m, rsleep_5m = map(lambda x: x/(60*fs) - np.mod(x/(60*fs), 5),
            [lsleep, rsleep])
    lrec_hr = recordlen[0]/(3600*fs)
    rrec_hr = recordlen[1]/(3600*fs)
    lsleep_hr, rsleep_hr = map(lambda x: x/60, [lsleep_5m, rsleep_5m])
    lmovrate = lmovs_del.shape[0]/(lrec_hr-lsleep_hr)
    rmovrate = rmovs_del.shape[0]/(rrec_hr-rsleep_hr)
    return {'lrate':lmovrate, 'rrate':rmovrate,
            'lrec_hr':lrec_hr, 'rrec_hr':rrec_hr,
            'lsleep_hr':lsleep_hr, 'rsleep_hr':rsleep_hr}

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

    def convert_to_utc(datetime_obj, site):
        local = pytz.timezone(site)
        local_dt = local.localize(datetime_obj, is_dst = None)
        utc_dt = local_dt.astimezone(pytz.utc)
        return utc_dt

    idx = redcap_csv.filename.apply(lambda x: match_any(x, filename))

    don_n_doff = redcap_csv.loc[np.where(idx)[0][0], ['don_t', 'doff_t']]
    #don_n_doff = times.values[0]  # times is a Pandas Series

    # 2/10/23, don_n_doff could be NaN, meaning that people forgot to
    # enter times to the REDCap. NaN is "float".
    # If that's the case, return None
    if isinstance(don_n_doff[0], float):
        utc_don_doff = None
    else:
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

    nextday = any((all((int(doff_h) < 14, 
        abs(int(don_h) - int(doff_h)) < 10)), int(doff_h) < 12))

    if nextday:
        doff_dt = doff_temp + timedelta(days=1)
    else:
        doff_dt = doff_temp

    # site-specific don/doff times are converted to UTC time
    utc_don_doff = list(map(lambda lst: convert_to_utc(lst, site), [don_dt, doff_dt]))
    return utc_don_doff
