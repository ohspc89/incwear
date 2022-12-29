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
from itertools import chain
import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, detrend
import pytz
#filename = '/Users/joh/Downloads/20220322-084542_LL020_M2_D2.h5'

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
        lpos_temp = self.accmags['lmag'] > self.thresholds['laccth']
        lneg_temp = self.accmags['lmag'] < self.thresholds['lnaccth']
        rpos_temp = self.accmags['rmag'] > self.thresholds['raccth']
        rneg_temp = self.accmags['rmag'] < self.thresholds['rnaccth']

        self.over_accth = {'L': lpos_temp, 'R': rpos_temp}
        self.under_naccth = {'L': lneg_temp, 'R': rneg_temp}

        self.th_crossed = {
                'L': lpos_temp + lneg_temp * -1,
                'R': rpos_temp + rneg_temp * -1}
@dataclass
class SubjectInfo:
    """ Storing miscellaneous information """
    fname: str
    record_times: list
    label_r: str
    rowidx: list | None

class opalv2:
    """
    A class that will store (preliminarily) processed data
        recorded in the OPAL V2sensors
    """
    def __init__(self, filename, in_en_dts, label_r):

        with h5py.File(filename, 'r') as h5file:
            # Sensors has TWO members.
            # Each subgroup has a name with the format: XI-XXXXXX
            # The number after the dash is the Case ID
            sensors = h5file['Sensors']
            sensorids = list(sensors.keys())

            # We need to find out which sensor was attached to which leg.
            # First, we read the label of the second Case ID (sensorlabel)
            sensorlabel = sensors[sensorids[1]]\
                    ['Configuration']['Config Strings'][0][2].decode()
            # Second, we compare sensorlabel with the user provided label
            # for the "right" (ex. 'right', 'R', 'Right_leg', 'derecho'...)
            # If the match is True, then ridx = 1, or the second Case ID.
            # If the match if False, then ridx = 0, the first Case ID.
            ridx = label_r.lower() in sensorlabel.lower()
            sensordict = {'L': sensors[sensorids[not ridx]],
                          'R': sensors[sensorids[ridx]]}

            # Index with the recording start time(in) and the end time(en)
            # Note that in_en_dts are given in UTC as well
            rowidx = self._prep_row_idx(sensordict['L'], in_en_dts)

            # Accelerometer and Gyrocscope norms / this part takes some time
            accmags = self._get_mag(sensordict, 'acc', rowidx)
            velmags = self._get_mag(sensordict, 'gyro', rowidx)

            # Storing the acceleration threshold values:
            #   [laccth, lnaccth, raccth, rnaccth]
            thresholds = self._get_ind_acc_threshold(accmags)

            if rowidx is not None:
                record_ts = [in_en_dts[0] + timedelta(seconds=rowidx[0]*0.05),
                             in_en_dts[1]]

            self.info = SubjectInfo(fname = filename,
                                    record_times = record_ts,
                                    label_r = label_r,
                                    rowidx = rowidx)

            self.measures = Processed(accmags = accmags,
                                      velmags = velmags,
                                      thresholds = thresholds)

            #acounts = self._get_count(accmags, angvels)

            # Tmov will be 1 if acceleration value
            #  crossed thresholds (neg or pos) two times.
            # Why have it as an instance attribute? For plotting, of course.
            #self.tcounts = self._get_tcount(acounts)
            #self.Tmov = self._get_mov(self.tcounts)

            # This is a dictionary with 2 keys ("Lkinematics", "Rkinematics")
            # Each key will have a corresponding dictionary as its value
            #   [MovIdx, MovLength, MovStart, MovEnd, avepMov, peakpMov]
            #self.kinematics = self._raw_mov_kinematics(accmags, acounts)

    # This function will calculate the actual date and time from the time recorded in the sensor
    def _calc_datetime(self, time_stamp):
        """
        A function to convert a timestamp into a datetime instance

        Parameters:
            time_stamp: int
                a timestamp in units of microseconds since 1970-1-1-0:00 UTC

        Returns:
            datetime_utc: datetime
                a datetime instance whose value is converted from the time_stamp
        """
        datetime_utc = datetime.fromtimestamp(time_stamp/1e6, timezone.utc)
        return datetime_utc

    def _get_mag(self, sensors, device, row_idx=None, det_option='median'):
        """
        A function to calculate the magnitude of a provided dtype.

        Parameters:
            sensors: dict
                items are the members of the group 'Sensors' in h5py.File
            device: str
                'acc' or 'gyro'
            row_idx: list
                Recording start and end indices (each int)
            det_option: str
                'median' or 'customfunc'
                Subtract median or use some custom method to detrend

        Returns:
            out: dict
                detrended acceleration or angular velocity norm
                keys = ['lmag', 'rmag']
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
                         [x[dtype] for x in sensors.values()])

        # MATLAB's detrend function is not used,
        #   so we can consider that the default option
        #   for detrending the data is subtracting the median
        if det_option == 'median':
            out = map(lambda x: x - np.median(x), [lmag, rmag])
        else:
            out = map(detrend, [lmag, rmag])

        return dict(zip(['lmag', 'rmag'], out))

    # winsize in second unit
    def _mov_avg_filt(self, winsize, pd_series):
        win_len = int(20 * winsize)   # sfreq = 20Hz
        return pd_series.rolling(win_len).mean()

    def _calc_ind_threshold(self, maxprop, ivan=False):
        """
        ind_threshold = mean(peak_heights) - C * std(peak_heights)

        C = 1       Smith et al. (2015)
            0.5     Trujillo-Priego et al. (2017), ivan=True
        """
        return np.mean(maxprop['peak_heights'])\
                - (1-ivan + 0.5*ivan)*np.std(maxprop['peak_heights'])

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

        tkeys = ['laccth', 'raccth', 'lnaccth', 'rnaccth']
        return dict(zip(tkeys, np.multiply([1,1,-1,-1], list(accths))))

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
            poten = np.ceil((d_in_micro[1].seconds*convert
                           + d_in_micro[1].microseconds)/microlen)\
                                   .astype('int')
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
                count values (L/R)
        """
        raw_counts = map(np.sign, self.measures.accmags.values())
        acounts = [np.multiply(x, y) for x, y in\
                zip(raw_counts,
                    map(lambda x: np.greater(x, 0),
                        self.measures.velmags.values()))]
        return acounts

    def _get_cntc(self):
        """
        A function to get counts and tcounts

        Parameters:
            acounts: pd.DataFrame
                output of self._get_count(accmags, angvels)

        Returns:
            tcounts: dict
                keys: count, tcount
        """
        acounts = self._get_count()

        # Let's start with collecting all acc values that went over the threshold
        # The output of np.where would be a tuple - so take the first value
        # I do this to reduce the preprocessing time...
        #temp_posth_l = np.where(self.measures.over_accth['L'])[0]
        #temp_posth_r = np.where(self.measures.over_accth['R'])[0]
        #temp_negth_l = np.where(self.measures.under_naccth['L'])[0]
        #temp_negth_r = np.where(self.measures.under_naccth['R'])[0]

        temp_l = map(lambda x: np.where(x)[0],
                [self.measures.over_accth['L'],
                    self.measures.under_naccth['L']])

        temp_r = map(lambda x: np.where(x)[0],
                [self.measures.over_accth['R'],
                    self.measures.under_naccth['R']])

        # angular velocity should be taken into account...
        # let's just make sure that the detrended angvel[i] > 0
        angvel_gt_l = np.nonzero(acounts[0])[0]
        angvel_gt_r = np.nonzero(acounts[1])[0]

        over_posth_l, under_negth_l = map(lambda x:
                np.intersect1d(x, angvel_gt_l), temp_l)
        over_posth_r, under_negth_r = map(lambda x:
                np.intersect1d(x, angvel_gt_r), temp_r)

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

        ltcounts  = mark_tcount(over_posth_l, acounts[0], pos=True)
        lntcounts = mark_tcount(under_negth_l, acounts[0], pos=False)
        rtcounts  = mark_tcount(over_posth_r, acounts[1], pos=True)
        rntcounts = mark_tcount(under_negth_r, acounts[1], pos=False)

        return dict(zip(['L', 'R'], [[acounts[0], ltcounts+lntcounts],
                                     [acounts[1], rtcounts+rntcounts]]))

    def get_mov(self, side='L'):
        """
        A function to get movement counts

        Parameters:
            tcounts: pd.DataFrame
                the output of self._get_tcounts(accmags)

        Returns:
            tmov: list
                movement counts (L/R)
        """

        assert(side in ['L', 'R']), "side should be 'L' or 'R'"

        # tcounts is a dictionary (keys: 'L', 'R')
        # Each value is a list of TWO lists (counts, tcounts)
        tcounts = self._get_cntc()

        # index | tcount | th_crossed
        arr_a = np.column_stack((np.arange(len(tcounts[side][0])),
                                 tcounts[side][0],  # counts
                                 tcounts[side][1],  # tcounts
                                 self.measures.th_crossed[side]))
        arr_b = arr_a[np.nonzero(arr_a[:,2])[0], :]
        movidx = np.zeros((arr_b.shape[0], 3), dtype=int)

        for i in range(arr_b.shape[0]-1):
            pairdiff = np.diff(arr_b[i:i+2,:], axis=0).ravel()
            # Two Tcounts are different (-1 vs. 1)
            if all((pairdiff[2] != 0, pairdiff[0] <= 14)):
                sidx = int(arr_b[i+1, 0])     # second threshold crossing
                addi = int(sidx + 30 - pairdiff[0])  # a mov < 30 data points
                if arr_b[i,3] == arr_b[i,2]:  # if th_cross == t_count
                    movstart = arr_b[i,0]
                else:
                    if arr_a[int(arr_b[i,0]-1),3] == arr_b[i,2]:
                        movstart = arr_b[i,0]-1
                    else:
                        movstart = arr_b[i,0]-2
                try:
                    movend = np.where(
                            arr_a[sidx:addi,1] == -arr_a[sidx,2])[0][0]
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

    def acc_per_mov(self, side='L'):
        """
        A function to calculate average acceleration per movement
            and the peak acceleration per movement

        Parameters:
            side: str
                'L' or 'R'

        Returns:
            acc_arr: numpy ndarray
                [mov start | ave accmag for mov[i] | peak accmag for mov[i]]
        """
        assert(side in ['L', 'R']), "side should be 'L' or 'R'"

        movidx = self.get_mov(side)

        if side == 'L':
            accvec = np.abs(self.measures.accmags['lmag'].copy())
        else:
            accvec = np.abs(self.measures.accmags['rmag'].copy())

        acc_arr = np.array([[x, np.mean(accvec[x:y]), max(accvec[x:y])]\
                for x, y in zip(movidx[:,0], movidx[:,2]+1)])

        return acc_arr

    def plot_segment(self, time_passed, duration=20, side='L'):
        """
        A function to let user visually check movement counts

        Parameters:
            time_passed: float
                time passed from the start of the recording in seconds

            duration: int
                duration in seconds, default is 20

            side: str
                'L' or 'R'

        Returns:
            a diagnostic figure to check movement counts
        """
        assert(side in ['L', 'R']), "side should be 'L' or 'R'"

        movidx = self.get_mov(side)
        if side == 'L':
            labels = ['lmag', 'laccth', 'lnaccth']
        else:
            labels = ['rmag', 'raccth', 'rnaccth']

        if self.info.rowidx is not None:
            #new_t = self.info.record_times[0]\
            #        + timedelta(seconds=time_passed)
            #end_t = self.info.record_times[1]
            startidx = time_passed * 20
            endidx = startidx + duration * 20 + 1
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
            velline, = ax.plot(self.measures.velmags[labels[0]][startidx:endidx],
                    c='deepskyblue', linestyle='dashdot', label='angular velocity')
            ax.legend(handles = [accline, pthline, nthline, velline])

            if mov_st.size:
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
                    f"{(self.info.record_times[0] + timedelta(seconds=time_passed)).ctime()}"\
                    f" UTC\n(recording ended at {self.info.record_times[1].ctime()} UTC)"
            ax.set_title(title)
            ax.set_xlabel("Time since onset (sec)")
            ax.set_ylabel("Acc. magnitude (m/s^2)")
            xticks = np.arange(0, (duration+1)*20, 400)
            xlabs = [str(x) for x in np.arange(0, (duration+1), 20)]
            ax.set_xticks(ticks=xticks, labels=xlabs)

            plt.show()

        else:
            raise Exception("Please check your REDCap export. No time_donned was provided")


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

    def mark_simultaneous(self):
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
