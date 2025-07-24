"""
Written by Jinseok Oh, Ph.D.
2022/9/13 - present (as of 2025/7/16)

Calibration related functions

© 2023-2025 Infant Neuromotor Control Laboratory. All rights reserved.
"""
import warnings
from dataclasses import dataclass, field
from typing import Optional
from itertools import repeat
import numpy as np


@dataclass
class CalibFileInfo:
    """
    A data container for calibration file metadata.

    Attributes
    ----------
    fname : str
        File name (.cwa or .h5)
    fs : int
        Sampling frequency (Hz)
    timezone : str
        Timezone string (e.g.m, 'US/Pacific')
    offset : list[numpy.ndarray]
        Offset along a measurement axis
    misalign : list[numpy.ndarray]
        Misalignments
    raw_gs : list[numpy.ndarray]
        Original +/- 1g measurements
    off_gs : list[numpy.ndarray]
        Offset removed +/- 1g measurements
    boost_gs : list
        Gain corrected?
    samp_num : list
        Sample numbers of the windows where
        x, y, z +/- 1g measurements were obtained
    """
    offset: list = field(default_factory=list)
    misalign: list = field(default_factory=list)
    raw_gs: list = field(default_factory=list)
    off_gs: list = field(default_factory=list)
    boost_gs: list = field(default_factory=list)
    samp_num: list = field(default_factory=list)
    fname: str = ''
    fs: Optional[int] = 25
    timezone: str = ''


class CalibProcess:
    """ Handles axis-specific calibration using known +/-1g postures."""
    def __init__(self, absolute=False, g_thresholds=None,
                 winlen=5, stdcut=0.02, **kwargs):
        """
        Parameters
        ----------
        absolute : bool
            Default is False; if True, misalignment level is calculated
            based on absolute deviation from 0 g.
        g_thresholds : list
            The lower and the upper limits of measured g.
            Default is None (bcz of the warning: list is a dangerous default),
            and [0.9, 1.1] will be used.
        winlen : int | list | None
            Length of a window to measure offset along each axis in seconds.
            Default is 5 for all three axes.
            If a single integer is given, it will assume the same window length
            for all three axes.
        stdcut : float
            Cutoff of the standard deviation of the values within a window.
            Default is 0.02.
        """
        self._name = self.__class__.__name__
        self._kw = kwargs
        self.absolute = absolute
        self.g_thresholds = g_thresholds or [0.9, 1.1]
        if winlen is None:
            self.winlen = {'x': 5, 'y': 5, 'z': 5}
        elif isinstance(winlen, list):
            if len(winlen) == 1:
                self.winlen = {k: winlen[0] for k in 'xyz'}
            elif len(winlen) == 3:
                self.winlen = dict(zip(['x', 'y', 'z'], winlen))
            else:
                raise ValueError("winlen should be a list of 3, or an integer")
        else:
            self.winlen = {k: winlen for k in 'xyz'}

        self.stdcut = stdcut
        self.info = CalibFileInfo()

    def find_window_both(self, arr, axis, winlen=None):
        """ This is a wrapper to use find_window with a dict obj """
        thrs = self.g_thresholds.copy()
        sns = self.find_sns(arr, thrs[0], thrs[1])
        # If axis value is not provided, or not one of 'x', 'y', or 'z',
        # raise an error
        axis = axis.lower()
        if axis not in 'xyz':
            raise ValueError("axis value incorrect - pick one: 'x', 'y', 'z'")
        # start with the initial threshold values.
        winlen = winlen or self.winlen[axis]
        print(f"winlen, {axis.upper()}-axis: {winlen}")
        pwin, nwin = map(self.find_window, [arr]*2,
                         [sns['p'], sns['n']],
                         repeat(winlen, 2))

        while (pwin is None or nwin is None) and \
                thrs[0] > .75 and thrs[1] < 1.25:
            thrs[0] -= 0.05
            thrs[1] += 0.05
            print("Searching windows with new thresholds: ",
                  f"{thrs[0]:.2f}, {thrs[1]:.2f}")
            sns = self.find_sns(arr, thrs[0], thrs[1])
            pwin, nwin = map(self.find_window,
                             [arr]*2,
                             [sns['p'], sns['n']],
                             repeat(winlen, 2))

        if pwin is None or nwin is None:
            new_winlen = int(winlen - 1)
            if new_winlen < 1:
                if pwin is None:
                    warnings.warn(
                            f"winlen < 1s. Calibration output for {axis} (positive) will be NaNs.")
                    pwin = np.empty(0, int)
                if nwin is None:
                    warnings.warn(
                            f"winlen < 1s. Calibration output for {axis} (negative) will be NaNs.")
                    nwin = np.empty(0, int)
                return {'p': pwin, 'n': nwin}
            return self.find_window_both(arr, axis, winlen=new_winlen)

        self.winlen[axis] = winlen
        return {'p': pwin, 'n': nwin}

    def find_window(self, arr, sampnums, winlen=None):
        """
        A function to find sample numbers of THE window (length = winlen).
        The window should have adjacent points, and the std of the points
        should be no greater than a cut-off for std (default: 0.02).

        Parameters
        ----------
        arr : numpy.ndarray
            An array of measured gravitational acceleration along an axis.
        sampnums : numpy.ndarray
            An array of sample numbers corresponding to the periods
            when one of the three (X, Y, or Z) axes was measuring +/-1g.
        winlen: int | None
            Length of the window

        Returns
        -------
        x: numpy.ndarray | None
            An array of sample numbers. The difference of the two adjacent
            array elements will ALWAYS be one, and the standard deviation
            of the entire array will be less than a cut-off (stdcut).
            If you don't find any such array, return None.
        """
        winlen = winlen or 5
        win_size = self.info.fs * winlen
        ia, ib = 0, int(win_size)

        # (8/7/23) ib should not be greater than the end of sampnums...
        while ib < len(sampnums):
            # Continuity should be kept!
            # 0.02 is experimental - could adjust later (6/22/23)
            segment = sampnums[ia:ib]
            if np.all(np.diff(segment) == 1) and np.std(arr[segment]) < self.stdcut:
                return segment
            ia += 10
            ib += 10    # push back by 10 data points.
        return None

    @staticmethod
    def find_sns(arr, thr_low, thr_high):
        """
        A function to find SampleNumberS (sns)

        Paramters
        ---------
        arr : numpy.ndarray
            An array of measured gravitational acceleration along an axis.
        thr_low : float
            Lower threshold of 1g. It's expected to be self.g_thresholds[0].
        thr_high : float
            Upper threshold of 1g. It's expected to be self.g_thresholds[1].

        Returns
        -------
        x: dict
            'p' = arr sample numbers between thr_high and thr_low.
            'n' = arr sample numbers between -(thr_low) and -(thr_high).
        """
        return {
                'p': np.where((arr < thr_high) & (arr > thr_low))[0],
                'n': np.where((arr < -thr_low) & (arr > -thr_high))[0]
        }

    def get_gs(self, ndarr):
        """
        A function to calculate gain, offset, and misalignment

        Sometimes calibration is not done properly. If so,
        NA values should be filled...

        Parameters
        ----------
        ndarr : numpy.ndarray
            Raw measurement of +/-1g along the measurement axis.

        Returns
        -------
        x: dict
            offset: measurement offset along each axis
            misalign: misalignment along each axis
            processed: +/- 1g measurements minus axis-specific offset
            gs_orig: original measurement of +/-1g along each axis
            gs: corrected measurement of +/-1g along each axis
            gs_boost: misalignment compensated gs. Not used.
            samp_num: sample numbers of the WINDOWS (return of find_windows)
        """
        absolute = self.absolute
        x_w, y_w, z_w = map(self.find_window_both,
                            [ndarr[:, i] for i in range(3)],
                            ['x', 'y', 'z'])
        # non-g sample numbers
        # For example, x_P(ositive)N(on)G is the concatenated regions
        # where y or z axis was measuring positive g (1 g)
        x_png = sorted(np.concatenate((y_w['p'], z_w['p'])))
        x_nng = sorted(np.concatenate((y_w['n'], z_w['n'])))
        y_png = sorted(np.concatenate((x_w['p'], z_w['p'])))
        y_nng = sorted(np.concatenate((x_w['n'], z_w['n'])))
        z_png = sorted(np.concatenate((x_w['p'], y_w['p'])))
        z_nng = sorted(np.concatenate((x_w['n'], y_w['n'])))
        x_non_g = sorted(np.concatenate((x_png, x_nng)))
        y_non_g = sorted(np.concatenate((y_png, y_nng)))
        z_non_g = sorted(np.concatenate((z_png, z_nng)))

        # Correct method of calculating offset - 6/25/23 (GEL)
        offset = [
                np.mean(ndarr[x_non_g, 0]),
                np.mean(ndarr[y_non_g, 1]),
                np.mean(ndarr[z_non_g, 2])
                ]
        # measured g values, offset NOT removed
        gs_orig = [
                np.mean(ndarr[x_w['n'], 0]), np.mean(ndarr[x_w['p'], 0]),
                np.mean(ndarr[y_w['n'], 1]), np.mean(ndarr[y_w['p'], 1]),
                np.mean(ndarr[z_w['n'], 2]), np.mean(ndarr[z_w['p'], 2]),
                ]
        # Remove offsets
        corrected = ndarr - np.array(offset)
        # Then check for the amount of misalignment
        # Use only the positive values... doesn't matter much
        if absolute:
            xm, ym, zm = map(lambda x: np.mean(abs(x)),
                             [corrected[x_png, 0],
                              corrected[y_png, 1],
                              corrected[z_png, 2]])
        else:
            xm, ym, zm = map(np.mean,
                             [corrected[x_png, 0],
                              corrected[y_png, 1],
                              corrected[z_png, 2]])
        # measured g values, offset removed
        gs = [
                np.mean(corrected[x_w['n'], 0]),
                np.mean(corrected[x_w['p'], 0]),
                np.mean(corrected[y_w['n'], 1]),
                np.mean(corrected[y_w['p'], 1]),
                np.mean(corrected[z_w['n'], 2]),
                np.mean(corrected[z_w['p'], 2]),
                ]
        # misalignment reflected (percentage boost)
        gs_boost = [gs[0]*(1+xm), gs[1]*(1+xm),
                    gs[2]*(1+ym), gs[3]*(1+ym),
                    gs[4]*(1+zm), gs[5]*(1+zm)]

        return {'offset': offset,
                'misalign': [xm, ym, zm],
                'gs_orig': gs_orig,
                'gs': gs,
                'gs_boost': gs_boost,
                'samp_num': {'x': x_w, 'y': y_w, 'z': z_w}}
