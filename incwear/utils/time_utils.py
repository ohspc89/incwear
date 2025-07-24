"""
Written by Jinseok Oh, Ph.D.
2022/9/13 - present (as of 2025/7/16)

Utility functions handling the timestamps of the IMU data

© 2023-2024 Infant Neuromotor Control Laboratory. All rights reserved.
"""
import re
from datetime import datetime, timedelta, timezone
import pytz
import numpy as np
import pandas as pd


def convert_timestamp_to_utc(timestamp, sensor_type, timezone_str):
    """
    Convert various timestamp formats to UTC datetime.

    Parameters
    ----------
    timestamp: int | str | numpy.datetime64
         A timestamp in units of microseconds since 1970-1-1-0:00 UTC.
    sensor_type: str
        One of ['OpalV2', 'OpalV1', 'Ax6', 'Ax6OmGui']
         For Opal sensors, directly transform this to a datetime object
         For Axivity sensors, dtype is numpy.datetime64.
         If you use OmConvert, timestamp is a string.
    timezone_str : str
        Local timezone of the data.

    Returns
    -------
    datetime_utc: datetime.datetime
        UTC datetime.
    """
    if sensor_type in ['OpalV2', 'OpalV1', 'OpalV2Single']:
        ts = timestamp / 1e6
        return datetime.fromtimestamp(ts, timezone.utc)

    if sensor_type in ['Ax6', 'Ax6Single']:
        ts = np.array(timestamp * 1e3, dtype='datetime64[ms]')
        dt_local = pd.to_datetime(ts).tz_localize(timezone_str)
        return dt_local.astimezone(pytz.utc)

    if sensor_type == 'Ax6OmGui':
        # If you use OmConvert, time stamp format is
        # %Y-%m-%d %H:%M:%S.%f
        dt_naive = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S.%f")
        dt_local = pd.to_datetime(dt_naive).tz_localize(timezone_str)
        return dt_local.astimezone(pytz.utc)

    raise ValueError(f"Unknown sensor type: {sensor_type}")


def get_don_doff_indices(sensorts, fs, don_doff_utc, sensor_type='OpalV2',
                         btnstatus=None):
    """
    Convert don/doff times to sample indices.

    Parameters
    ----------
    sensorts : array-like
        Timestamps (usually from the sensor data).
        If preprocessing Opal data, the type is "h5py._hl.group.Group[0]
        Otherwise, it will be an array.
    fs : int
        Sampling frequency.
    don_doff_utc : list[datetime, datetime] | None
        Don and doff datetime in UTC
    sensor_type : str
        Sensor type('OpalV2', etc.)
    btnstatus : int | None
        If present, indicates button press index (for OpalV1)

    Returns
    -------
    row_idx : list[int] or None
        List of two indices (start_idx, end_idx) or None.
        If donned and doffed times are found from the REDCap export,
        (or any other source) return a list of two indices of datapoints
        that each corresponds to the start and the end of the recording.
    """
    if don_doff_utc is None:
        if btnstatus:
            print("No don/doff time. Starting from button press.")
            return list(range(btnstatus, len(sensorts)))
        print("No don/doff time. Using full range.")
        return None

    if btnstatus:
        don_doff_utc[0] = convert_timestamp_to_utc(
                sensorts[btnstatus], sensor_type, 'UTC')

    def delta_to_index(delta):
        micro = (delta.days * 86400 + delta.seconds) * 1e6 + delta.microseconds
        return int(np.round(micro / (1e6 / fs)))

    delta_start = don_doff_utc[0] - convert_timestamp_to_utc(
            sensorts[0], sensor_type, 'UTC')
    delta_end = don_doff_utc[1] - convert_timestamp_to_utc(
            sensorts[0], sensor_type, 'UTC')

    start_idx = delta_to_index(delta_start)
    end_idx = delta_to_index(delta_end)
    end_idx = min(end_idx, len(sensorts) - 1)

    return list(range(start_idx, end_idx))


def local_to_utc(timelist, study_tz):
    """
    A function to make a datetime object with its time in UTC

    Parameters
    ----------
    timelist: list
        In the following order:
        [Year, month, day, hour, minute, second, microsecond]
        If the list length is less than 5, it will inform the
        user to provide the argument in the proper format.
        When second and microsecond are not given, they are zeros.

    study_tz: str
        Timezone where the data collection happened
        ex. 'US/Pacific', 'America/Los_Angeles'

    Returns
    -------
    local_dt: datetime
        A datetime object with its timezone set to UTC
    """
    warning_msg = """You need to provide at least five numbers:
    Year (ex. 2023)
    Month (ex. 7)
    Day (ex. 12)
    Hour (ex. 18)
    Minute (ex. 29)"""
    assert len(timelist) >= 5, warning_msg
    local = pytz.timezone(study_tz)
    naive = datetime(*timelist)
    local_dt = local.localize(naive, is_dst=None)
    return local_dt.astimezone(pytz.utc)


def convert_to_utc(datetime_obj, site):
    """A function converting a local time to UTC"""
    local = pytz.timezone(site)
    local_dt = local.localize(datetime_obj, is_dst=None)
    utc_dt = local_dt.astimezone(pytz.utc)
    return utc_dt


def make_start_end_datetime(redcap_csv, filename, site):
    """
    A function to make two datetime objects based on the
        entries from the REDCap export

    Parameters
    ----------
    redcap_csv: pandas.DataFrame
    filename: string
        format: '/Some path/YYYYmmdd-HHMMSS_[identifier].h5'
    site: str
        where the data were collected (ex. America/Guatemala)

    Returns
    -------
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
            lowered = list(map(lambda x: x.lower(), splitted))
            result = np.mean([x in filename.lower() for x in lowered]) > 0.5

        return result


    idx = redcap_csv.filename.apply(lambda x: match_any(x, filename))

    don_n_doff = redcap_csv.loc[np.where(idx)[0][0], ['don_t', 'doff_t']]
    # don_n_doff = times.values[0]  # times is a Pandas Series

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
