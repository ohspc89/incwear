"""
Written by Jinseok Oh, Ph.D.
2022/9/13 - present (as of 2026/4/16)

Utility functions handling the timestamps of the IMU data

© 2023-2026 Infant Neuromotor Control Laboratory. All rights reserved.
"""
from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta, timezone
from typing import Sequence

import numpy as np
import pandas as pd
import pytz

from incwear.errors import TimingError

logger = logging.getLogger(__name__)

OPAL_SENSOR_TYPES = {"OpalV1", "OpalV2", "OpalV2Single"}
AXIVITY_SENSOR_TYPES = {"Ax6", "Ax6Single"}


def convert_timestamp_to_utc(
        timestamp,
        sensor_type: str,
        timezone_str: str):
    """
    Convert various timestamp formats to UTC datetime.

    Parameters
    ----------
    timestamp: int | str | numpy.datetime64
         A timestamp from the sensor stream.
    sensor_type: str
        One of ['OpalV2', 'OpalV1', 'OpalV2Single', 'Ax6', 'Ax6Single',
        'Ax6OmGui'].
         For Opal sensors, directly transform this to a datetime object
         For Axivity sensors, dtype is numpy.datetime64.
         If you use OmConvert, timestamp is a string.
    timezone_str : str
        Local timezone of the data when required by the sensor format.

    Returns
    -------
    datetime_utc: datetime.datetime
        UTC datetime.
    """
    if sensor_type in OPAL_SENSOR_TYPES:
        ts = timestamp / 1e6
        return datetime.fromtimestamp(ts, timezone.utc)

    if sensor_type in AXIVITY_SENSOR_TYPES:
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


def _delta_to_index(delta: timedelta, fs: int) -> int:
    microseconds = (delta.days * 86400 + delta.seconds) * 1e6 + delta.microseconds
    return int(np.round(microseconds / (1e6 / fs)))


def get_don_doff_indices(
    sensorts,
    fs: int,
    don_doff_utc,
    sensor_type: str = 'OpalV2',
        btnstatus: int | None = None,
):
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
    list[int] or None
        List of two indices (start_idx, end_idx) or None.
        If donned and doffed times are found from the REDCap export,
        (or any other source) return a list of two indices of datapoints
        that each corresponds to the start and the end of the recording.
    """
    n_samples = len(sensorts)
    if n_samples == 0:
        raise ValueError("sensorts must not be empty")

    if fs is None or fs <= 0:
            raise TimingError(f"Invalid sampling frequency: {fs}")

    if don_doff_utc is None:
        if btnstatus is not None:
            logger.info("No don/doff time provided; using range from button press.")
            return list(range(btnstatus, n_samples))
        logger.info("No don/doff time; using full range.")
        return None

    start_utc, end_utc = don_doff_utc

    if btnstatus is not None:
        start_utc = convert_timestamp_to_utc(sensorts[btnstatus], sensor_type, "UTC")

    sensor_start_utc = convert_timestamp_to_utc(sensorts[0], sensor_type, "UTC")
    start_idx = max(0, _delta_to_index(start_utc - sensor_start_utc, fs))
    end_idx = min(_delta_to_index(end_utc - sensor_start_utc, fs), n_samples - 1)

    if end_idx <= start_idx:
        logger.warning(
            "Computed end index (%s) is not greater than start index (%s); "
            "returning an empty range.",
            end_idx,
            start_idx,
        )
        return []

    return list(range(start_idx, end_idx))


def local_to_utc(timelist: Sequence[int], study_tz: str):
    """
    Convert a local datetime component list to a UTC datetime.

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
    if len(timelist) < 5:
        raise ValueError(warning_msg)
    local = pytz.timezone(study_tz)
    naive = datetime(*timelist)
    local_dt = local.localize(naive, is_dst=None)
    return local_dt.astimezone(pytz.utc)


def convert_to_utc(datetime_obj: datetime, site: str):
    """Convert a naive local datetime to UTC using the given site timezone."""
    local = pytz.timezone(site)
    local_dt = local.localize(datetime_obj, is_dst=None)
    return local_dt.astimezone(pytz.utc)


def _filename_match_score(candidate: str, filename: str) -> float:
    if not isinstance(candidate, str):
        return 0.0
    tokens = [tok.lower() for tok in re.split(r"[-:_.]", candidate) if tok]
    if not tokens:
        return 0.0
    return float(np.mean([token in filename.lower() for token in tokens]))


def _infer_doff_datetime(record_date: str, don_time:str, doff_time: str) -> datetime:
    don_dt = datetime.strptime(f"{record_date}-{don_time}", "%Y%m%d-%H:%M")
    doff_dt = datetime.strptime(f"{record_date}-{doff_time}", "%Y%m%d-%H:%M")

    don_hour = int(don_time.split(":")[0])
    doff_hour = int(doff_time.split(":")[0])
    nextday = ((doff_hour < 14) and (abs(don_hour - doff_hour) < 10)) or (doff_hour < 12)

    if nextday:
        doff_dt += timedelta(days=1)
    return doff_dt


def make_start_end_datetime(redcap_csv, filename: str, site: str):
    """
    Build UTC don/doff datetimes from REDCap-style metadata.

    Notes
    -----
    This helper is project-specific and intentionally retains the existing
    REDCap filename matching and overnight heuristic. It should be considered
    study metadata glue rather than a universal IMU timestamp utility.

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
    match_scores = redcap_csv.filename.apply(lambda x: _filename_match_score(x, filename))
    match_idx = np.where(match_scores > 0.5)[0]

    if match_idx.size == 0:
        logger.warning("No REDCap filename match found for %s", filename)
        return None

    don_n_doff = redcap_csv.loc[match_idx[0], ["don_t", "doff_t"]]
    if pd.isna(don_n_doff[0]) or pd.isna(don_n_doff[1]):
        logger.info("Missing don/doff values for %s; returning None.", filename)
        return None

    record_date = path_removed.split("-")[0]
    don_dt = datetime.strptime(f"{record_date}-{don_n_doff[0]}", "%Y%m%d-%H:%M")
    # Preserve the original +1 minute protection against slightly early REDCap don times.
    don_dt += timedelta(minutes=1)
    doff_dt = _infer_doff_datetime(record_date, don_n_doff[0], don_n_doff[1])

    return [convert_to_utc(don_dt, site), convert_to_utc(doff_dt, site)]
