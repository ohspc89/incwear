"""
I/O utility functions for Axivity data formats (.cwa, .tsv, .csv)
Written by Jinseok Oh
© 2023-2025 Infant Neuromotor Control Laboratory. All rights reserved.
"""

import numpy as np
from pyarrow import csv
from skdh.io import ReadCwa

sv_readOptions = csv.ReadOptions(column_names=[],
                                 autogenerate_column_names=True)


def read_tsv(filepath, columns=None):
    """
    Read TSV file using pyarrow.

    Parameters
    ----------
    filepath : str
        Path to TSV file
    columns : list or None
        Column names to load

    Returns
    -------
    numpy.ndarray
    """
    opts = csv.ReadOptions(column_names=columns) if columns else sv_readOptions
    parse_opts = csv.ParseOptions(delimiter='\t')
    with open(filepath, 'rb') as f1:
        tsv_data = np.asarray(csv.read_csv(f1,
                                           read_options=opts,
                                           parse_options=parse_opts))
    return tsv_data


def read_csv(filepath, columns=None):
    """
    Read CSV using pyarrow.

    Parameters
    ----------
    filepath : str
        Path to CSV file
    columns : list or None
        Column names to load

    Returns
    -------
    dict
        Dictionary of numpy arrays for each requested column
    """
    opts = csv.ReadOptions(column_names=columns) if columns else sv_readOptions
    csv_data = csv.read_csv(filepath, read_options=opts).to_pandas()
    return csv_data


def read_cwa(filepath):
    """
    Read Axivity .cwa binary file using skdh.io.ReadCwa.

    Parameters
    ----------
    filepath : str
        Path to .cwa file
    resample_to_hz : int or None
        If provided, resample to this frequency
    convert_units : bool
        If True, scale raw values to m/s^2 or rad/s

    Returns
    -------
    dict
        Dictionary with 'accel', 'gyro', and 'time'
    """
    cwa_reader = ReadCwa()
    data = cwa_reader.predict(file=filepath)
    return data
