"""
I/O utility functions for Axivity data formats (.cwa, .tsv, .csv)

Written by Jinseok Oh
© 2023-2026 Infant Neuromotor Control Laboratory. All rights reserved.

Third-party components
----------------------
- Axivity .cwa file reading is performed via `skdh.io.ReadCwa`
  from scikit-digital-health (SKDH), licensed under the MIT License.
"""

import numpy as np
from pyarrow import csv
from skdh.io import ReadCwa

from incwear.errors import InvalidSensorFormatError

sv_readOptions = csv.ReadOptions(
    column_names=[],
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
        tsv_data = np.asarray(csv.read_csv(
            f1,
            read_options=opts,
            parse_options=parse_opts,
        ))
    if tsv_data.ndim != 2 or tsv_data.shape[1] < 7:
        raise InvalidSensorFormatError(
            f"Expected TSV with at least 7 columns, got shape {tsv_data.shape}."
        )
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
    pandas.DataFrame
    """
    opts = csv.ReadOptions(column_names=columns) if columns else sv_readOptions
    csv_data = csv.read_csv(filepath, read_options=opts).to_pandas()
    return csv_data


def read_cwa(filepath):
    """
    Read Axivity .cwa binary file using `skdh.io.ReadCwa`.

    Notes
    -----
    This is a thin wrapper around scikit-digital-health's ``ReadCwa`` reader.
    See the package's MIT license and project documentation for details.

    Parameters
    ----------
    filepath : str
        Path to .cwa file

    Returns
    -------
    dict
        Dictionary with 'accel', 'gyro', and 'time'
    """
    cwa_reader = ReadCwa()
    return cwa_reader.predict(file=filepath)
