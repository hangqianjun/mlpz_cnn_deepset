from pathlib import Path

import h5py
import numpy as np
import pandas as pd

__all__ = [
    "read_catalog",
    "load_filter_curves",
    "catalog_to_mags",
]

# Functions taking `bands` expect a dict band -> (catalogue magnitude column, filter curve file),
# supplied by the user, e.g. {"u": ("mag_u_lsst", "filters/DC2LSST_u.res"), ...}. Its order
# fixes the row order of every downstream array (mags, filters_array, ...).


def read_catalog(path, columns=None):
    """
    Read a flat catalogue into a DataFrame, picking the reader from the file extension.

    path: .hdf5/.h5 file with one 1-D dataset per column at the top level, or a .parquet file.
    columns: optional list of columns to read; None reads everything.
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix in (".hdf5", ".h5"):
        with h5py.File(path, "r") as f:
            keys = [k for k in f.keys() if isinstance(f[k], h5py.Dataset)] if columns is None else columns
            return pd.DataFrame({k: f[k][:] for k in keys})

    if suffix == ".parquet":
        return pd.read_parquet(path, columns=columns)

    raise ValueError(f"Unsupported catalogue format '{suffix}' for {path}")


def load_filter_curves(bands):
    """
    Load the raw (wavelength, transmission) curve of each band from its filter file.

    bands: dict band -> (column, filter file path).

    Returns dict band -> (M, 2) array, in the order of `bands`, ready for interpolate_filter_curves.
    """
    return {b: np.loadtxt(filter_file) for b, (_, filter_file) in bands.items()}


def catalog_to_mags(df, bands, nondetect_value=np.inf):
    """
    Extract the per-band magnitudes as a (n_bands, n_sources) array, rows in the order of `bands`.

    Non-detections (np.nan) are replaced by nondetect_value; the default np.inf treats them as
    unobserved, the convention of convert_data_format. A band whose column is missing from df
    is marked unobserved (np.inf) for every source.
    """
    mags = np.full((len(bands), len(df)), np.inf)
    for row, (column, _) in enumerate(bands.values()):
        if column in df:
            mags[row] = df[column].to_numpy(dtype=float)

    mags[np.isnan(mags)] = nondetect_value
    return mags
