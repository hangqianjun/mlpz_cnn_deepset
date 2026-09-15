import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

FILTER_ORDER = "ugrizyJH"


@pytest.fixture
def lambda_array_cen():
    """Wavelength bin centres: one bin per filter, in ugrizyJH order."""
    return np.linspace(0.5, 2.0, len(FILTER_ORDER))


@pytest.fixture
def filter_blocks(lambda_array_cen):
    """One-hot filter->wavelength-bin map: filter i owns bin i exactly."""
    n = len(lambda_array_cen)
    blocks = {}
    for i, b in enumerate(FILTER_ORDER):
        arr = np.zeros(n, dtype=int)
        arr[i] = 1
        blocks[b] = arr
    return blocks


@pytest.fixture
def synthetic_df():
    """10 galaxies with finite mags/redshift, plus one NaN (no-detect) and one inf (no-observe) case."""
    n = 10
    rng = np.random.default_rng(0)
    data = {
        "mag_u_lsst": 24.0 + rng.normal(0, 0.1, n),
        "mag_g_lsst": 23.5 + rng.normal(0, 0.1, n),
        "mag_r_lsst": 23.0 + rng.normal(0, 0.1, n),
        "mag_i_lsst": 22.5 + rng.normal(0, 0.1, n),
        "mag_z_lsst": 22.2 + rng.normal(0, 0.1, n),
        "mag_y_lsst": 22.0 + rng.normal(0, 0.1, n),
        "mag_J_roman": 21.8 + rng.normal(0, 0.1, n),
        "mag_H_roman": 21.7 + rng.normal(0, 0.1, n),
        "redshift": np.linspace(0.1, 2.0, n),
    }
    df = pd.DataFrame(data)
    df.loc[1, "mag_u_lsst"] = np.nan  # no detection
    df.loc[2, "mag_J_roman"] = np.inf  # no observation
    return df
