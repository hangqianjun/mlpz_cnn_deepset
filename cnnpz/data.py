import numpy as np

__all__ = [
    "rebin_filter",
    "get_bin_edges",
    "convert_data_format",
    "stretch",
    "transform_data_to_XY",
    "make_incomplete_nir_data",
]


def rebin_filter(bin_edges, counts, new_edges):
    new_counts = []
    for i in range(len(new_edges) - 1):
        lo, hi = new_edges[i], new_edges[i + 1]
        total = 0.0
        for j, (b0, b1, c) in enumerate(zip(bin_edges[:-1], bin_edges[1:], counts)):
            overlap = max(0, min(hi, b1) - max(lo, b0))
            total += c * overlap  # /(b1 - b0)  # proportional share
        new_counts.append(total)
    return np.array(new_counts)


def get_bin_edges(centres):
    step = centres[1] - centres[0]  # uniform spacing
    inner_edges = (centres[:-1] + centres[1:]) / 2
    first_edge = centres[0] - step / 2
    last_edge = centres[-1] + step / 2
    return np.concatenate([[first_edge], inner_edges, [last_edge]])


def convert_data_format(
    df, lambda_array_cen, filter_blocks, no_detect=np.nan, no_obs=np.inf, no_detect_val=0, no_obs_val=0
):
    # no detection: setting to nan, sensitivity = 1
    # no observation: setting to inf, sensitivity = 0
    # normalize by i-band mag
    num_galaxies = len(df)
    num_wavelength_bins = len(lambda_array_cen)

    mags_data = np.zeros((num_galaxies, num_wavelength_bins))
    filter_availability_data = np.zeros((num_galaxies, num_wavelength_bins))

    for b in "ugrizy":
        ind = filter_blocks[b].astype(bool)
        mags_data[:, ind] = df[f"mag_{b}_lsst"].to_numpy()[:, None]
        filter_availability_data[:, ind] = 1

    for b in "JH":
        ind = filter_blocks[b].astype(bool)
        mags_data[:, ind] = df[f"mag_{b}_roman"].to_numpy()[:, None]
        filter_availability_data[:, ind] = 1

    mags_data -= df["mag_i_lsst"].to_numpy()[:, None]  # normalize by i-band mag

    # convert any nan or inf to zero:
    ind_nan = np.isnan(mags_data)
    ind_inf = np.isinf(mags_data)

    if no_obs == np.inf:
        filter_availability_data[ind_inf] = 0
        mags_data[ind_inf] = no_obs_val
        mags_data[ind_nan] = no_detect_val
    elif no_obs == np.nan:
        filter_availability_data[ind_nan] = 0
        mags_data[ind_nan] = no_obs_val
        mags_data[ind_inf] = no_detect_val

    wave_labels = np.arange(num_wavelength_bins)
    # normalize this
    wave_labels = wave_labels / wave_labels[-1]

    lambda_labels = np.outer(np.ones(num_galaxies), wave_labels)

    transformed_df = [mags_data, lambda_labels, filter_availability_data]
    transformed_df = np.stack(transformed_df, axis=-1)

    return transformed_df


def stretch(x, c=0.5, k=10):
    """
    x: array of values in [0,1]
    c: center of your typical fluctuation band
    k: stretch strength (try 5-50; higher = more aggressive)
    """
    x = np.asarray(x)
    u = np.where(x >= c, (x - c) / (1 - c), (x - c) / c)  # -> [-1, 1], c -> 0
    stretched = np.arcsinh(k * u) / np.arcsinh(k)  # amplify near u=0
    y = np.where(stretched >= 0, c + stretched * (1 - c), c + stretched * c)
    return np.clip(y, 0, 1)


def transform_data_to_XY(data, lambda_array_cen, filter_blocks, apply_stretch=True, c=0.8, k=20, missingY=False):
    data_transformed = convert_data_format(data, lambda_array_cen, filter_blocks)
    # now split training and validation set:
    if missingY is False:
        Y = data["redshift"]
    else:
        Y = 0
    X = np.copy(data_transformed)
    # X[:,:,0] /= max_mag
    if apply_stretch is True:
        X[:, :, 0] = stretch(X[:, :, 0], c=c, k=k)
    return X, Y


def make_incomplete_nir_data(data, lambda_array_cen, filter_blocks, frac=0.5, sub_val=np.inf, apply_stretch=False):
    subset = data.sample(frac=frac)
    idx = subset.index
    idx = list(subset.index)
    data_copy = data.copy()
    data_copy.loc[idx, "mag_J_roman"] = sub_val
    data_copy.loc[idx, "mag_H_roman"] = sub_val
    X_misnir, Y_misnir = transform_data_to_XY(data_copy, lambda_array_cen, filter_blocks, apply_stretch=apply_stretch)
    return X_misnir, Y_misnir
