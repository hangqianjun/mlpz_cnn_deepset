import numpy as np

__all__ = [
    "rebin_filter",
    "get_bin_edges",
    "stretch",
    "build_filter_bank",
    "make_lambda_bins",
    "build_binned_filter_operator",
    "convert_data_format",
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


# --- Continuous filter-curve representation ---
# The supported photometry representation: each band's real transmission curve is
# interpolated onto a shared wavelength grid and combined into a "filter bank" that
# fairly splits credit between overlapping filters (no double-counting), then binned
# down to a fixed number of CNN input bins.


def build_filter_bank(filter_curves, bands, lambda_common):
    """
    Interpolate each band's raw (wavelength, transmission) curve onto lambda_common and
    area-normalize it so it integrates to 1 over lambda_common.

    filter_curves: dict[str -> (M, 2) array] raw curves keyed by band letter.
    bands: sequence of band letters fixing row order, e.g. "ugrizyJH".
    lambda_common: (L,) shared, uniformly-spaced wavelength grid.

    Math
    ----
    Let T_b(lambda) be band b's raw transmission curve, interpolated onto lambda_common
    (zero outside its native support). Each band is independently rescaled by its own
    integral:

        filters_array_b(lambda) = T_b(lambda) / integral(T_b(lambda) dlambda)

    so overlapping/wide bands don't dominate purely by having a larger raw area -- credit
    between overlapping bands is instead split downstream, per wavelength bin, by
    build_binned_filter_operator (equal split among the bands active in that bin, not by
    relative transmission).

    Returns filters_array, shape (n_bands, L).
    """
    dlambda = lambda_common[1] - lambda_common[0]
    raw_filters = np.array(
        [np.interp(lambda_common, filter_curves[b][:, 0], filter_curves[b][:, 1], left=0, right=0) for b in bands]
    )

    area = raw_filters.sum(axis=1, keepdims=True) * dlambda
    filters_array = np.divide(raw_filters, area, out=np.zeros_like(raw_filters), where=area > 0)

    return filters_array


def make_lambda_bins(lambda_common, n_bins):
    """
    Build n_bins equal-width wavelength bins spanning lambda_common's range.

    Returns (lambda_bin_centers, bin_idx):
      - lambda_bin_centers: (n_bins,) bin center wavelengths.
      - bin_idx: (L,) int array mapping each lambda_common point to its bin, in [0, n_bins).
    """
    lambda_bin_edges = np.linspace(lambda_common[0], lambda_common[-1], n_bins + 1)
    lambda_bin_centers = (lambda_bin_edges[:-1] + lambda_bin_edges[1:]) / 2
    bin_idx = np.clip(np.digitize(lambda_common, lambda_bin_edges) - 1, 0, n_bins - 1)
    return lambda_bin_centers, bin_idx


def build_binned_filter_operator(filters_array, bin_idx, n_bins, lambda_common):
    """
    Turn per-lambda-point, area-normalized filter curves (from build_filter_bank) into a
    fixed (n_bands, n_bins) linear operator, so the whole dataset can be transformed with
    a single matmul: amplitudes (N, n_bands) @ filters_binned -> (N, n_bins).

    Each band's curve is integrated (not averaged) within a bin, then each bin's column is
    rescaled to sum to 1 across bands -- a genuine weighted average, weighted by how much
    of each contributing band's response actually falls in that bin (so `amplitudes @
    filters_binned` stays on the same scale as the input magnitudes: a proper per-bin
    magnitude estimate, not a small fraction of it):

        raw_integral[b, i]   = integral_{bin i} filters_array[b](lambda) dlambda
        filters_binned[b, i] = raw_integral[b, i] / sum_b' raw_integral[b', i]

    A band contributes to bin i whenever filters_binned[b, i] > 0; that pattern is what
    convert_data_format checks (against which bands were actually observed for a galaxy)
    to build the coverage channel -- no separate operator is needed for it.

    Returns filters_binned, shape (n_bands, n_bins).
    """
    dlambda = lambda_common[1] - lambda_common[0]
    L = filters_array.shape[1]

    W = np.zeros((n_bins, L))
    W[bin_idx, np.arange(L)] = 1.0

    integral = (filters_array @ W.T) * dlambda  # (n_bands, n_bins)
    integral_sum = integral.sum(axis=0)  # (n_bins,)
    filters_binned = np.divide(integral, integral_sum, out=np.zeros_like(integral), where=integral_sum > 0)

    return filters_binned


def convert_data_format(
    df,
    filters_binned,
    bin_idx,
    lambda_bin_centers,
    bands,
    no_detect=np.nan,
    no_obs=np.inf,
    no_detect_val=0,
    no_obs_val=0,
):
    """
    Vectorized over all galaxies.

    df: galaxy dataframe with mag_{b}_lsst / mag_{b}_roman columns for b in `bands`.
    filters_binned: (n_bands, n_bins) from build_binned_filter_operator(filters_array, ...).
    bin_idx: unused directly here (kept for signature parity with the binning helpers);
        n_bins is inferred from lambda_bin_centers.
    bands: band letters matching filters_binned row order and df's mag columns, e.g.
        "ugrizyJH".

    no_detect_val/no_obs_val are accepted for parity with the historical block-based
    signature but don't map onto a post-binning fill: each output bin here is a weighted
    blend of possibly several bands rather than one band's passthrough, so a missing
    band's amplitude is zeroed before binning (see below) rather than the output bin
    being overwritten afterwards.

    Returns X, shape (N, n_bins, 3):
      channel 0: binned, amplitude-weighted curve value, i-band-normalized -- the i-band
        subtraction is applied to the binned curve itself (mags_data - mag_i_lsst), not
        to the per-band amplitudes before binning.
      channel 1: wavelength-bin position label (0..1), identical across galaxies.
      channel 2: binary coverage mask (1 if any observed band contributes to this bin, 0
        if no observed band covers it at all).
    """
    n_bins = len(lambda_bin_centers)
    n_galaxies = len(df)

    roman_bands = "YJH"
    mag_columns = [f"mag_{b}_roman" if b in roman_bands else f"mag_{b}_lsst" for b in bands]

    amplitudes = df[mag_columns].to_numpy()

    ind_nan = np.isnan(amplitudes)  # no detection
    ind_inf = np.isinf(amplitudes)  # no observation

    amplitudes_clean = np.where(ind_nan | ind_inf, 0.0, amplitudes)

    mags_data = amplitudes_clean @ filters_binned  # (N, n_bins)
    mags_data -= df["mag_i_lsst"].to_numpy()[:, None]  # normalize by i-band mag

    observed = (~ind_inf).astype(float)  # 1 if band was observed (nan still counts, inf does not)
    coverage = (observed @ (filters_binned > 0)) > 0  # (N, n_bins), binary
    coverage = coverage.astype(float)

    wave_labels = np.arange(n_bins)
    wave_labels = wave_labels / wave_labels[-1]
    lambda_labels = np.outer(np.ones(n_galaxies), wave_labels)

    transformed_df = np.stack([mags_data, lambda_labels, coverage], axis=-1)
    return transformed_df


def transform_data_to_XY(
    data,
    filters_binned,
    bin_idx,
    lambda_bin_centers,
    bands,
    apply_stretch=True,
    c=0.8,
    k=20,
    missingY=False,
):
    data_transformed = convert_data_format(data, filters_binned, bin_idx, lambda_bin_centers, bands)
    if missingY is False:
        Y = data["redshift"]
    else:
        Y = 0
    X = np.copy(data_transformed)
    if apply_stretch is True:
        X[:, :, 0] = stretch(X[:, :, 0], c=c, k=k)
    return X, Y


def make_incomplete_nir_data(
    data,
    filters_binned,
    bin_idx,
    lambda_bin_centers,
    bands,
    frac=0.5,
    sub_val=np.inf,
    apply_stretch=False,
):
    subset = data.sample(frac=0.5)
    idx = list(subset.index)
    data_copy = data.copy()
    data_copy.loc[idx, "mag_J_roman"] = np.inf
    data_copy.loc[idx, "mag_H_roman"] = np.inf
    X_misnir, Y_misnir = transform_data_to_XY(
        data_copy, filters_binned, bin_idx, lambda_bin_centers, bands, apply_stretch=apply_stretch
    )
    return X_misnir, Y_misnir
