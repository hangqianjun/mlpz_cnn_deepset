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
    combine them into a filter bank that avoids double-counting overlapping filters.

    filter_curves: dict[str -> (M, 2) array] raw curves keyed by band letter.
    bands: sequence of band letters fixing row order, e.g. "ugrizyJH".
    lambda_common: (L,) shared wavelength grid.

    Math
    ----
    Let T_b(lambda) be band b's raw transmission curve, interpolated onto lambda_common
    (zero outside its native support). Three quantities are built from it, each an
    (n_bands, L) array:

    1. ownership_b(lambda) = T_b(lambda) / sum_b' T_b'(lambda)
       (0 where the denominator is 0). At every wavelength this is a partition of unity
       across bands: sum_b ownership_b(lambda) = 1 wherever any band has transmission
       there. It answers "of all the transmission present at this wavelength, what
       fraction belongs to band b?" -- and is what lets overlapping bands split credit
       instead of each counting the full raw value (double-counting).

    2. shape_b(lambda) = T_b(lambda) / max_lambda T_b(lambda)
       Each band's own transmission curve, rescaled to peak at 1. This preserves a
       band's internal structure (its jaggedness) instead of letting it flatten to a
       top-hat wherever it is the sole contributor at a wavelength (where
       ownership_b(lambda) = T_b(lambda)/T_b(lambda) = 1 regardless of T_b's shape).

    3. filters_array_b(lambda) = ownership_b(lambda) * shape_b(lambda)
       This is the final filter bank: bounded in [0, 1], and order-1 wherever a single
       band dominates a wavelength (ownership_b = 1 there, so filters_array_b = shape_b,
       which peaks at 1). This is deliberately NOT further rescaled by a global integral
       constant: dividing by integral over lambda_common of sum_b filters_array_b(lambda)
       (~15000, the wavelength range in Angstrom) would shrink every entry by ~1e-4-1e-5,
       collapsing the per-galaxy amplitude signal built downstream (amplitudes @
       filters_binned in convert_data_format) to near-zero variance and stalling CNN
       training -- a real regression hit once, worth flagging so it isn't reintroduced.
       That kind of "integrates to 1" normalization is fine for a standalone plot of the
       filter shapes (rescale at plot time instead), but must not leak into the operator
       that multiplies real per-band magnitudes.

    Returns (filters_array, ownership), each shape (n_bands, L). `ownership` alone
    (without the shape term) is also returned because it is exactly what a later
    per-galaxy "was this band observed here" flag should be propagated through to get a
    coverage channel (see convert_data_format) -- filters_array's shape term would
    distort a 0/1 coverage flag into something no longer bounded in [0, 1].
    """
    raw_filters = np.array(
        [np.interp(lambda_common, filter_curves[b][:, 0], filter_curves[b][:, 1], left=0, right=0) for b in bands]
    )

    total_transmission = raw_filters.sum(axis=0)
    ownership = np.divide(raw_filters, total_transmission, out=np.zeros_like(raw_filters), where=total_transmission > 0)

    shape = np.divide(
        raw_filters,
        raw_filters.max(axis=1, keepdims=True),
        out=np.zeros_like(raw_filters),
        where=raw_filters.max(axis=1, keepdims=True) > 0,
    )
    filters_array = ownership * shape

    return filters_array, ownership


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


def build_binned_filter_operator(filters_array, bin_idx, n_bins):
    """
    Turn per-lambda-point filter curves into a fixed (n_bands, n_bins) linear operator
    that averages each band's curve within each wavelength bin. Folding this into a
    matrix (rather than binning per galaxy) lets the whole dataset be transformed with a
    single matmul: amplitudes (N, n_bands) @ filters_binned -> (N, n_bins).
    """
    L = filters_array.shape[1]
    counts = np.bincount(bin_idx, minlength=n_bins)
    W = np.zeros((n_bins, L))
    W[bin_idx, np.arange(L)] = 1.0
    W = W / np.maximum(counts, 1)[:, None]
    return filters_array @ W.T


def convert_data_format(
    df,
    filters_binned,
    bin_idx,
    lambda_bin_centers,
    bands,
    ownership_binned,
    no_detect=np.nan,
    no_obs=np.inf,
    no_detect_val=0,
    no_obs_val=0,
):
    """
    Vectorized over all galaxies.

    df: galaxy dataframe with mag_{b}_lsst / mag_{b}_roman columns for b in `bands`.
    filters_binned: (n_bands, n_bins) from build_binned_filter_operator(filters_array, ...).
    ownership_binned: (n_bands, n_bins) from build_binned_filter_operator(ownership, ...).
    bin_idx: unused directly here (kept for signature parity with the binning helpers);
        n_bins is inferred from lambda_bin_centers.
    bands: band letters matching filters_binned/ownership_binned row order and df's mag
        columns, e.g. "ugrizyJH".

    no_detect_val/no_obs_val are accepted for parity with the historical block-based
    signature but don't map onto a post-binning fill: each output bin here is a weighted
    blend of possibly several bands rather than one band's passthrough, so a missing
    band's amplitude is zeroed before binning (see below) rather than the output bin
    being overwritten afterwards.

    Returns X, shape (N, n_bins, 3):
      channel 0: binned, i-band-normalized, amplitude-weighted curve value.
      channel 1: wavelength-bin position label (0..1), identical across galaxies.
      channel 2: coverage/availability mask in [0, 1] (1 = bin fully backed by observed
        bands, 0 = bin has no observed-band support, fractional at the boundary between
        an observed and unobserved band).
    """
    n_bins = len(lambda_bin_centers)
    n_galaxies = len(df)

    roman_bands = "JH"
    mag_columns = [f"mag_{b}_roman" if b in roman_bands else f"mag_{b}_lsst" for b in bands]

    amplitudes = df[mag_columns].to_numpy()
    amplitudes_normed = amplitudes - df["mag_i_lsst"].to_numpy()[:, None]  # normalize by i-band mag

    ind_nan = np.isnan(amplitudes_normed)  # no detection
    ind_inf = np.isinf(amplitudes_normed)  # no observation

    amplitudes_clean = np.where(ind_nan | ind_inf, 0.0, amplitudes_normed)

    mags_data = amplitudes_clean @ filters_binned  # (N, n_bins)

    observed = (~ind_inf).astype(float)  # 1 if band was observed (nan still counts, inf does not)
    coverage = np.clip(observed @ ownership_binned, 0, 1)  # (N, n_bins)

    wave_labels = np.arange(n_bins)
    wave_labels = wave_labels / wave_labels[-1]
    lambda_labels = np.outer(np.ones(n_galaxies), wave_labels)

    transformed_df = np.stack([mags_data, lambda_labels, coverage], axis=-1)
    return transformed_df


def transform_data_to_XY(
    data,
    filters_binned,
    ownership_binned,
    bin_idx,
    lambda_bin_centers,
    bands,
    apply_stretch=True,
    c=0.8,
    k=20,
    missingY=False,
):
    data_transformed = convert_data_format(data, filters_binned, bin_idx, lambda_bin_centers, bands, ownership_binned)
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
    ownership_binned,
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
        data_copy, filters_binned, ownership_binned, bin_idx, lambda_bin_centers, bands, apply_stretch=apply_stretch
    )
    return X_misnir, Y_misnir
