import pickle

import numpy as np
from scipy.stats import sigmaclip

__all__ = [
    "biweight_location",
    "biweight_midvariance",
    "biweight_scale",
    "get_biweight_mean_sigma_outlier",
    "get_all_stats",
    "read_stats",
    "stats_to_markdown",
]


def _mad(x, M):
    """Median absolute deviation about M (unscaled)."""
    return np.median(np.abs(x - M))


def biweight_location(data, c=6.0, ignore_nan=False):
    """Robust estimate of the centre. Analogue of np.mean."""
    x = np.asarray(data, dtype=float).ravel()
    if ignore_nan:
        x = x[~np.isnan(x)]
    if x.size == 0:
        return np.nan

    M = np.median(x)
    mad = _mad(x, M)
    if mad == 0 or not np.isfinite(mad):
        return M

    u = (x - M) / (c * mad)
    mask = np.abs(u) < 1
    if not mask.any():
        return M

    u = u[mask]
    w = (1.0 - u**2) ** 2
    return M + np.sum((x[mask] - M) * w) / np.sum(w)


def biweight_midvariance(data, c=9.0, ignore_nan=False, modify_sample_size=False):
    """Robust estimate of the variance."""
    x = np.asarray(data, dtype=float).ravel()
    if ignore_nan:
        x = x[~np.isnan(x)]
    if x.size == 0:
        return np.nan

    M = np.median(x)
    mad = _mad(x, M)
    if mad == 0 or not np.isfinite(mad):
        return 0.0

    u = (x - M) / (c * mad)
    mask = np.abs(u) < 1
    u = u[mask]

    n = mask.sum() if modify_sample_size else x.size

    f1 = np.sum((x[mask] - M) ** 2 * (1.0 - u**2) ** 4)
    f2 = np.abs(np.sum((1.0 - u**2) * (1.0 - 5.0 * u**2))) ** 2
    if f2 == 0:
        return 0.0
    return n * f1 / f2


def biweight_scale(data, c=9.0, ignore_nan=False, modify_sample_size=False):
    """Robust estimate of the scatter. Analogue of np.std."""
    return np.sqrt(biweight_midvariance(data, c=c, ignore_nan=ignore_nan, modify_sample_size=modify_sample_size))


def get_biweight_mean_sigma_outlier(subset, nclip=3, abs_out_thresh=0.2):
    subset_clip, _, _ = sigmaclip(subset, low=3, high=3)
    for _j in range(nclip):
        subset_clip, _, _ = sigmaclip(subset_clip, low=3, high=3)

    mean = biweight_location(subset_clip)
    std = biweight_scale(subset_clip)
    # mean = np.mean(subset_clip)
    # std = np.std(subset_clip)
    # outlier_rate = np.sum(np.abs(subset) > 3 * biweight_scale(subset_clip)) / len(
    #    subset
    # )
    outlier_rate = np.sum(np.abs(subset) > 3 * np.std(subset_clip)) / len(subset)
    abs_outlier_rate = np.sum(np.abs(subset) > abs_out_thresh) / len(subset)

    return (
        mean,
        std / np.sqrt(len(subset_clip)),
        std,
        outlier_rate,
        abs_outlier_rate,
    )


def get_all_stats(
    y_train,
    y_pred,
    imag_data,
    save=True,
    saveroot="",
    redshift_bins=np.linspace(0, 2.5, 11),
    imag_bins=np.linspace(18, 25.5, 11),
):
    Y = np.asarray(y_train)
    Y2 = y_pred.flatten()
    dz = (Y2 - Y) / (1 + Y)
    stats = get_biweight_mean_sigma_outlier(dz, nclip=3, abs_out_thresh=0.2)

    # split in terms of i-mags and redshifts
    redshift_stats = []
    imag_stats = []
    for i in range(10):
        ind = (Y > redshift_bins[i]) & (Y < redshift_bins[i + 1])
        redshift_stats.append(get_biweight_mean_sigma_outlier(dz[ind]))
        ind = (imag_data > imag_bins[i]) & (imag_data < imag_bins[i + 1])
        imag_stats.append(get_biweight_mean_sigma_outlier(dz[ind]))

    if save is True:
        with open(saveroot, "wb") as f:
            pickle.dump([stats, redshift_stats, imag_stats], f)

    return stats, redshift_stats, imag_stats


def read_stats(fname):
    with open(fname, "rb") as f:
        data = pickle.load(f)
    stats, redshift_stats, imag_stats = data
    return stats, redshift_stats, imag_stats


def stats_to_markdown(
    stats1_input,
    stats2_input,
    labels=("Mean", "Mean err", "Std", "Outlier rate", "Abs outlier rate"),
    skip_indices=(1,),
    data_title=("Dataset 1", "Dataset 2"),
):
    # 0. Filter out skipped indices
    keep = [i for i in range(len(stats1_input)) if i not in skip_indices]
    stats1 = [stats1_input[i] for i in keep]
    stats2 = [stats2_input[i] for i in keep]
    labels = [labels[i] for i in keep]

    # 1. Fractional change vs dataset 1
    pct_change = [100 * (s2 - s1) / s1 if s1 != 0 else float("nan") for s1, s2 in zip(stats1, stats2)]

    # 3. Build rows, rounded to 3 digits
    rows = []
    for label, s1, s2, pc in zip(labels, stats1, stats2, pct_change):
        rows.append(f"| {label} | {s1:.3g} | {s2:.3g} | {pc:.3g}% |")

    # 4. Assemble Markdown
    md = f"| Statistic | {data_title[0]} | {data_title[1]} | % change |\n" "|---|---:|---:|---:|\n" + "\n".join(rows)
    return md
