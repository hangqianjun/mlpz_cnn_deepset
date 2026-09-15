import numpy as np
import pandas as pd
import pytest

from cnnpz.stats import (
    biweight_location,
    biweight_midvariance,
    biweight_scale,
    get_all_stats,
    get_biweight_mean_sigma_outlier,
    read_stats,
    stats_to_markdown,
)


@pytest.fixture
def data_with_outliers():
    rng = np.random.default_rng(1)
    core = rng.normal(loc=5.0, scale=1.0, size=200)
    outliers = np.array([100.0, -100.0, 80.0])
    return np.concatenate([core, outliers])


def test_biweight_location_close_to_mean_without_outliers():
    rng = np.random.default_rng(2)
    x = rng.normal(loc=3.0, scale=1.0, size=500)
    assert np.isclose(biweight_location(x), 3.0, atol=0.2)


def test_biweight_location_robust_to_outliers(data_with_outliers):
    loc = biweight_location(data_with_outliers)
    assert abs(loc - 5.0) < abs(np.mean(data_with_outliers) - 5.0)


def test_biweight_midvariance_robust_to_outliers(data_with_outliers):
    robust_var = biweight_midvariance(data_with_outliers)
    assert robust_var < np.var(data_with_outliers)


def test_biweight_scale_is_sqrt_of_midvariance(data_with_outliers):
    scale = biweight_scale(data_with_outliers)
    midvar = biweight_midvariance(data_with_outliers)
    assert np.isclose(scale, np.sqrt(midvar))


def test_biweight_location_ignore_nan():
    x = np.array([1.0, 2.0, 3.0, np.nan])
    assert np.isnan(biweight_location(x, ignore_nan=False))
    assert not np.isnan(biweight_location(x, ignore_nan=True))


def test_biweight_location_empty_returns_nan():
    assert np.isnan(biweight_location(np.array([])))


def test_get_biweight_mean_sigma_outlier_shape_and_outlier_rate():
    rng = np.random.default_rng(3)
    core = rng.normal(0, 0.05, 200)
    outliers = np.full(10, 5.0)
    subset = np.concatenate([core, outliers])

    mean, mean_err, std, outlier_rate, abs_outlier_rate = get_biweight_mean_sigma_outlier(
        subset, nclip=3, abs_out_thresh=0.2
    )
    assert np.isclose(mean, 0.0, atol=0.05)
    assert outlier_rate > 0
    assert abs_outlier_rate > 0


def test_get_all_stats_default_bins_and_roundtrip(tmp_path):
    n = 200
    y_train = pd.Series(np.linspace(0.05, 2.45, n))
    imag_data = np.linspace(18.1, 25.4, n)
    y_pred = (y_train.to_numpy() + np.linspace(-0.05, 0.05, n)).reshape(-1, 1)

    save_path = tmp_path / "stats.pkl"
    stats, redshift_stats, imag_stats = get_all_stats(y_train, y_pred, imag_data, save=True, saveroot=str(save_path))

    assert len(stats) == 5
    assert len(redshift_stats) == 10
    assert len(imag_stats) == 10
    assert save_path.exists()

    loaded_stats, loaded_redshift_stats, loaded_imag_stats = read_stats(str(save_path))
    assert loaded_stats == stats
    assert loaded_redshift_stats == redshift_stats
    assert loaded_imag_stats == imag_stats


def test_stats_to_markdown_contains_labels_and_handles_zero():
    stats1 = (0.0, 0.01, 0.1, 0.02, 0.03)
    stats2 = (0.01, 0.01, 0.12, 0.02, 0.03)
    md = stats_to_markdown(stats1, stats2)
    assert "Mean" in md
    assert "Std" in md
    assert "Mean err" not in md  # skipped by default (skip_indices=(1,))
    assert "nan%" in md  # division by zero on stats1's mean=0 is guarded, not raised
