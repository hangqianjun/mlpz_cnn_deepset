import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from cnnpz.data import transform_data_to_XY
from cnnpz.plotting import (
    compare_binned_stats,
    plot_ensemble_losses,
    plot_stats,
    set_plot_style,
    visualize_the_data,
)


class _FakeHistory:
    def __init__(self, history):
        self.history = history


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


@pytest.fixture
def binned_stats(n_bins=3):
    # (mean, mean_err, std, outlier_rate, abs_outlier_rate) per bin
    return [(0.01, 0.001, 0.05, 0.02, 0.03) for _ in range(n_bins)]


def test_set_plot_style_runs():
    set_plot_style()


def test_plot_stats_smoke_and_savefig(tmp_path, binned_stats):
    n = 50
    y_train = pd.Series(np.linspace(0.05, 2.4, n))
    # plot_stats's scatter panel passes y_pred straight into hist2d (unlike the dz
    # calculation, which flattens it), so it must already be 1D here.
    y_pred = y_train.to_numpy() + 0.01
    i_mag_data = np.linspace(18.0, 25.0, n)
    redshift_bins = np.linspace(0, 2.5, 4)
    imag_bins = np.linspace(18, 25.5, 4)
    stats = (0.01, 0.001, 0.05, 0.02, 0.03)

    save_path = tmp_path / "stats.png"
    plot_stats(
        stats,
        binned_stats,
        binned_stats,
        y_train,
        y_pred,
        redshift_bins,
        imag_bins,
        i_mag_data,
        save_path=str(save_path),
    )
    assert save_path.exists()


def test_compare_binned_stats_smoke(binned_stats):
    redshift_bins = np.linspace(0, 2.5, 4)
    imag_bins = np.linspace(18, 25.5, 4)
    compare_binned_stats(redshift_bins, imag_bins, binned_stats, binned_stats, binned_stats, binned_stats)
    assert len(plt.gcf().axes) > 0


def test_visualize_the_data_smoke(synthetic_df, lambda_array_cen, filter_blocks):
    X, Y = transform_data_to_XY(synthetic_df, lambda_array_cen, filter_blocks)
    visualize_the_data(X, Y, lambda_array_cen, filter_blocks)
    assert len(plt.gcf().axes) > 0


def test_plot_ensemble_losses_smoke():
    histories = [
        _FakeHistory({"loss": [1.0, 0.8, 0.6], "val_loss": [1.1, 0.9, 0.7]}),
        _FakeHistory({"loss": [1.2, 0.9], "val_loss": [1.3, 1.0]}),
    ]
    plot_ensemble_losses(histories)
    assert len(plt.gcf().axes) > 0
