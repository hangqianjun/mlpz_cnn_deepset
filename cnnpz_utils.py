import sys

from packaging import version
import sklearn
from sklearn.model_selection import KFold, train_test_split

assert version.parse(sklearn.__version__) >= version.parse("1.0.1")

import tensorflow as tf

assert version.parse(tf.__version__) >= version.parse("2.8.0")

#import tensorflow_probability as tfp

import matplotlib.pyplot as plt

plt.rc('font', size=14)
plt.rc('axes', labelsize=14, titlesize=14)
plt.rc('legend', fontsize=14)
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)

import pandas as pd
import h5py
import numpy as np

import json
import os

from matplotlib import colors
import pickle
from scipy.stats import sigmaclip
import matplotlib.gridspec as gridspec

from tensorflow.keras import layers, models, callbacks


def rebin_filter(bin_edges, counts, new_edges):
  new_counts = []
  for i in range(len(new_edges) - 1):
      lo, hi = new_edges[i], new_edges[i+1]
      total = 0.0
      for j, (b0, b1, c) in enumerate(
              zip(bin_edges[:-1], bin_edges[1:], counts)):
          overlap = max(0, min(hi, b1) - max(lo, b0))
          total  += c * overlap#/(b1 - b0)  # proportional share
      new_counts.append(total)
  return np.array(new_counts)

def get_bin_edges(centres):
    step = centres[1] - centres[0]          # uniform spacing
    inner_edges = (centres[:-1] + centres[1:]) / 2
    first_edge  = centres[0]  - step / 2
    last_edge   = centres[-1] + step / 2
    return np.concatenate([[first_edge], inner_edges, [last_edge]])


def convert_data_format(df, lambda_array_cen, filter_blocks, no_detect = np.nan, no_obs = np.inf, no_detect_val = 0,
                       no_obs_val = 0):
    # no detection: setting to nan, sensitivity = 1
    # no observation: setting to inf, sensitivity = 0
    # normalize by i-band mag
    num_galaxies = len(df)
    num_wavelength_bins = len(lambda_array_cen)
    
    mags_data = np.zeros((num_galaxies, num_wavelength_bins))
    filter_availability_data = np.zeros((num_galaxies, num_wavelength_bins))
    
    for b in "ugrizy":
        ind = filter_blocks[b].astype(bool)
        mags_data[:, ind] = df[f'mag_{b}_lsst'].to_numpy()[:, None]
        filter_availability_data[:, ind] = 1
    
    for b in "JH":
        ind = filter_blocks[b].astype(bool)
        mags_data[:, ind] = df[f'mag_{b}_roman'].to_numpy()[:, None]
        filter_availability_data[:, ind] = 1

    mags_data -=  df[f'mag_i_lsst'].to_numpy()[:, None] # normalize by i-band mag
    
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
    wave_labels = wave_labels/wave_labels[-1]
    
    lambda_labels = np.outer(np.ones(num_galaxies), wave_labels)
    
    transformed_df = [mags_data, lambda_labels, filter_availability_data]
    transformed_df = np.stack(transformed_df, axis=-1)
    
    return transformed_df


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
    return np.sqrt(biweight_midvariance(
        data, c=c, ignore_nan=ignore_nan, modify_sample_size=modify_sample_size
    ))


def get_biweight_mean_sigma_outlier(subset, nclip= 3, abs_out_thresh=0.2):
    subset_clip, _, _ = sigmaclip(subset, low=3, high=3)
    for _j in range(nclip):
        subset_clip, _, _ = sigmaclip(subset_clip, low=3, high=3)

    mean = biweight_location(subset_clip)
    std = biweight_scale(subset_clip)
    #mean = np.mean(subset_clip)
    #std = np.std(subset_clip)
    #outlier_rate = np.sum(np.abs(subset) > 3 * biweight_scale(subset_clip)) / len(
    #    subset
    #)
    outlier_rate = np.sum(np.abs(subset) > 3 * np.std(subset_clip)) / len(
        subset
    )
    abs_outlier_rate = np.sum(np.abs(subset) > abs_out_thresh) / len(
        subset
    )

    return (
        mean,
        std / np.sqrt(len(subset_clip)),
        std,
        outlier_rate,
        abs_outlier_rate,
    )



def get_all_stats(y_train, y_pred, imag_data, save=True, saveroot="", redshift_bins = np.linspace(0,2.5,11), imag_bins = np.linspace(18, 25.5,11)):
    Y = y_train.to_numpy()
    Y2 = y_pred.flatten()
    dz = (Y2 - Y)/(1+Y)
    stats = get_biweight_mean_sigma_outlier(dz, nclip= 3, abs_out_thresh=0.2)

    # split in terms of i-mags and redshifts
    redshift_stats = []
    imag_stats = []
    for i in range(10):
        ind= (Y > redshift_bins[i]) & (Y < redshift_bins[i+1])
        redshift_stats.append(get_biweight_mean_sigma_outlier(dz[ind]))
        ind= (imag_data > imag_bins[i]) & (imag_data < imag_bins[i+1])
        imag_stats.append(get_biweight_mean_sigma_outlier(dz[ind]))

    if save == True:
        with open(saveroot, "wb") as f:
            pickle.dump([stats, redshift_stats, imag_stats], f)

    return stats, redshift_stats, imag_stats

def read_stats(fname):
    with open(fname, "rb") as f:
        data = pickle.load(f)
    stats, redshift_stats, imag_stats = data
    return stats, redshift_stats, imag_stats


def plot_stats(stats, redshift_stats, imag_stats, y_train, y_pred, redshift_bins, imag_bins, i_mag_data, save_path=None):
    Y  = y_train.to_numpy()
    Y2 = y_pred.flatten()
    dz = (Y2 - Y) / (1 + Y)

    fig = plt.figure(figsize=(15, 5))   # ← reduced height; 15/3 = 5 per column → square
    gs  = gridspec.GridSpec(
        2, 3,
        figure=fig,
        height_ratios=[2, 1],
        hspace=0.05,
        wspace=0.35
    )
    
    ax_scatter = fig.add_subplot(gs[:, 0])
    ax_zstats  = fig.add_subplot(gs[0, 1])
    ax_zdz     = fig.add_subplot(gs[1, 1], sharex=ax_zstats)
    ax_istats  = fig.add_subplot(gs[0, 2])
    ax_idz     = fig.add_subplot(gs[1, 2], sharex=ax_istats)
    
    # Only scatter panel needs to be square
    ax_scatter.set_box_aspect(1)    # ← only this one

    # ── Panel 1: scatter plot ────────────────────────────────────────────────
    mean, _mean_err, std, outlier_rate, abs_outlier_rate = stats
    mean, std, outlier_rate, abs_outlier_rate = (
        round(mean, 4), round(std, 4),
        round(outlier_rate, 4), round(abs_outlier_rate, 4)
    )

    #ax_scatter.scatter(y_train, y_pred, s=0.1, color='k')
    bin_edges = np.linspace(0,3,101)
    ax_scatter.hist2d(
        y_train,
        y_pred,
        bins=(bin_edges, bin_edges),
        norm=colors.LogNorm(),
        cmap="gray",
        )
    ax_scatter.plot([0, 3], [0, 3], 'r-')
    ax_scatter.plot([0, 3], [0 - 3*std, 3 - 3*std], 'r--')
    ax_scatter.plot([0, 3], [0 + 3*std, 3 + 3*std], 'r--')
    ax_scatter.set_xlim([0, 3])
    ax_scatter.set_ylim([0, 3])
    ax_scatter.set_xlabel('truth redshift')
    ax_scatter.set_ylabel('predicted redshift')

    abs_out_thresh = 0.2
    label = (
        rf"$\Delta z = {mean}$" + "\n" +
        rf"$\sigma z = {std}$"  + "\n" +
        rf"outlier rate (>3$\sigma$) = {outlier_rate}" + "\n" +
        f"outlier rate (>{abs_out_thresh}) = {abs_outlier_rate}"
    )
    ax_scatter.text(0.1, 1.8, label, fontsize=12,
                   bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"))

    # ── Panel 2: redshift statistics ─────────────────────────────────────────
    N_bin  = len(redshift_bins) - 1
    z_mean = np.zeros(N_bin)
    z_err  = np.zeros(N_bin)
    z_std  = np.zeros(N_bin)
    z_out  = np.zeros(N_bin)
    z_abs  = np.zeros(N_bin)
    for i in range(N_bin):
        z_mean[i], z_err[i], z_std[i], z_out[i], z_abs[i] = redshift_stats[i]

    z_centres = (redshift_bins[1:] + redshift_bins[:-1]) / 2
    ax_zstats.plot(z_centres, z_mean, '-', label="bias")
    ax_zstats.plot(z_centres, z_std,  '-', label=r"$\sigma_z$")
    ax_zstats.plot(z_centres, z_out,  '-', label="outlier rate")
    ax_zstats.legend()
    ax_zstats.set_ylabel("statistics")
    plt.setp(ax_zstats.get_xticklabels(), visible=False)

    #ax_zdz.scatter(Y, dz, s=0.1, color='k')
    dzbin_edges = np.linspace(-0.7,0.7,51)
    ax_zdz.hist2d(
        Y,
        dz,
        bins=(bin_edges, dzbin_edges),
        norm=colors.LogNorm(),
        cmap="gray",
        )
    ax_zdz.set_xlabel("redshift")
    ax_zdz.set_ylabel(r"$(z_{pred} - z_{true})/(1 + z_{true})$")
    ax_zdz.set_ylim([-0.5, 0.5])    # ← y-axis limit on lower panel

    # ── Panel 3: i-magnitude statistics ──────────────────────────────────────
    N_bin  = len(imag_bins) - 1
    i_mean = np.zeros(N_bin)
    i_err  = np.zeros(N_bin)
    i_std  = np.zeros(N_bin)
    i_out  = np.zeros(N_bin)
    i_abs  = np.zeros(N_bin)
    for i in range(N_bin):
        i_mean[i], i_err[i], i_std[i], i_out[i], i_abs[i] = imag_stats[i]

    i_centres = (imag_bins[1:] + imag_bins[:-1]) / 2
    ax_istats.plot(i_centres, i_mean, '-', label="bias")
    ax_istats.plot(i_centres, i_std,  '-', label=r"$\sigma_z$")
    ax_istats.plot(i_centres, i_out,  '-', label="outlier rate")
    ax_istats.legend()
    ax_istats.set_ylabel("statistics")
    plt.setp(ax_istats.get_xticklabels(), visible=False)

    #ax_idz.scatter(i_mag_data, dz, s=0.1, color='k')
    ibins = np.linspace(imag_bins[0], imag_bins[-1], 51)
    ax_idz.hist2d(
        i_mag_data,
        dz,
        bins=(ibins, dzbin_edges),
        norm=colors.LogNorm(),
        cmap="gray",
        )
    ax_idz.set_xlabel("i magnitude")
    ax_idz.set_ylabel(r"$(z_{pred} - z_{true})/(1 + z_{true})$")
    ax_idz.set_ylim([-0.5, 0.5])    # ← y-axis limit on lower panel

    plt.suptitle("Photo-z Statistics", fontsize=14)
    plt.tight_layout()
    plt.show()

    # Save if filename provided
    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {save_path}")


def stats_to_markdown(stats1_input, stats2_input, labels=("Mean", "Mean err", "Std", "Outlier rate", "Abs outlier rate"), skip_indices=(1,),
                     data_title=("Dataset 1", "Dataset 2")):
    # 0. Filter out skipped indices
    keep = [i for i in range(len(stats1_input)) if i not in skip_indices]
    stats1 = [stats1_input[i] for i in keep]
    stats2 = [stats2_input[i] for i in keep]
    labels = [labels[i] for i in keep]
    
    # 1. Fractional change vs dataset 1
    pct_change = [100 * (s2 - s1) / s1 if s1 != 0 else float('nan')
                  for s1, s2 in zip(stats1, stats2)]
    
    # 3. Build rows, rounded to 3 digits
    rows = []
    for label, s1, s2, pc in zip(labels, stats1, stats2, pct_change):
        rows.append(f"| {label} | {s1:.3g} | {s2:.3g} | {pc:.3g}% |")
    
    # 4. Assemble Markdown
    md = (
        f"| Statistic | {data_title[0]} | {data_title[1]} | % change |\n"
        "|---|---:|---:|---:|\n"
        + "\n".join(rows)
    )
    return md


def compare_binned_stats(redshift_bins, imag_bins, redshift_stats1, imag_stats1, redshift_stats2, imag_stats2):
    fig,axarr=plt.subplots(1, 2,figsize=(10,4))

    xlabels=["redshifts", "i magnitude"]

    for j, bins, stats1, stats2 in zip([0,1], [redshift_bins, imag_bins], 
                                      [redshift_stats1, imag_stats1], [redshift_stats2, imag_stats2]):
        plt.sca(axarr[j])
        N_bin = len(bins) - 1
        xx = (bins[1:] + bins[:-1])/2
        
        mean1 = np.zeros(N_bin)
        std1 = np.zeros(N_bin)
        outlier_rate1 = np.zeros(N_bin)
        for i in range(N_bin):
          mean1[i], _, std1[i], outlier_rate1[i], _ = stats1[i]
    
    
        mean2 = np.zeros(N_bin)
        std2 = np.zeros(N_bin)
        outlier_rate2 = np.zeros(N_bin)
        for i in range(N_bin):
          mean2[i], _, std2[i], outlier_rate2[i], _ = stats2[i]
        
        plt.plot(xx, mean1, '-',label="bias", color="C0")
        plt.plot(xx, std1, '-',label="$\sigma_z$", color="C1")
        plt.plot(xx, outlier_rate1, '-',label="outlier rate", color="C2")

        plt.plot(xx, mean2, '--', color="C0")
        plt.plot(xx, std2, '--', color="C1")
        plt.plot(xx, outlier_rate2, '--', color="C2")
        plt.legend()
        plt.ylabel("statistics")
        plt.xlabel(xlabels[j])
        plt.grid()
    plt.tight_layout()


def stretch(x, c=0.5, k=10):
    """
    x: array of values in [0,1]
    c: center of your typical fluctuation band
    k: stretch strength (try 5-50; higher = more aggressive)
    """
    x = np.asarray(x)
    u = np.where(x >= c, (x - c) / (1 - c), (x - c) / c)  # -> [-1, 1], c -> 0
    stretched = np.arcsinh(k * u) / np.arcsinh(k)          # amplify near u=0
    y = np.where(stretched >= 0, c + stretched * (1 - c), c + stretched * c)
    return np.clip(y, 0, 1)


def transform_data_to_XY(data, lambda_array_cen, filter_blocks, apply_stretch = True, c=0.8, k=20, missingY=False):
    data_transformed = convert_data_format(data, lambda_array_cen, filter_blocks)
    # now split training and validation set:
    if missingY==False:
        Y = data['redshift']
    else:
        Y=0
    X = np.copy(data_transformed)
    #X[:,:,0] /= max_mag
    if apply_stretch == True:
        X[:,:,0] = stretch(X[:,:,0], c=c, k=k)
    return X, Y


def make_incomplete_nir_data(data, frac=0.5, sub_val = np.inf, apply_stretch = False):
    subset = data.sample(frac=0.5)
    idx = subset.index
    idx = list(subset.index)
    data_copy = data.copy()
    data_copy.loc[idx, 'mag_J_roman'] = np.inf 
    data_copy.loc[idx, 'mag_H_roman'] = np.inf
    X_misnir, Y_misnir = transform_data_to_XY(data_copy, apply_stretch = apply_stretch)
    return X_misnir, Y_misnir


def visualize_the_data(X, Y, lambda_array_cen, filter_blocks, title="Example data vector"):
    delta_wave = 1/len(lambda_array_cen)
    for i, b in enumerate("ugrizyJH"):
      plt.bar(X[i,:,1] , filter_blocks[b].astype(int), color=f'C{i}', width=delta_wave, alpha=0.2,
             edgecolor='white')
    for i in range(5):
      z = Y[i]
      plt.plot(X[i,:,1], X[i,:,0], label=f"z={round(z,2)}")
      plt.plot(X[i,:,1], X[i,:,2], color='k')
    plt.plot(X[i,:,1], X[i,:,2], color='k', label="sensitivity")
    plt.ylabel("mag (normed)")
    plt.xlabel("wavelength label")
    plt.legend()
    plt.title(title)


# here code up a CNN to with a 1D filter:


def build_model(input_shape):
    # 6 layers
    model = models.Sequential([
        layers.Conv1D(32,  kernel_size=3, activation='relu',
                      padding='same', input_shape=input_shape),
        layers.MaxPooling1D(),
        layers.Conv1D(64,  kernel_size=3, activation='relu',
                      padding='same'),
        layers.MaxPooling1D(),
        layers.Conv1D(128, kernel_size=3, activation='relu',
                      padding='same'),
        layers.MaxPooling1D(),
        layers.Conv1D(256, kernel_size=3, activation='relu',
                      padding='same'),
        #layers.MaxPooling1D(),
        layers.Conv1D(512, kernel_size=3, activation='relu',
                      padding='same'),
        layers.Conv1D(512, kernel_size=3, activation='relu',
                      padding='same'),
        layers.GlobalAveragePooling1D(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(1)
    ])
    return model

def build_model_v2(input_shape):
    # 5 layers
    model = models.Sequential([
        layers.Conv1D(32,  kernel_size=3, activation='relu',
                      padding='same', input_shape=input_shape),
        layers.MaxPooling1D(),
        layers.Conv1D(64,  kernel_size=3, activation='relu',
                      padding='same'),
        layers.MaxPooling1D(),
        layers.Conv1D(128, kernel_size=3, activation='relu',
                      padding='same'),
        layers.MaxPooling1D(),
        layers.Conv1D(256, kernel_size=3, activation='relu',
                      padding='same'),
        layers.MaxPooling1D(),
        layers.Conv1D(512, kernel_size=3, activation='relu',
                      padding='same'),
        layers.GlobalAveragePooling1D(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(1)
    ])
    return model

# --- Prediction: average across all models ---
def ensemble_predict(trained_models, X_test):
    predictions = []
    
    for model, Y_mean, Y_std in trained_models:
        # predict in normalised space, then denormalise
        y_pred_norm = model.predict(X_test)
        y_pred = y_pred_norm * Y_std + Y_mean   # denormalise
        predictions.append(y_pred)
    
    # average predictions across all models
    return np.mean(predictions, axis=0), np.std(predictions, axis=0)
    

def plot_ensemble_losses(histories, ylim=(0, 0.1)):
    n_folds = len(histories)
    
    # Define colours and line styles explicitly
    style = {
        'loss':     {'color': 'blue',  'linestyle': '-'},   # solid blue
        'val_loss': {'color': 'blue',  'linestyle': '--'},  # dashed blue
        'mae':      {'color': 'red',   'linestyle': '-'},   # solid red
        'val_mae':  {'color': 'red',   'linestyle': '--'},  # dashed red
    }
    
    # n_folds panels + 1 summary panel
    fig, axes = plt.subplots(
        1, n_folds + 1,
        figsize=(5 * (n_folds + 1), 5),
        sharey=True
    )
    
    # --- Individual fold panels ---
    for fold, (ax, history) in enumerate(zip(axes[:-1], histories)):
        for metric, s in style.items():
            if metric in history.history:
                ax.plot(
                    history.history[metric],
                    label=metric,
                    color=s['color'],
                    linestyle=s['linestyle'],
                    linewidth=2
                )
        ax.set_title(f"Fold {fold+1}")
        ax.set_xlabel("Epoch")
        ax.set_ylim(ylim)
        ax.grid(True)
        ax.legend()
        if fold == 0:
            ax.set_ylabel("Loss / MAE")
    
    # --- Summary panel ---
    ax_summary = axes[-1]
    
    for metric, s in style.items():
        # collect this metric across all folds that have it
        all_folds = [
            h.history[metric]
            for h in histories
            if metric in h.history
        ]
        
        if len(all_folds) == 0:
            continue
        
        # pad to same length (early stopping may stop folds at different epochs)
        max_len = max(len(f) for f in all_folds)
        padded  = [
            np.pad(f, (0, max_len - len(f)), mode='edge')
            for f in all_folds
        ]
        
        mean   = np.mean(padded, axis=0)
        std    = np.std(padded,  axis=0)
        epochs = np.arange(max_len)
        
        # plot mean line
        ax_summary.plot(
            epochs, mean,
            label=metric,
            color=s['color'],
            linestyle=s['linestyle'],
            linewidth=2
        )
        
        # plot shaded std band around mean
        ax_summary.fill_between(
            epochs,
            mean - std,
            mean + std,
            color=s['color'],
            alpha=0.15
        )
    
    ax_summary.set_title("Summary (mean ± std)")
    ax_summary.set_xlabel("Epoch")
    ax_summary.set_ylim(ylim)
    ax_summary.grid(True)
    ax_summary.legend()
    
    plt.suptitle("Ensemble Model — Loss per Fold", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.show()


def save_ensemble(trained_models, save_dir=''):
    """
    Save all ensemble models and their normalisation parameters.
    trained_models: list of (model, Y_mean, Y_std) tuples
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Save each model separately
    norm_params = []
    for fold, (model, Y_mean, Y_std) in enumerate(trained_models):
        # Save model weights
        model_path = os.path.join(save_dir, f'model_fold_{fold+1}.keras')
        model.save(model_path)
        print(f"Saved fold {fold+1} model to {model_path}")
        
        # Collect normalisation parameters
        norm_params.append({
            'fold':   fold + 1,
            'Y_mean': float(Y_mean),
            'Y_std':  float(Y_std)
        })
    
    # Save normalisation parameters as a single JSON file
    norm_path = os.path.join(save_dir, 'norm_params.json')
    with open(norm_path, 'w') as f:
        json.dump(norm_params, f, indent=2)
    print(f"Saved normalisation parameters to {norm_path}")


def load_ensemble(save_dir=''):
    """
    Load all ensemble models and their normalisation parameters.
    Returns: list of (model, Y_mean, Y_std) tuples
    """
    # Load normalisation parameters
    norm_path = os.path.join(save_dir, 'norm_params.json')
    with open(norm_path, 'r') as f:
        norm_params = json.load(f)
    
    # Load each model
    trained_models = []
    for params in norm_params:
        fold      = params['fold']
        model_path = os.path.join(save_dir, f'model_fold_{fold}.keras')
        
        model  = tf.keras.models.load_model(model_path)
        Y_mean = params['Y_mean']
        Y_std  = params['Y_std']
        
        trained_models.append((model, Y_mean, Y_std))
        print(f"Loaded fold {fold} model from {model_path}")
    
    return trained_models

def train_ensembles(build_model_func, X, Y, N_SPLITS = 5, EPOCHS = 100, BATCH_SIZE = 256, random_state=42):

    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=random_state)
    
    trained_models = []
    histories = []

    for fold, (idx_train, idx_val) in enumerate(kf.split(X)):
        
        print(f"\n--- Fold {fold+1}/{N_SPLITS} ---")
    
        # Clear session before each fold
        tf.keras.backend.clear_session()
    
        # ── Callbacks ─────────────────────────────────────────────────────────────────
        early_stop = callbacks.EarlyStopping(
        monitor='val_loss', patience=5, restore_best_weights=True
        )
        reduce_lr = callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1
        )
        
        # Split data using fold indices
        X_train, X_val = X[idx_train], X[idx_val]
        Y_train, Y_val = Y[idx_train], Y[idx_val]
        
        # Normalise Y using train statistics only
        #Y_mean, Y_std = Y_train.mean(), Y_train.std()
        Y_mean = 0
        Y_std = 3
        Y_train_norm = (Y_train - Y_mean) / Y_std
        Y_val_norm   = (Y_val   - Y_mean) / Y_std
        
        # Build fresh model for each fold
        model = build_model_func(input_shape=X_train.shape[1:])
        model.compile(optimizer='adam', loss='mse')
        
        # Train
        history = model.fit(
            X_train, Y_train_norm,
            validation_data=(X_val, Y_val_norm),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=[early_stop, reduce_lr],
            verbose=2
        )
        
        trained_models.append((model, Y_mean, Y_std))  # save model + its normalisation
        histories.append(history)

    return trained_models, histories


def fine_tune_pre_trained_model(X_fortrain, Y_fortrain, pretrained_models = None, model_root = "", nlayers_forzen = 4):

    # need to provide either the model or the model_dir to load the model
    # will always load model if both are provided

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    trained_models = []
    histories = []
    
    for fold, (idx_train, idx_val) in enumerate(kf.split(X_fortrain)):
    
        print(f"\n--- Fold {fold+1}/5 ---")
        tf.keras.backend.clear_session()
        
        # ── Callbacks ─────────────────────────────────────────────────────────────────
        early_stop = callbacks.EarlyStopping(
        monitor='val_loss', patience=5, restore_best_weights=True
        )
        reduce_lr = callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1
        )
        
        # Split data using fold indices
        X_train, X_val = X_fortrain[idx_train], X_fortrain[idx_val]
        Y_train, Y_val = Y_fortrain[idx_train], Y_fortrain[idx_val]

        if model_root != "":
            model = tf.keras.models.load_model(model_root + f'model_fold_{fold+1}.keras')
        else:
            model = pretrained_models[fold][0]
        
        # Step 2: optionally freeze early layers (keep low-level features fixed)
        for layer in model.layers[:nlayers_forzen]:     # freeze first 4 layers
            layer.trainable = False
    
        # same normalization as the pre-training
        Y_mean = 0
        Y_std = 3
        # Normalise Y using train statistics only
        Y_mean, Y_std = Y_train.mean(), Y_train.std()
        Y_train_norm = (Y_train - Y_mean) / Y_std
        Y_val_norm   = (Y_val   - Y_mean) / Y_std
    
        
        # Step 3: recompile with a LOWER learning rate — important
        # too high a learning rate will destroy what the model already learned
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),  # much lower than default 1e-3
            loss='mse',
            metrics=['mae']
        )
        
        # Step 4: train on new data
        history = model.fit(
            X_train, Y_train_norm,
            validation_data=(X_val, Y_val_norm),
            epochs=60,           # fewer epochs than original training
            batch_size=256,
            callbacks=[early_stop, reduce_lr],
            verbose=2
        )
    
        trained_models.append((model, Y_mean, Y_std))  # save model + its normalisation
        histories.append(history)
        
    return trained_models, histories