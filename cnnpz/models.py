import json
import os
import pickle

import numpy as np
import tensorflow as tf
from sklearn.model_selection import KFold
from tensorflow.keras import callbacks, layers, models

from .io import load_filter_curves
from .qp_output import package_mixture_predictions, package_predictions

__all__ = [
    "build_model",
    "build_model_v2",
    "gaussian_nll",
    "ensemble_predict",
    "save_ensemble",
    "load_ensemble",
    "save_ensemble_file",
    "load_ensemble_file",
    "train_ensembles",
    "fine_tune_pre_trained_model",
]


# Smallest predicted width, in normalised redshift units (multiply by Y_std for redshift)
SIGMA_FLOOR = 1e-3


def build_model(input_shape, n_outputs=1):
    # 6 layers
    # n_outputs=1: point redshift (train with MSE); n_outputs=2: Gaussian mean and raw width (train
    # with gaussian_nll)
    model = models.Sequential(
        [
            layers.Conv1D(32, kernel_size=3, activation="relu", padding="same", input_shape=input_shape),
            layers.MaxPooling1D(),
            layers.Conv1D(64, kernel_size=3, activation="relu", padding="same"),
            layers.MaxPooling1D(),
            layers.Conv1D(128, kernel_size=3, activation="relu", padding="same"),
            layers.MaxPooling1D(),
            layers.Conv1D(256, kernel_size=3, activation="relu", padding="same"),
            # layers.MaxPooling1D(),
            layers.Conv1D(512, kernel_size=3, activation="relu", padding="same"),
            layers.Conv1D(512, kernel_size=3, activation="relu", padding="same"),
            layers.GlobalAveragePooling1D(),
            layers.Dense(64, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(n_outputs),
        ]
    )
    return model


def build_model_v2(input_shape, n_outputs=1):
    # 5 layers
    model = models.Sequential(
        [
            layers.Conv1D(32, kernel_size=3, activation="relu", padding="same", input_shape=input_shape),
            layers.MaxPooling1D(),
            layers.Conv1D(64, kernel_size=3, activation="relu", padding="same"),
            layers.MaxPooling1D(),
            layers.Conv1D(128, kernel_size=3, activation="relu", padding="same"),
            layers.MaxPooling1D(),
            layers.Conv1D(256, kernel_size=3, activation="relu", padding="same"),
            layers.MaxPooling1D(),
            layers.Conv1D(512, kernel_size=3, activation="relu", padding="same"),
            layers.GlobalAveragePooling1D(),
            layers.Dense(64, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(n_outputs),
        ]
    )
    return model


def _softplus(x):
    return np.logaddexp(0, x)


def gaussian_nll(y_true, y_pred, beta=0.0):
    """
    Gaussian negative log-likelihood loss for models with two outputs per object: the mean and a raw
    width, mapped to sigma = softplus(raw) + SIGMA_FLOOR so it stays positive.

    beta > 0 gives the beta-NLL of Seitzer et al. (2022): each object's loss is weighted by
    sigma**(2 * beta), held constant (no gradient through the weight). Plain NLL (beta = 0) scales
    the gradient on the mean by 1 / sigma**2, so objects with large predicted widths barely train
    it; beta = 1 gives the mean an MSE-like gradient, beta = 0.5 is the paper's recommendation.
    The weight does not change the optimal sigma, so the widths stay calibrated.
    """
    y_true = tf.reshape(tf.cast(y_true, y_pred.dtype), [-1])
    mu = y_pred[:, 0]
    sigma = tf.nn.softplus(y_pred[:, 1]) + SIGMA_FLOOR
    nll = tf.math.log(sigma) + 0.5 * tf.square((y_true - mu) / sigma)
    if beta > 0:
        nll = nll * tf.stop_gradient(sigma ** (2 * beta))
    return tf.reduce_mean(nll)


def _gaussian_loss(beta):
    """gaussian_nll as a Keras loss (y_true, y_pred) with beta fixed."""
    if beta == 0:
        return gaussian_nll

    def loss(y_true, y_pred):
        return gaussian_nll(y_true, y_pred, beta=beta)

    loss.__name__ = f"beta_nll_{beta:g}"
    return loss


def _is_gaussian(model):
    """True for models predicting a Gaussian (mean and width) rather than a point redshift."""
    return model.output_shape[-1] == 2


# --- Prediction: average across all models ---
def ensemble_predict(trained_models, X_test, ids=None):
    """
    Predict redshifts with an ensemble of models and package the result as a
    per-object p(z).

    Parameters
    ----------
    trained_models : list of (model, Y_mean, Y_std)
        Ensemble members, as produced by :func:`train_ensembles`.
    X_test : array-like
        Input features to predict on.
    ids : array-like, optional
        Per-object identifiers. Defaults to the sample order index (0..N-1).

    Returns
    -------
    qp.Ensemble or pandas.DataFrame
        For point-redshift members: a qp Ensemble of per-object Gaussians whose mean and std are
        those of the members' predictions (see :func:`cnnpz.qp_output.package_predictions`).
        For Gaussian members (two outputs, trained with :func:`gaussian_nll`): a qp Ensemble of
        per-object equal-weight Gaussian mixtures, one component per member (see
        :func:`cnnpz.qp_output.package_mixture_predictions`). Without qp, a DataFrame with
        "object_id", "mean", "std" columns.
    """
    if all(_is_gaussian(model) for model, _, _ in trained_models):
        means, sigmas = [], []
        for model, Y_mean, Y_std in trained_models:
            out = model.predict(X_test)
            means.append(out[:, 0] * Y_std + Y_mean)  # denormalise
            sigmas.append((_softplus(out[:, 1]) + SIGMA_FLOOR) * Y_std)
        return package_mixture_predictions(np.stack(means, axis=1), np.stack(sigmas, axis=1), ids=ids)

    predictions = []

    for model, Y_mean, Y_std in trained_models:
        # predict in normalised space, then denormalise
        y_pred_norm = model.predict(X_test)
        y_pred = y_pred_norm * Y_std + Y_mean  # denormalise
        predictions.append(y_pred)

    # average predictions across all models
    y_pred_mean = np.mean(predictions, axis=0)
    y_pred_std = np.std(predictions, axis=0)

    return package_predictions(y_pred_mean, y_pred_std, ids=ids)


def save_ensemble(trained_models, save_dir=""):
    """
    Save all ensemble models and their normalisation parameters.
    trained_models: list of (model, Y_mean, Y_std) tuples
    """
    os.makedirs(save_dir, exist_ok=True)

    # Save each model separately
    norm_params = []
    for fold, (model, Y_mean, Y_std) in enumerate(trained_models):
        # Save model weights
        model_path = os.path.join(save_dir, f"model_fold_{fold+1}.keras")
        model.save(model_path)
        print(f"Saved fold {fold+1} model to {model_path}")

        # Collect normalisation parameters
        norm_params.append({"fold": fold + 1, "Y_mean": float(Y_mean), "Y_std": float(Y_std)})

    # Save normalisation parameters as a single JSON file
    norm_path = os.path.join(save_dir, "norm_params.json")
    with open(norm_path, "w") as f:
        json.dump(norm_params, f, indent=2)
    print(f"Saved normalisation parameters to {norm_path}")


def load_ensemble(save_dir=""):
    """
    Load all ensemble models and their normalisation parameters.
    Returns: list of (model, Y_mean, Y_std) tuples
    """
    # Load normalisation parameters
    norm_path = os.path.join(save_dir, "norm_params.json")
    with open(norm_path, "r") as f:
        norm_params = json.load(f)

    # Load each model
    trained_models = []
    for params in norm_params:
        fold = params["fold"]
        model_path = os.path.join(save_dir, f"model_fold_{fold}.keras")

        model = tf.keras.models.load_model(model_path, compile=False)  # inference only; no custom loss needed
        Y_mean = params["Y_mean"]
        Y_std = params["Y_std"]

        trained_models.append((model, Y_mean, Y_std))
        print(f"Loaded fold {fold} model from {model_path}")

    return trained_models


def _architecture_json(model):
    """Model architecture as JSON, without compile settings (loss, optimiser), which inference does not need."""
    architecture = json.loads(model.to_json())
    architecture.pop("compile_config", None)
    return json.dumps(architecture)


def save_ensemble_file(path, trained_models, bands, ref_band, config):
    """
    Save an ensemble and everything needed to rebuild its inputs into a single pickle file.

    trained_models: list of (model, Y_mean, Y_std) tuples, as returned by train_ensembles.
    bands, ref_band, config: the feature settings the ensemble was trained with (see
        cnnpz.catalog_to_XY). The filter curves are loaded and stored too, so estimation does not
        depend on the filter files still existing.

    Each member is stored as its architecture (JSON) and weights rather than a pickled Keras object.
    """
    payload = {
        "members": [
            {
                "architecture": _architecture_json(model),
                "weights": model.get_weights(),
                "Y_mean": float(Y_mean),
                "Y_std": float(Y_std),
            }
            for model, Y_mean, Y_std in trained_models
        ],
        "features": {
            "bands": bands,
            "ref_band": ref_band,
            "config": config,
            "filter_curves": load_filter_curves(bands),
        },
    }
    with open(path, "wb") as f:
        pickle.dump(payload, f)


def load_ensemble_file(path):
    """
    Load an ensemble saved with save_ensemble_file.

    Returns (trained_models, features): trained_models is a list of (model, Y_mean, Y_std) tuples
    for ensemble_predict, and features holds the stored settings as keyword arguments for
    catalog_to_XY, i.e. ``X, _ = catalog_to_XY(df, **features)``.
    """
    with open(path, "rb") as f:
        payload = pickle.load(f)

    trained_models = []
    for member in payload["members"]:
        # files saved before compile settings were dropped still name the custom loss
        model = tf.keras.models.model_from_json(member["architecture"], custom_objects={"gaussian_nll": gaussian_nll})
        model.set_weights(member["weights"])
        trained_models.append((model, member["Y_mean"], member["Y_std"]))

    return trained_models, payload["features"]


def train_ensembles(
    build_model_func, X, Y, N_SPLITS=10, EPOCHS=100, BATCH_SIZE=256, random_state=42, gaussian=False, beta=0.0
):
    """
    Train one model per K-fold split. With gaussian=True each model predicts a Gaussian (mean and
    width, built with n_outputs=2) and is trained with gaussian_nll, as beta-NLL when beta > 0; otherwise a
    point redshift
    trained with MSE.
    """
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=random_state)

    trained_models = []
    histories = []

    for fold, (idx_train, idx_val) in enumerate(kf.split(X)):
        print(f"\n--- Fold {fold+1}/{N_SPLITS} ---")

        # Clear session before each fold
        tf.keras.backend.clear_session()

        # ── Callbacks ─────────────────────────────────────────────────────────────────
        early_stop = callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
        reduce_lr = callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1)

        # Split data using fold indices
        X_train, X_val = X[idx_train], X[idx_val]
        Y_train, Y_val = Y[idx_train], Y[idx_val]

        # Normalise Y using train statistics only
        # Y_mean, Y_std = Y_train.mean(), Y_train.std()
        Y_mean = 0
        Y_std = 3
        Y_train_norm = (Y_train - Y_mean) / Y_std
        Y_val_norm = (Y_val - Y_mean) / Y_std

        # Build fresh model for each fold
        if gaussian:
            model = build_model_func(input_shape=X_train.shape[1:], n_outputs=2)
            model.compile(optimizer="adam", loss=_gaussian_loss(beta))
        else:
            model = build_model_func(input_shape=X_train.shape[1:])
            model.compile(optimizer="adam", loss="mse")

        # Train
        history = model.fit(
            X_train,
            Y_train_norm,
            validation_data=(X_val, Y_val_norm),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=[early_stop, reduce_lr],
            verbose=2,
        )

        trained_models.append((model, Y_mean, Y_std))  # save model + its normalisation
        histories.append(history)

    return trained_models, histories


def fine_tune_pre_trained_model(
    X_fortrain, Y_fortrain, pretrained_models=None, model_root="", nlayers_forzen=4, beta=0.0
):
    # need to provide either the model or the model_dir to load the model
    # will always load model if both are provided

    # Each member keeps the Y normalisation it was pre-trained with, so its output layer stays on the
    # scale it learned
    if model_root != "":
        with open(os.path.join(model_root, "norm_params.json"), "r") as f:
            pretrained_norms = [(p["Y_mean"], p["Y_std"]) for p in json.load(f)]
    elif pretrained_models is not None:
        pretrained_norms = [(Y_mean, Y_std) for _, Y_mean, Y_std in pretrained_models]
    else:
        raise ValueError("provide either pretrained_models or model_root")

    # one fold per pre-trained member
    n_splits = len(pretrained_norms)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    trained_models = []
    histories = []

    for fold, (idx_train, idx_val) in enumerate(kf.split(X_fortrain)):
        print(f"\n--- Fold {fold+1}/{n_splits} ---")
        tf.keras.backend.clear_session()

        # ── Callbacks ─────────────────────────────────────────────────────────────────
        early_stop = callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
        reduce_lr = callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1)

        # Split data using fold indices
        X_train, X_val = X_fortrain[idx_train], X_fortrain[idx_val]
        Y_train, Y_val = Y_fortrain[idx_train], Y_fortrain[idx_val]

        if model_root != "":
            model = tf.keras.models.load_model(os.path.join(model_root, f"model_fold_{fold+1}.keras"), compile=False)
        else:
            model = pretrained_models[fold][0]

        # Step 2: optionally freeze early layers (keep low-level features fixed)
        for layer in model.layers[:nlayers_forzen]:  # freeze first 4 layers
            layer.trainable = False

        # same normalization as the pre-training
        Y_mean, Y_std = pretrained_norms[fold]
        Y_train_norm = (Y_train - Y_mean) / Y_std
        Y_val_norm = (Y_val - Y_mean) / Y_std

        # Step 3: recompile with a LOWER learning rate — important
        # too high a learning rate will destroy what the model already learned
        # (the loss follows the pre-trained output: Gaussian models keep training on gaussian_nll, with beta)
        gaussian = _is_gaussian(model)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),  # much lower than default 1e-3
            loss=_gaussian_loss(beta) if gaussian else "mse",
            metrics=[] if gaussian else ["mae"],
        )

        # Step 4: train on new data
        history = model.fit(
            X_train,
            Y_train_norm,
            validation_data=(X_val, Y_val_norm),
            epochs=60,  # fewer epochs than original training
            batch_size=256,
            callbacks=[early_stop, reduce_lr],
            verbose=2,
        )

        trained_models.append((model, Y_mean, Y_std))  # save model + its normalisation
        histories.append(history)

    return trained_models, histories
