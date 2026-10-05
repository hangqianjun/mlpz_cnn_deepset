import json
import os

import numpy as np
import tensorflow as tf
from sklearn.model_selection import KFold
from tensorflow.keras import callbacks, layers, models

from .qp_output import package_predictions

__all__ = [
    "build_model",
    "build_model_v2",
    "ensemble_predict",
    "save_ensemble",
    "load_ensemble",
    "train_ensembles",
    "fine_tune_pre_trained_model",
]


def build_model(input_shape):
    # 6 layers
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
            layers.Dense(1),
        ]
    )
    return model


def build_model_v2(input_shape):
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
            layers.Dense(1),
        ]
    )
    return model


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
        A qp Ensemble of per-object Gaussians (mean/std from the ensemble) if
        qp is installed, otherwise a DataFrame with "id", "mean", "std"
        columns. See :func:`cnnpz.qp_output.package_predictions`.
    """
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

        model = tf.keras.models.load_model(model_path)
        Y_mean = params["Y_mean"]
        Y_std = params["Y_std"]

        trained_models.append((model, Y_mean, Y_std))
        print(f"Loaded fold {fold} model from {model_path}")

    return trained_models


def train_ensembles(build_model_func, X, Y, N_SPLITS=5, EPOCHS=100, BATCH_SIZE=256, random_state=42):
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


def fine_tune_pre_trained_model(X_fortrain, Y_fortrain, pretrained_models=None, model_root="", nlayers_forzen=4):
    # need to provide either the model or the model_dir to load the model
    # will always load model if both are provided

    n_splits = 5 if pretrained_models is None else len(pretrained_models)
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
            model = tf.keras.models.load_model(model_root + f"model_fold_{fold+1}.keras")
        else:
            model = pretrained_models[fold][0]

        # Step 2: optionally freeze early layers (keep low-level features fixed)
        for layer in model.layers[:nlayers_forzen]:  # freeze first 4 layers
            layer.trainable = False

        # same normalization as the pre-training
        Y_mean = 0
        Y_std = 3
        # Normalise Y using train statistics only
        Y_mean, Y_std = Y_train.mean(), Y_train.std()
        Y_train_norm = (Y_train - Y_mean) / Y_std
        Y_val_norm = (Y_val - Y_mean) / Y_std

        # Step 3: recompile with a LOWER learning rate — important
        # too high a learning rate will destroy what the model already learned
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),  # much lower than default 1e-3
            loss="mse",
            metrics=["mae"],
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
