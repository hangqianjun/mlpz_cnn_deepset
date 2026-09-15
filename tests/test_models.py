import json

import numpy as np
import pytest

from cnnpz.models import (
    build_model,
    build_model_v2,
    ensemble_predict,
    fine_tune_pre_trained_model,
    load_ensemble,
    save_ensemble,
    train_ensembles,
)

INPUT_SHAPE = (16, 3)


def _dummy_xy(n=10, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, *INPUT_SHAPE)).astype("float32")
    Y = rng.normal(loc=1.0, scale=0.5, size=n).astype("float32")
    return X, Y


@pytest.mark.parametrize("builder", [build_model, build_model_v2])
def test_build_model_output_shape(builder):
    model = builder(INPUT_SHAPE)
    X = np.zeros((4, *INPUT_SHAPE), dtype="float32")
    out = model.predict(X, verbose=0)
    assert out.shape == (4, 1)


def test_ensemble_predict_single_model_denormalizes():
    model = build_model_v2(INPUT_SHAPE)
    X = np.random.default_rng(0).normal(size=(3, *INPUT_SHAPE)).astype("float32")
    Y_mean, Y_std = 1.5, 2.0

    raw_pred = model.predict(X, verbose=0)
    mean_pred, std_pred = ensemble_predict([(model, Y_mean, Y_std)], X)

    assert mean_pred.shape == (3, 1)
    assert std_pred.shape == (3, 1)
    assert np.allclose(mean_pred, raw_pred * Y_std + Y_mean)
    assert np.allclose(std_pred, 0.0)  # single model -> no spread


def test_save_and_load_ensemble_roundtrip(tmp_path):
    model = build_model_v2(INPUT_SHAPE)
    trained_models = [(model, 0.5, 1.2)]
    X = np.random.default_rng(0).normal(size=(2, *INPUT_SHAPE)).astype("float32")
    pred_before = model.predict(X, verbose=0)

    save_dir = str(tmp_path)
    save_ensemble(trained_models, save_dir=save_dir)

    norm_params = json.loads((tmp_path / "norm_params.json").read_text())
    assert norm_params == [{"fold": 1, "Y_mean": 0.5, "Y_std": 1.2}]

    loaded = load_ensemble(save_dir=save_dir)
    assert len(loaded) == 1
    loaded_model, Y_mean, Y_std = loaded[0]
    assert (Y_mean, Y_std) == (0.5, 1.2)
    assert np.allclose(loaded_model.predict(X, verbose=0), pred_before, atol=1e-5)


def test_train_ensembles_runs_and_returns_all_folds():
    X, Y = _dummy_xy(n=10)
    trained_models, histories = train_ensembles(build_model_v2, X, Y, N_SPLITS=2, EPOCHS=1, BATCH_SIZE=4)
    assert len(trained_models) == 2
    assert len(histories) == 2
    for model, Y_mean, Y_std in trained_models:
        assert Y_mean == 0
        assert Y_std == 3
        out = model.predict(X[:2], verbose=0)
        assert out.shape == (2, 1)


def test_fine_tune_pre_trained_model_freezes_layers_and_runs():
    # fine_tune_pre_trained_model hardcodes KFold(n_splits=5) internally and
    # indexes pretrained_models[fold] for fold in 0..4, so it requires exactly
    # 5 pretrained models regardless of how they were trained (see TESTING_NOTES.md).
    X, Y = _dummy_xy(n=20)
    pretrained, _ = train_ensembles(build_model_v2, X, Y, N_SPLITS=5, EPOCHS=1, BATCH_SIZE=4)

    nlayers_forzen = 2
    fine_tuned, histories = fine_tune_pre_trained_model(
        X, Y, pretrained_models=pretrained, nlayers_forzen=nlayers_forzen
    )

    assert len(fine_tuned) == 5
    assert len(histories) == 5
    model, Y_mean, Y_std = fine_tuned[0]
    for layer in model.layers[:nlayers_forzen]:
        assert layer.trainable is False
