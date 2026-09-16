import warnings

import numpy as np

try:
    import qp

    QP_INSTALLED = True
except Exception:  # pragma: no cover - qp missing or unimportable in this env
    QP_INSTALLED = False

__all__ = ["QP_INSTALLED", "save_predictions"]


def save_predictions(y_pred_mean, y_pred_std, output_filename=None, ids=None):
    """
    Package ensemble predictions (mean, std) as per-object p(z).

    Each object's predicted redshift is represented as a Gaussian centred on
    ``y_pred_mean`` with width ``y_pred_std`` (the mean/std produced by
    :func:`cnnpz.models.ensemble_predict`).

    If ``qp`` is installed, this builds a ``qp.Ensemble`` of Gaussians (one per
    object) and, if ``output_filename`` is given, writes it to disk (e.g. as
    ``.hdf5``). If ``qp`` is not installed, it falls back to returning the raw
    mean/std arrays, since there is no p(z) representation to build without it.

    Parameters
    ----------
    y_pred_mean, y_pred_std : array-like
        Per-object predicted redshift mean and std, as returned by
        ``ensemble_predict``.
    output_filename : str, optional
        If given, the result is written to this path.
    ids : array-like, optional
        Per-object identifiers, stored as ancillary data on the qp Ensemble.
        Ignored if qp is not installed.

    Returns
    -------
    qp.Ensemble or dict
        A qp Ensemble of Gaussians if qp is installed, otherwise a dict with
        "mean" and "std" arrays.
    """
    y_pred_mean = np.asarray(y_pred_mean).reshape(-1, 1)
    y_pred_std = np.asarray(y_pred_std).reshape(-1, 1)

    if not QP_INSTALLED:
        warnings.warn(
            "qp is not installed; returning raw mean/std instead of a qp Ensemble. "
            "Install qp-prob to get p(z) ensembles.",
            stacklevel=2,
        )
        result = {"mean": y_pred_mean.flatten(), "std": y_pred_std.flatten()}
        if output_filename is not None:
            np.savez(output_filename, **result)
        return result

    data = {"loc": y_pred_mean, "scale": y_pred_std}
    ancil = {"ids": np.asarray(ids)} if ids is not None else None
    ensemble = qp.stats.norm.create_ensemble(data, ancil)

    if output_filename is not None:
        ensemble.write_to(output_filename)

    return ensemble
