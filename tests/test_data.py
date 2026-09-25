import numpy as np

from cnnpz.data import (
    convert_data_format,
    get_bin_edges,
    make_incomplete_nir_data,
    rebin_filter,
    stretch,
    transform_data_to_XY,
)


def test_rebin_filter_conserves_total_on_matching_range():
    bin_edges = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    counts = np.array([1.0, 2.0, 3.0, 4.0])
    new_edges = np.array([0.0, 2.0, 4.0])
    result = rebin_filter(bin_edges, counts, new_edges)
    assert result.shape == (2,)
    # each new bin spans exactly two old bins -> sums of counts*width preserved
    assert np.isclose(result.sum(), (counts * np.diff(bin_edges)).sum())
    assert np.isclose(result[0], 1.0 * 1.0 + 2.0 * 1.0)
    assert np.isclose(result[1], 3.0 * 1.0 + 4.0 * 1.0)


def test_rebin_filter_partial_overlap_drops_outside_range():
    bin_edges = np.array([0.0, 1.0, 2.0, 3.0])
    counts = np.array([1.0, 1.0, 1.0])
    new_edges = np.array([-1.0, 1.0])  # only overlaps [0,1)
    result = rebin_filter(bin_edges, counts, new_edges)
    assert result.shape == (1,)
    assert np.isclose(result[0], 1.0)


def test_get_bin_edges_uniform_spacing():
    centres = np.array([1.0, 2.0, 3.0, 4.0])
    edges = get_bin_edges(centres)
    assert edges.shape == (5,)
    assert np.allclose(edges, [0.5, 1.5, 2.5, 3.5, 4.5])


def test_stretch_stays_within_unit_interval():
    x = np.linspace(0, 1, 21)
    y = stretch(x)
    assert np.all(y >= 0) and np.all(y <= 1)


def test_stretch_is_monotonic():
    x = np.linspace(0, 1, 50)
    y = stretch(x)
    assert np.all(np.diff(y) >= 0)


def test_stretch_at_center_returns_center():
    c = 0.5
    assert np.isclose(stretch(np.array([c]), c=c)[0], c)


def test_convert_data_format_shape(synthetic_df, lambda_array_cen, filter_blocks):
    out = convert_data_format(synthetic_df, lambda_array_cen, filter_blocks)
    assert out.shape == (len(synthetic_df), len(lambda_array_cen), 3)


def test_convert_data_format_normalizes_by_iband(synthetic_df, lambda_array_cen, filter_blocks):
    out = convert_data_format(synthetic_df, lambda_array_cen, filter_blocks)
    i_idx = "ugrizyJH".index("i")
    # i-band bin is normalized against itself -> exactly zero for rows without nan/inf
    row0 = 0
    assert np.isclose(out[row0, i_idx, 0], 0.0)


def test_convert_data_format_handles_no_detect_nan(synthetic_df, lambda_array_cen, filter_blocks):
    out = convert_data_format(synthetic_df, lambda_array_cen, filter_blocks, no_detect_val=0)
    u_idx = "ugrizyJH".index("u")
    # row 1 has mag_u_lsst = NaN in the fixture ("no detection" -> mag zeroed, but
    # per the function's own comment sensitivity/availability stays 1, unlike no-observation/inf)
    assert out[1, u_idx, 0] == 0
    assert out[1, u_idx, 2] == 1


def test_convert_data_format_handles_no_observation_inf(synthetic_df, lambda_array_cen, filter_blocks):
    out = convert_data_format(synthetic_df, lambda_array_cen, filter_blocks, no_obs_val=0)
    j_idx = "ugrizyJH".index("J")
    # row 2 has mag_J_roman = inf in the fixture -> no observation
    assert out[2, j_idx, 0] == 0
    assert out[2, j_idx, 2] == 0  # filter marked unavailable


def test_transform_data_to_XY_returns_redshift_by_default(synthetic_df, lambda_array_cen, filter_blocks):
    X, Y = transform_data_to_XY(synthetic_df, lambda_array_cen, filter_blocks)
    assert X.shape == (len(synthetic_df), len(lambda_array_cen), 3)
    assert np.allclose(Y.to_numpy(), synthetic_df["redshift"].to_numpy())


def test_transform_data_to_XY_missing_y_returns_zero(synthetic_df, lambda_array_cen, filter_blocks):
    _, Y = transform_data_to_XY(synthetic_df, lambda_array_cen, filter_blocks, missingY=True)
    assert Y == 0


def test_transform_data_to_XY_stretch_changes_values(synthetic_df, lambda_array_cen, filter_blocks):
    X_stretched, _ = transform_data_to_XY(synthetic_df, lambda_array_cen, filter_blocks, apply_stretch=True)
    X_raw, _ = transform_data_to_XY(synthetic_df, lambda_array_cen, filter_blocks, apply_stretch=False)
    assert not np.allclose(X_stretched[:, :, 0], X_raw[:, :, 0])


def test_make_incomplete_nir_data_shape(synthetic_df, lambda_array_cen, filter_blocks):
    X, Y = make_incomplete_nir_data(synthetic_df, lambda_array_cen, filter_blocks)
    assert X.shape == (len(synthetic_df), len(lambda_array_cen), 3)
    assert len(Y) == len(synthetic_df)


def test_make_incomplete_nir_data_blanks_some_nir_bands(synthetic_df, lambda_array_cen, filter_blocks):
    np.random.seed(0)  # pandas .sample() with no random_state draws from the numpy global RNG
    X, _ = make_incomplete_nir_data(synthetic_df, lambda_array_cen, filter_blocks)
    j_idx = "ugrizyJH".index("J")
    h_idx = "ugrizyJH".index("H")
    # some (not necessarily all) rows should have had J/H marked unavailable
    assert X[:, j_idx, 2].min() == 0
    assert X[:, h_idx, 2].min() == 0
