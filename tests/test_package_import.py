import cnnpz


def test_import_succeeds_and_has_version():
    assert hasattr(cnnpz, "__version__")


def test_reexports_from_submodules_are_accessible():
    # spot check one re-export from each submodule's __all__
    assert callable(cnnpz.stretch)  # data
    assert callable(cnnpz.build_model)  # models
    assert callable(cnnpz.set_plot_style)  # plotting
    assert callable(cnnpz.biweight_location)  # stats
