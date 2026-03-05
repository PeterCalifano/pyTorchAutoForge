import pytest
import numpy as np
import os
from pathlib import Path
from pyTorchAutoForge.evaluation.ResultsPlotter import ResultsPlotterHelper, ResultsPlotterConfig, backend_module

@pytest.fixture
def setup_plotter(tmp_path: Path):
    output_dir_ = tmp_path / "plots"
    stats = {
        'prediction_err': np.random.randn(100, 3),
        'mean_prediction_err': np.random.randn(3)
    }
    config = ResultsPlotterConfig(
        save_figs=False,
        unit_scalings={'Component 0': 1.0, 'Component 1': 2.0, 'Component 2': 3.0},
        num_of_bins=50,
        colours=['red', 'green', 'blue'],
        units=['unit1', 'unit2', 'unit3'],
        entriesNames=['Component 0', 'Component 1', 'Component 2'],
        output_folder=str(output_dir_)
    )
    plotter = ResultsPlotterHelper(stats=stats, backend_module_=backend_module.SEABORN, config=config)
    yield plotter, stats, config

def test_initialization(setup_plotter):
    plotter, stats, config = setup_plotter
    assert plotter.loaded_stats == stats
    assert plotter.backend_module == backend_module.SEABORN
    assert plotter.save_figs == config.save_figs
    assert plotter.unit_scalings == config.unit_scalings
    assert plotter.num_of_bins == config.num_of_bins
    assert plotter.colours == config.colours
    assert plotter.units == config.units
    assert plotter.entriesNames == config.entriesNames
    assert str(plotter.output_folder).startswith(str(config.output_folder))

def test_histPredictionErrors(setup_plotter):
    plotter, _, _ = setup_plotter
    try:
        plotter.histPredictionErrors()
    except Exception as e:
        pytest.fail(f"histPredictionErrors raised an exception {e}")

def test_histPredictionErrors_with_custom_stats(setup_plotter):
    plotter, _, _ = setup_plotter
    custom_stats = {
        'prediction_err': np.random.randn(100, 3),
        'mean_prediction_err': np.random.randn(3)
    }
    try:
        plotter.histPredictionErrors(stats=custom_stats)
    except Exception as e:
        pytest.fail(f"histPredictionErrors with custom stats raised an exception {e}")

def test_histPredictionErrors_with_invalid_stats(setup_plotter):
    plotter, _, _ = setup_plotter
    with pytest.raises(TypeError):
        plotter.histPredictionErrors(stats="invalid_stats")


# FIXME test failing, check class method implementation.
# The assertion below fails because the stats attribute is not None!

def test_histPredictionErrors_with_missing_prediction_err(setup_plotter):
    plotter, _, _ = setup_plotter
    incomplete_stats = {'mean_prediction_err': np.random.randn(3)}
    plotter.histPredictionErrors(stats=incomplete_stats)
    assert plotter.stats is None  

def test_histPredictionErrors_with_missing_mean_prediction_err(setup_plotter):
    plotter, _, _ = setup_plotter
    incomplete_stats = {'prediction_err': np.random.randn(100, 3)}
    try:
        plotter.histPredictionErrors(stats=incomplete_stats)
    except Exception as e:
        pytest.fail(f"histPredictionErrors with missing mean_prediction_err raised an exception {e}")

def test_save_figs(setup_plotter):
    plotter, _, config = setup_plotter
    plotter.save_figs = True
    plotter.histPredictionErrors(stats=plotter.loaded_stats)
    assert os.path.isdir(plotter.output_folder)
    assert os.path.isfile(os.path.join(plotter.output_folder, "prediction_errors_all_components.png"))
