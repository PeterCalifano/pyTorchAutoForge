import pytest
import numpy as np
import os
from pyTorchAutoForge.evaluation.ResultsPlotter import ResultsPlotter, ResultsPlotterConfig, backend_module

@pytest.fixture
def setup_plotter():
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
        output_folder='test_output'
    )
    plotter = ResultsPlotter(stats=stats, backend_module_=backend_module.SEABORN, config=config)
    yield plotter, stats, config
    if os.path.isdir('test_output'):
        for file in os.listdir('test_output'):
            os.remove(os.path.join('test_output', file))
        os.rmdir('test_output')

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
    assert plotter.output_folder == config.output_folder

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
    plotter.output_folder = 'test_output'
    plotter.histPredictionErrors()
    assert os.path.isdir('test_output')
    for entry in config.entriesNames:
        assert os.path.isfile(os.path.join('test_output', f"prediction_errors_{entry}.png"))