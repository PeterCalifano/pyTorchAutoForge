import numpy as np
import torch
import pytest
from pyTorchAutoForge.evaluation.ModelProfiler import ModelProfilerHelper

# Dummy model for testing purposes
class DummyModel(torch.nn.Module):
    def forward(self, x):
        return x

@pytest.fixture
def dummy_model():
    return DummyModel()

def test_profiler_with_shape(dummy_model, capsys):
    # Test using an input shape provided as a list.
    profiler = ModelProfilerHelper(dummy_model, [2, 3])
    prof = profiler.run_prof()
    # Capture printed profiling table output.
    captured = capsys.readouterr().out.lower()
    # Verify that output includes an indication of CPU timing.
    assert "cpu" in captured
    assert profiler.last_prof is not None

def test_profiler_with_tensor(dummy_model):
    # Test using an actual tensor sample for input.
    sample = torch.randn(2, 3)
    profiler = ModelProfilerHelper(dummy_model, sample)
    prof = profiler.run_prof()
    assert profiler.last_prof is not None

def test_profiler_with_numpy(dummy_model):
    # Test using a numpy array as an input sample.
    input_array = np.random.randn(2, 3)
    profiler = ModelProfilerHelper(dummy_model, input_array)
    # This should convert the numpy array to a torch tensor and run inference.
    prof = profiler.run_prof()
    assert profiler.last_prof is not None

def test_invalid_input(dummy_model):
    # Test that providing an invalid type for input_sample raises a TypeError.
    with pytest.raises(TypeError):
        ModelProfilerHelper(dummy_model, {"invalid": "input"})

def test_export_trace(tmp_path, dummy_model, monkeypatch):
    # Test that the profiler attempts to export a chrome trace when output filename is provided.
    outfile = str(tmp_path / "trace.json")
    profiler = ModelProfilerHelper(dummy_model, [2, 3], output_prof_filename=outfile)
    
    # Prepare a flag to check whether export_chrome_trace gets called.
    trace_exported = {"called": False}
    def fake_export_chrome_trace(path):
        trace_exported["called"] = True
        # Do nothing instead of writing a file.
    
    # Run the profiler and monkeypatch the export_chrome_trace method on the returned profile object.
    original_run_prof = profiler.run_prof
    def fake_run_prof(*args, **kwargs):
        prof = original_run_prof(*args, **kwargs)
        setattr(prof, "export_chrome_trace", fake_export_chrome_trace)
        prof.export_chrome_trace(profiler.output_prof_filename)
        return prof
    monkeypatch.setattr(profiler, "run_prof", fake_run_prof)
    
    profiler.run_prof()
    assert trace_exported["called"] == True


def test_profiler_with_numpy_dump_trace(dummy_model):
    # Test using a numpy array as an input sample.
    input_array = np.random.randn(2, 3)
    outfile = "trace_dump.json"
    profiler = ModelProfilerHelper(dummy_model, input_array, output_prof_filename=outfile)
    # This should convert the numpy array to a torch tensor and run inference.
    
    prof = profiler.run_prof()
    assert profiler.last_prof is not None

    import os
    assert os.path.exists(outfile)
    # Clean up
    os.remove(outfile)



if __name__ == '__main__':
    import sys, pytest
    sys.exit(pytest.main([__file__]))
