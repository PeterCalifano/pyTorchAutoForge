from pyTorchAutoForge.utils import timeit_averaged, timeit_averaged_
import time

@timeit_averaged(2)
def dummy_function():
    print("Dummy function called")
    time.sleep(1)

def test_timeit_averaged():
    print("Testing timeit_averaged as decorator...")
    dummy_function()

def test_timeit_averaged_wrapped():
    print("Testing timeit_averaged wrapped...")
    dummy_function()

def test_timeit_averaged_function():
    def sample_function(x, y):
        time.sleep(0.5)
        return x + y

    num_trials = 3
    average_time = timeit_averaged_(sample_function, num_trials, 2, 3)

    assert isinstance(average_time, float), "Average time should be a float"
    assert average_time > 0, "Average time should be greater than 0"
    assert average_time < 1, "Average time should be less than 1 for this test case"

def test_timeit_averaged_function_with_kwargs():
    def sample_function(x, y, delay=0.5):
        time.sleep(delay)
        return x * y

    num_trials = 2
    average_time = timeit_averaged_(sample_function, num_trials, 3, 4, delay=0.3)

    assert isinstance(average_time, float), "Average time should be a float"
    assert average_time > 0, "Average time should be greater than 0"
    assert average_time < 1, "Average time should be less than 1 for this test case"

