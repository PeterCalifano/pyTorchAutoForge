from collections.abc import Callable
from functools import wraps
import time


def timeit_averaged(num_trials: int = 10) -> Callable:
    def timeit_averaged_(fcn_to_time: Callable) -> Callable:
        """
        Function decorator to perform averaged timing of a Callable object.
        This decorator measures the execution time of the decorated function over a number of trials
        and prints the average execution time.
        :param fcn_to_time: The function to be timed.
        :param num_trials: The number of trials to average the timing over. (defualt=10)
        :return: The wrapped function with timing functionality.
        """
        @wraps(fcn_to_time)
        def wrapper(*args, **kwargs):

            # Perform timing of the function using best counter available in time module
            total_elapsed_time = 0.0

            print(
                f'Timing function "{fcn_to_time.__name__}" averaging {num_trials} trials...')

            for idT in range(num_trials):
                start_time = time.perf_counter()
                result = fcn_to_time(*args, **kwargs)  # Returns Any
                end_time = time.perf_counter()
                elapsed_time = end_time - start_time
                print(f"\rFunction call {idT} took {elapsed_time:.6f} seconds")

                # Calculate the elapsed time
                total_elapsed_time += elapsed_time

            # Calculate the average elapsed time
            average_elapsed_time = total_elapsed_time / num_trials
            print(
                f"\nAverage time over {num_trials} trials: {average_elapsed_time:.6f} seconds")

            return result
        return wrapper
    return timeit_averaged_


def timeit_averaged_(fcn_to_time: Callable, num_trials: int = 10, *args, **kwargs) -> float:
    # Perform timing of the function using best counter available in time module
    total_elapsed_time = 0.0

    for idT in range(num_trials):

        start_time = time.perf_counter()
        out = fcn_to_time(*args, **kwargs)  # Returns Any
        end_time = time.perf_counter()

        total_elapsed_time += end_time - start_time

    # Calculate the average elapsed time
    average_elapsed_time = total_elapsed_time / num_trials
    return average_elapsed_time
