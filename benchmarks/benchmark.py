"""
This module contains functions used to benchmark the code in this project.
"""

import statistics
import time


def benchmark_function(function, *args, number_of_runs=5, **kwargs):
    """
    Benchmark function
    ==================

    Calls a specified function with specified arguments, and times how long it takes
    for the function to finish execution. Repeats this process a specified number of
    times, and returns the output from the function, the average execution time and the
    standard deviation in the execution times.
    """
    times = []

    for _ in range(number_of_runs):
        # Start the timer.
        start_time = time.perf_counter()

        # Execute the function.
        result = function(*args, **kwargs)

        # Stop the timer.
        stop_time = time.perf_counter()
        times.append(stop_time - start_time)

    return result, statistics.mean(times), statistics.stdev(times)
