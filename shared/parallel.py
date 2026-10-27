"""Run a function over many items (usually output times) in parallel processes.

    from shared.parallel import parallel_map
    values = parallel_map(analyse_time, TIMES, workers=N_WORKERS)

The processes are forked, so any function can be used (notebook functions, lambdas, closures). Each process gets
threads_per_worker numpy/FFT threads (needs threadpoolctl; without it the threads are not limited).
"""
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor

try:
    from threadpoolctl import threadpool_limits
except ImportError:
    threadpool_limits = None

_thread_limit = None                 # holds the thread limit for the life of each worker process
_function = None                     # the function being mapped; forked workers inherit it, so it is never pickled


def _limit_threads(n_threads):
    global _thread_limit
    if threadpool_limits is not None:
        _thread_limit = threadpool_limits(limits=n_threads)


def _call(item):
    return _function(item)


def parallel_map(function, items, workers=None, threads_per_worker=1):
    """[function(item) for item in items], computed in up to `workers` processes (default: all cores) and
    returned in the order of items. With one worker, or one item, it runs in this process."""
    items = list(items)
    workers = min(workers or os.cpu_count() or 1, len(items))
    if workers <= 1:
        return [function(item) for item in items]
    global _function
    _function = function
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("fork"),
                             initializer=_limit_threads, initargs=(threads_per_worker,)) as executor:
        return list(executor.map(_call, items))
