#!/usr/bin/env python3

""" Plugin class of the GPU Memory Usage. """

import os
import time
from threading import Lock

import pandas as pd
from distributed.diagnostics.plugin import SchedulerPlugin
from distributed.scheduler import Scheduler

from dask_memusage_gpus import definitions as defs
from dask_memusage_gpus import gpu_handler as gpu


class MemoryUsageGPUsPlugin(SchedulerPlugin):
    """
    GPUs Memory Usage Scheduler Plugin class

    Parameters
    ----------
    scheduler : Scheduler
        Dask Scheduler object.
    path : string
        Path of the record file.
    filetype : string
        Type of the record file. It can be CSV, JSON or dataframe.
    interval : int
        Interval of the time to fetch the GPU used memory by the plugin
        daemon.
    mem_max : bool
        Collect maximum memory usage.
    """
    def __init__(self, scheduler: Scheduler, path: str, filetype: str,
                 interval: int, mem_max: bool):
        """ Constructor of the MemoryUsageGPUsPlugin class. """
        SchedulerPlugin.__init__(self)

        self._scheduler: Scheduler = scheduler
        self._path: str = path
        self._filetype: str = filetype.lower()
        self._interval: int = interval
        self._mem_max: bool = mem_max

        self._lock = Lock()
        self._plugin_start = time.perf_counter()

        if os.path.exists(self._path):
            # If there is an existing file, delete it.
            os.remove(self._path)

        self._setup_record()

        self._workers_thread = gpu.WorkersThread(self._scheduler.address,
                                                 self._interval,
                                                 self._mem_max)

        self._workers_thread.start()

    def _setup_record(self):
        """
        Setup the record structure.
        """
        self._record_df = pd.DataFrame(columns=["task_key",
                                                "time",
                                                "min_gpu_memory_mb",
                                                "max_gpu_memory_mb",
                                                "worker_id"])

    def _record(self, key, min_gpu_mem_usage, max_gpu_mem_usage, worker_id):
        """
        Record a new data into the target file.

        Parameters
        ----------
        key : string
            Name of the task executed by Dask.
        min_gpu_mem_usage : int
            Lowest value of the GPU memory usage.
        max_gpu_mem_usage : int
            Highest value of the GPU memory usage.
        worker_id : string
            Identification of the worker for that row.
        """
        with self._lock:
            row = {'task_key': key,
                   'time': time.perf_counter() - self._plugin_start,
                   'min_gpu_memory_mb': min_gpu_mem_usage,
                   'max_gpu_memory_mb': max_gpu_mem_usage,
                   'worker_id': worker_id}

            new_row = pd.DataFrame([row])
            self._record_df = pd.concat([self._record_df,
                                         new_row], axis=0, ignore_index=True)

            if self._filetype == defs.CSV:
                header: bool = (not os.path.exists(self._path))

                new_row.to_csv(self._path, mode='a', header=header)
            # XXX: Only CSV has the option to append
            elif self._filetype == defs.PARQUET:
                self._record_df.to_parquet(self._path)
            elif self._filetype == defs.JSON:
                self._record_df.to_json(self._path)
            elif self._filetype == defs.XML:
                self._record_df.to_xml(self._path)
            elif self._filetype == defs.EXCEL:
                self._record_df.to_excel(self._path, sheet_name='Dask GPUs', header=True)

    def transition(self, key, start, finish, *args, **kwargs):
        """
        Transition function when a task is being processed.

        Parameters
        ----------
        key: string
            Identifier of the task.
        start : string
            Start state of the transition. One of released, waiting,
            processing, memory, error.
        finish : string
            Final state of the transition.
        *args, **kwargs : Any
            More options passed when transitioning This may include
            worker ID, compute time, etc.
        """
        if start == 'processing' and finish in ("memory", "erred"):
            worker_id = kwargs["worker"]
            min_gpu_mem_usage, max_gpu_mem_usage = \
                self._workers_thread.fetch_task_used_memory(worker_id)
            self._record(key, min_gpu_mem_usage, max_gpu_mem_usage, worker_id)

    async def before_close(self):
        """
        Shutdown plugin structures before closing the scheduler.
        """
        self._workers_thread.stop()
