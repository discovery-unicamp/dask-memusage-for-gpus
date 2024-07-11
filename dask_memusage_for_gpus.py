#!/usr/bin/env python3

import asyncio
import csv

import click
from distributed.client import Client
from distributed.diagnostics.plugin import SchedulerPlugin
from distributed.scheduler import Scheduler

from . import definitions as defs


class MemoryUsageGPUsPlugin(SchedulerPlugin):
    def __init__(self, scheduler: Scheduler, path: str, filetype: str):
        SchedulerPlugin.__init__(self)

        self.scheduler = scheduler
        self.__path = path
        self.__filetype = filetype

    def transition(self, key, start, finish, *args, **kwargs):
        if start == 'processing' and finish in ("memory", "erred"):
            memory_usage = self._worker_memory.memory_for_task(worker_address)
            max_memory_usage = max(memory_usage)
            min_memory_usage = min(memory_usage)

    def before_close(self):
        pass


def validate_file_type(filetype):
    if filetype not in defs.FILE_TYPES:
        raise defs.FileTypeException(f"'{filetype}' is not a valid "
                                     "output file.")


@click.command()
@click.option("--memusage-gpus-path", default=defs.DEFAULT_DATA_FILE)
@click.option("--memusage-gpus-type", default=defs.CSV)
def dask_setup(scheduler: Scheduler, path: str, filetype: str):
    validate_file_type(filetype)

    plugin = MemoryUsageGPUsPlugin(scheduler, path, filetype)
    scheduler.add_plugin(plugin)
