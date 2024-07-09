#!/usr/bin/env python3

import asyncio
import csv

import click
from distributed.diagnostics.plugin import SchedulerPlugin

from . import definitions as defs


class MemoryUsageGPUsPlugin(SchedulerPlugin):
    def __init__(self, path, filetype):
        SchedulerPlugin.__init__(self)

        self.__path = path
        self.__filetype = filetype


def validate_file_type(filetype):
    if filetype not in defs.FILE_TYPES:
        raise defs.FileTypeException(f"'{filetype}' is not a valid output file.")    


@click.command()
@click.option("--memusage-gpus-path", default=defs.DEFAULT_DATA_FILE)
@click.option("--memusage-gpus-type", default=defs.CSV)
def dask_setup(scheduler, path, filetype):
    validate_file_type(filetype)
