#!/usr/bin/env python3

""" Scheduler Plugin Base Module. """

import click
from distributed.scheduler import Scheduler

from dask_memusage_gpus import definitions as defs
from dask_memusage_gpus import plugin, utils


@click.command()
@click.option("--memusage-gpus-path", default=defs.DEFAULT_DATA_FILE)
@click.option("--memusage-gpus-record-type", default=defs.CSV)
@click.option("--memusage-gpus-interval", default=1)
@click.option("--memusage-gpus-max", is_flag=True)
def dask_setup(scheduler: Scheduler,
               memusage_gpus_path: str,
               memusage_gpus_type: str,
               memusage_gpus_interval: int,
               memusage_gpus_max: bool):
    """
    Setup Dask Scheduler Plugin.

    Parameters
    ----------
    scheduler : Scheduler
        Dask Scheduler object.
    memusage_gpus_path : string
        Path of the record file.
    memusage_gpus_filetype : string
        Type of the record file. It can be CSV, PARQUET, JSON, XML or EXCEL
        (default=CSV).
    memusage_gpus_interval : int
        Interval of the time to fetch the GPU used memory by the plugin
        daemon in seconds (default=1).
    memusage_gpus_max : bool
        Run plugin collection maximum memory usage.
    """
    utils.validate_file_type(memusage_gpus_type.lower())

    memory_plugin = plugin.MemoryUsageGPUsPlugin(scheduler,
                                                 memusage_gpus_path,
                                                 memusage_gpus_type,
                                                 memusage_gpus_interval,
                                                 memusage_gpus_max)
    scheduler.add_plugin(memory_plugin)
