#!/usr/bin/env python3

import asyncio
import logging
from collections import defaultdict
from contextlib import suppress
from dataclasses import dataclass
from threading import Thread

import dask
from distributed.client import Client

from dask_memusage_for_gpus import utils

logger = logging.getLogger(__name__)


@dataclass
class GPUProcess(dict):
    """ Object that represents a Process using GPU. """
    pid: int
    name: str
    memory_used: int


class WorkersThread(Thread):
    """
    Worker stanza to fetch GPU used memory

    Parameters
    ----------
    scheduler_address : string
        Addres of the Dask Scheduler.
    interval : int
        Interval of the time to fetch the GPU used memory by the plugin
        daemon.
    """
    def __init__(self, scheduler_address: str, interval: int):
        super().__init__()

        self._scheduler_address = scheduler_address
        self._interval = interval
        self._worker_memory = defaultdict()

        try:
            logger.setLevel(
                    dask.config.get("distributed.logging.distributed__scheduler")
            )
        except KeyError:
            logger.setLevel(logging.INFO)

        # create other internal variables
        self._loop = None
        self._task = None
        self._poll_task = None

    def run(self):
        """ Main thread loop. """

        logger.info("Memory loop thread is running.")

        self._loop = asyncio.new_event_loop()
        loop = self._loop
        asyncio.set_event_loop(loop)
        try:
            self._poll_task = asyncio.ensure_future(self._memory_loop())

            loop.run_forever()
            loop.run_until_complete(loop.shutdown_asyncgens())

            self._poll_task.cancel()
            with suppress(asyncio.CancelledError):
                loop.run_until_complete(self._poll_task)
        finally:
            loop.close()

    def stop(self):
        """ Stop the async loop event. """

        if self._loop:
            self._loop.call_soon_threadsafe(self._loop.stop)

        logger.info("Memory loop thread is stopped.")

    def cancel(self):
        """ Cancel the async task. """

        if self._task:
            self._task.cancel()

        logger.info("Memory loop thread is cancelled.")

    def fetch_task_used_memory(self, worker_address):
        """
        The GPU used memory of the finished previous task.

        Returns
        -------
        list
            Tracked memory usage per worker
        """
        if not self._worker_memory[worker_address]:
            return (0, 0)

        mem_max = max(self._worker_memory[worker_address])
        mem_min = min(self._worker_memory[worker_address])

        logger.debug("Cleaning the worker memory list.")

#        self._worker_memory[worker_address].clear()

        return (mem_min, mem_max)

    async def _memory_loop(self):
        """ Background function to monitor GPU used memory per process. """

        client = Client(self._scheduler_address, timeout=30)

        logger.debug("Main memory loop function running.")

        while True:
            worker_gpu_mem = client.run(utils.get_worker_gpu_memory_used)

            for address, memory in worker_gpu_mem.items():
                if address not in self._worker_memory:
                    self._worker_memory[address] = list()

                self._worker_memory[address].append(memory)

                logger.debug(f"Appending {memory} MiB into worker ID '{address}'.")

            await asyncio.sleep(self._interval)
