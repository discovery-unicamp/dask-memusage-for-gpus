#!/usr/bin/env python3

import asyncio
import time
from contextlib import suppress
from dataclasses import dataclass
from threading import Thread

from dask_memusage_for_gpus import utils


@dataclass
class GPUProcess(dict):
    """ Object that represents a Process using GPU. """
    pid: int
    name: str
    memory_used: int


class Worker(Thread):
    """ Worker stanza to fetch GPU used memory
    """
    def __init__(self, scheduler_address: str, interval: int):
        super().__init__()

        self._scheduler_address = scheduler_address
        self._interval = interval

        # create other internal variables
        self._loop = None
        self._poll_task = None

    def run(self):
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
        self._loop.call_soon_threadsafe(self._loop.stop)

    def cancel(self):
        self._task.cancel()

    async def _memory_loop(self):
        while True:
            processes = utils.generate_gpu_proccesses()

            await asyncio.sleep(self._interval)
