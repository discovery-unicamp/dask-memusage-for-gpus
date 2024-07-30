#!/usr/bin/env python3

""" Test all the structures and funtions inside gpu_handler submodule. """

import time
import unittest

from mock import patch

from dask_memusage_gpus import gpu_handler as gpu


class TestGPUHandler(unittest.TestCase):
    """ Test class for gpu_handler submodule. """
    @patch("dask_memusage_gpus.gpu_handler.Client")
    def test_workers_thread(self, client):
        """ Test a simple interaction with worker threads. """
        workers = [{'1.2.3.5': 234,
                    '1.2.3.6': 567},
                   {'1.2.3.5': 345,
                    '1.2.3.6': 678},
                   {'1.2.3.5': 234,
                    '1.2.3.6': 567}]

        client.return_value.run.side_effect = workers

        worker = gpu.WorkersThread("1.2.3.4", 1, False)
        worker.start()

        time.sleep(1)

        self.assertEqual(worker.fetch_task_used_memory('1.2.3.5'), (234, 234))
        self.assertEqual(worker.fetch_task_used_memory('1.2.3.6'), (567, 567))

        time.sleep(2)

        self.assertEqual(worker.fetch_task_used_memory('1.2.3.5'), (234, 345))
        self.assertEqual(worker.fetch_task_used_memory('1.2.3.6'), (567, 678))

        worker.cancel()
        worker.stop()

    @patch("dask_memusage_gpus.gpu_handler.Client")
    def test_workers_thread_max(self, client):
        """ Test a simple interaction with worker threads but using max aggr. memory. """
        workers = [{'1.2.3.5': 234,
                    '1.2.3.6': 567},
                   {'1.2.3.5': 345,
                    '1.2.3.6': 678},
                   {'1.2.3.5': 234,
                    '1.2.3.6': 567}]

        client.return_value.run.side_effect = workers

        worker = gpu.WorkersThread("1.2.3.4", 1, True)
        worker.start()

        time.sleep(3)

        worker.cancel()
        worker.stop()

        self.assertEqual(worker.fetch_task_used_memory('1.2.3.5'), (234, 345))
        self.assertEqual(worker.fetch_task_used_memory('1.2.3.6'), (567, 678))

    @patch("dask_memusage_gpus.gpu_handler.Client")
    def test_workers_thread_exception(self, client):
        """ Test an exception trigger during memory fetch. """
        workers = {'1.2.3.5': 234,
                   '1.2.3.6': 567}

        client.return_value.run.return_value = workers

        worker = gpu.WorkersThread("1.2.3.4", 1, False)

        self._worker_memory = None

        worker.start()

        time.sleep(2)

        worker.stop()

    @patch("dask_memusage_gpus.gpu_handler.Client")
    def test_workers_thread_no_worker(self, client):
        """ Test memory fetch when there is no worker connected. """
        workers = {}

        client.return_value.run.return_value = workers

        worker = gpu.WorkersThread("1.2.3.4", 1, True)
        worker.start()

        time.sleep(2)

        worker.cancel()
        worker.stop()

        self.assertEqual(worker.fetch_task_used_memory('1.2.3.5'), (0, 0))
        self.assertEqual(worker.fetch_task_used_memory('1.2.3.6'), (0, 0))
