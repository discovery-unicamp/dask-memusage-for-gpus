#!/usr/bin/env python3

""" Test all the structures and funtions inside gpu_handler submodule. """

import os
import unittest

import pandas as pd
from parameterized import parameterized

try:
    import cupy as cp
    from dask_cuda import LocalCUDACluster

    CUDA_SUPPORT = True
except ImportError:
    CUDA_SUPPORT = False

from dask import compute
from dask.bag import from_sequence
from dask.distributed import Client
from mock import Mock, patch

from dask_memusage_gpus import plugin


def allocate_50mb(x):
    """Allocate 50MB of GPU Memory."""
    cp.ones((50, 1024, 1024), dtype=cp.uint8)
    return x * 2


def no_allocate(y):
    """Don't allocate any memory."""
    return y * 2


def make_bag():
    """Create a bag."""
    return from_sequence(
        [1, 2], npartitions=2
    ).map(allocate_50mb).sum().apply(no_allocate)


class TestPlugin(unittest.TestCase):
    """ Test class for plugin submodule. """
    def setUp(self):
        """ Setup test method. """
        self.path = os.path.join(os.path.dirname(__file__), "memusage")

    def tearDown(self):
        """ Tear down the test class. """
        if os.path.exists(self.path):
            os.remove(self.path)

    @parameterized.expand([
         ("csv", pd.read_csv),
         ("parquet", pd.read_parquet),
         ("json", pd.read_json),
         ("xml", pd.read_xml),
         ("excel", pd.read_excel),
     ])
    @patch('dask_memusage_gpus.gpu_handler.WorkersThread')
    def test_plugin_record_to_file(self, file, func, thread):
        """ Test recording to file. """
        thread.return_value = Mock(start=Mock(),
                                   fetch_task_used_memory=Mock(return_value=(200, 400)))

        scheduler = Mock()
        scheduler.address = '1.2.3.4'

        if file == 'excel':
            self.path += '.xlsx'

        dask_plugin = plugin.MemoryUsageGPUsPlugin(scheduler=scheduler,
                                                   path=self.path,
                                                   filetype=file,
                                                   interval=2,
                                                   mem_max=True)

        dask_plugin.transition('func', 'queued', 'processing',
                               worker='tcp://1.2.3.4:34567')
        dask_plugin.transition('func', 'processing', 'memory',
                               worker='tcp://1.2.3.4:34567')

        df = func(self.path)

        if file == 'excel':
            if os.path.exists(self.path):
                os.remove(self.path)

        self.assertEqual(len(df), 1)

        dask_plugin.before_close()

    def test_install_plugin(self):
        """ Test install plugin from scheduler. """

        if not CUDA_SUPPORT:
            raise unittest.SkipTest("No GPU support from host.")

        with LocalCUDACluster() as cluster:
            client = Client(cluster)

            dask_plugin = plugin.MemoryUsageGPUsPlugin(scheduler=cluster.scheduler,
                                                       path=self.path,
                                                       filetype='csv',
                                                       interval=2,
                                                       mem_max=True)

            cluster.scheduler.add_plugin(dask_plugin)

            compute(make_bag())

            client.shutdown()
