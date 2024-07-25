#!/usr/bin/env python3

""" Test all the structures and funtions inside gpu_handler submodule. """

import os
import unittest

from mock import patch, Mock

from dask_memusage_gpus import plugin


class TestPlugin(unittest.TestCase):
    """ Test class for plugin submodule. """
    def setUp(self):
        self.path = os.path.join(os.path.dirname(__file__), "memusage")

    def tearDown(self):
        if os.path.exists(self.path):
            os.remove(self.path)

    @patch('dask_memusage_gpus.gpu_handler.WorkersThread')
    def test_plugin_record_to_csv(self, thread):
        thread.return_value = Mock(start=Mock(),
                                   fetch_task_used_memory=Mock(return_value=(200, 400)))

        scheduler = Mock()
        scheduler.address = '1.2.3.4'

        print(thread)
        print(thread.fetch_task_used_memory)

        dask_plugin = plugin.MemoryUsageGPUsPlugin(scheduler=scheduler,
                                                   path=self.path,
                                                   filetype='csv',
                                                   interval=2,
                                                   mem_max=True)

        dask_plugin.transition('func', 'queued', 'processing', worker='tcp://1.2.3.4:34567')
        dask_plugin.transition('func', 'processing', 'memory', worker='tcp://1.2.3.4:34567')

        dask_plugin.before_close()


