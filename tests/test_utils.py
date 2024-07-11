#!/usr/bin/env python3

import os
import unittest

from mock import patch

from dask_memusage_for_gpus import utils


class TestUtils(unittest.TestCase):
    """ Test class for utils submodule. """
    def test_generate_gpu_proccesses(self):
        fixture = os.path.join(os.path.dirname(__file__), "fixtures/nvidia_smi.1.xml")

        with patch("dask_memusage_for_gpus.utils.run_cmd") as run_cmd:
            with open(fixture, "rb") as xml:
                run_cmd.return_value = xml.readlines()

            processes = utils.generate_gpu_proccesses()

            self.assertEqual(len(processes), 3)

            self.assertEqual(processes[0].pid, 2563)
            self.assertEqual(processes[0].name, '/usr/libexec/Xorg')
            self.assertEqual(processes[0].memory_used, 136.0)
            self.assertEqual(processes[1].pid, 2735)
            self.assertEqual(processes[1].name, '/usr/bin/gnome-shell')
            self.assertEqual(processes[1].memory_used, 53.0)
            self.assertEqual(processes[2].pid, 8732)
            self.assertEqual(processes[2].name, '/usr/lib64/firefox/firefox')
            self.assertEqual(processes[2].memory_used, 230.0)

#    def test_run_cmd(self):
