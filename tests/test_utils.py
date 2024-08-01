#!/usr/bin/env python3

""" Test all the structures and funtions inside utils submodule. """

import os
import unittest

from mock import patch

from dask_memusage_gpus import definitions as defs
from dask_memusage_gpus import gpu_handler as gpu
from dask_memusage_gpus import utils


class TestUtils(unittest.TestCase):
    """ Test class for utils submodule. """
    def test_validate_file_type(self):
        """ Test what file type is supported for this plugin. """
        with self.assertRaises(defs.FileTypeException):
            for ftype in ["CSV", "Json", "txt", "html", "doc"]:
                utils.validate_file_type(ftype)

        for ftype in ["csv", "parquet", "json", "excel", "xml"]:
            utils.validate_file_type(ftype)

    def test_generate_gpu_proccesses(self):
        """ Test general usage of function generate_gpu_proccesses(). """
        fixture = os.path.join(os.path.dirname(__file__), "fixtures/nvidia_smi.1.xml")

        with patch("dask_memusage_gpus.utils.run_cmd") as run_cmd:
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

    def test_generate_gpu_proccesses_no_pids(self):
        """ Test function generate_gpu_proccesses() when there is no process. """
        fixture = os.path.join(os.path.dirname(__file__), "fixtures/nvidia_smi.2.xml")

        with patch("dask_memusage_gpus.utils.run_cmd") as run_cmd:
            with open(fixture, "rb") as xml:
                run_cmd.return_value = xml.readlines()

            processes = utils.generate_gpu_proccesses()

            self.assertEqual(len(processes), 0)

    def test_run_cmd_with_success(self):
        """ Test a command run successfully. """
        output = ""
        for line in utils.run_cmd("echo 'This is a test'"):
            output += line.decode('ascii')

        self.assertEqual('This is a test\n', output)

    def test_run_cmd_with_error(self):
        """ Test producing an error in command line execution. """
        with self.assertRaises(defs.CMDException) as context:
            output = ""
            for line in utils.run_cmd("echo 'This is a test' | exit 1"):
                output += line.decode('ascii')

        self.assertEqual('Error: ', str(context.exception))

    def test_run_cmd_with_delay(self):
        """ Test producing a delay in command line execution. """
        output = ""
        for line in utils.run_cmd("sleep 3 | echo 'This is a test'"):
            output += line.decode('ascii')

        self.assertEqual('This is a test\n', output)

    def test_run_cmd_with_delay_and_empty(self):
        """ Test producing a delay in command line execution but no output. """
        output = ""
        for line in utils.run_cmd("sleep 3"):
            output += line.decode('ascii')

        self.assertEqual('', output)

    def test_get_worker_gpu_memory_used(self):
        """ Test general use of function get_worker_gpu_memory_used(). """
        processes = [gpu.GPUProcess(pid=1234,
                                    name="foo",
                                    memory_used=10),
                     gpu.GPUProcess(pid=5678,
                                    name="bar",
                                    memory_used=25),
                     gpu.GPUProcess(pid=2222,
                                    name="/usr/bin/python3",
                                    memory_used=310)]

        with patch("dask_memusage_gpus.utils.generate_gpu_proccesses") as p_gen:
            with patch("os.getpid") as getpid:
                p_gen.return_value = processes
                getpid.return_value = 2222

                self.assertEqual(utils.get_worker_gpu_memory_used(), 310)

    def test_get_worker_gpu_memory_used_no_match(self):
        """ Test general use of function get_worker_gpu_memory_used(). """
        processes = [gpu.GPUProcess(pid=1234,
                                    name="foo",
                                    memory_used=10),
                     gpu.GPUProcess(pid=5678,
                                    name="bar",
                                    memory_used=25)]

        with patch("dask_memusage_gpus.utils.generate_gpu_proccesses") as p_gen:
            with patch("os.getpid") as getpid:
                p_gen.return_value = processes
                getpid.return_value = 2222

                self.assertEqual(utils.get_worker_gpu_memory_used(), 0)
