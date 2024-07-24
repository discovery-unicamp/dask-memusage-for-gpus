#!/usr/bin/env python3

""" All kinds of functions that are common to other modules. """

import os
import subprocess
import xml.etree.ElementTree as ET
from time import sleep

from dask_memusage_gpus import definitions as defs
from dask_memusage_gpus import gpu_handler as gpu


def validate_file_type(filetype):
    """
    Validate the type of the input file.

    Parameters
    ----------
    filetype : string
        Type of the input file to be recorded.

    Raises
    ------
    FileTypeException
        If the type does not match with the supported types.
    """
    if filetype not in defs.FILE_TYPES:
        raise defs.FileTypeException(f"'{filetype}' is not a valid "
                                     "output file.")


def run_cmd(cmd, shell=True):
    """
    Run a command line using python Popen.

    Parameters
    ----------
    cmd : str or list of str
        Command line to be executed by this function. It needs to be a list
        of strings if the parameter `shell` is False.
    shell : boolean, optional
        If the command line passed to this function is a command line or a
        shell script (default=True).

    Returns
    -------
    string
        Stdout in a string

    Raises
    ------
    CMDException
        If the process returns an error.
    """
    p = subprocess.Popen(cmd,
                         stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE,
                         shell=shell)

    for line in iter(p.stdout.readline, b''):
        if line:
            yield line

    while p.poll() is None:
        sleep(.1)

    err = p.stderr.read()
    if p.returncode != 0:
        raise defs.CMDException("Error: " + err.decode('utf-8'))


def generate_gpu_proccesses():
    """
    Parse the XML output returned by `nvidia_smi` command.

    Returns
    -------
    list
        A list of objects GPUProcess.
    """
    output = ""
    for line in run_cmd(defs.NVIDIA_SMI_QUERY_XML_CMD):
        output += line.decode("utf-8")

    root = ET.fromstring(output)

    processes = []
    for child in root:
        if child.tag == "gpu":
            for gpu_child in child:
                if gpu_child.tag == "processes":
                    for process in gpu_child:
                        if process.tag == "process_info":

                            pid = -1
                            name = None
                            memory = 0

                            for process_info in process:
                                if process_info.tag == "pid":
                                    pid = int(process_info.text)
                                elif process_info.tag == "process_name":
                                    name = process_info.text
                                elif process_info.tag == "used_memory":
                                    memory = process_info.text.split(' ')
                                    memory = float(memory[0])

                            processes.append(gpu.GPUProcess(pid=pid,
                                                            name=name,
                                                            memory_used=memory))
    return processes


def get_worker_gpu_memory_used():
    """
    Returns the GPU used memory per worker.

    Returns
    -------
    integer
        The used memory in MiB.
    """
    processes = generate_gpu_proccesses()

    for process in processes:
        if process.pid == os.getpid() and "python" in process.name:
            return int(process.memory_used)

    return 0
