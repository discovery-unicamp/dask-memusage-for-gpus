#!/usr/bin/env python3

import xml.etree.ElementTree as ET
import subprocess

from time import sleep

from . import definitions as defs


def run_cmd(cmd, shell=True):
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
       raise defs.CMDException("Error: " + str(err))


class generate_gpu_proccesses():
    output = ""
    for line in run_cmd(defs.NVIDIA_SMI_QUERY_XML_CMD):
        output += line

    root = ET.fromstring(output)

    for child in root:
        if child.tag == "gpu":
            for gpu_child in child:
                if gpu_child.tag == "processes":
                    for process in gpu_child:
                        if process.tag == "process_info":
                            for process_info in process:
                                if process_info.tag == "pid":
                                    pid = int(process_info.text)
                                elif process_info.tag == "process_name":
                                    name = process_info.text
                                elif process_info.tag == "used_memory":
                                    memory = float(process_info.text.split(' ')[0])
                            processes.append(GPUProcess(pid=pid, name=name, memory=memory)
