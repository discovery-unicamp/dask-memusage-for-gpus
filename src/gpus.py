#!/usr/bin/env python3

from dataclasses import dataclass


@dataclass
class GPUProcess(object):
    pid: int
    name: str
    memory_used: int
