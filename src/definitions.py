#!/usr/bin/env python3

# Variable definitions
DEFAULT_DATA_FILE = "memory_usage_gpus.csv"

CSV = "csv"
PARQUET = "parquet"
JSON = "json"

FILE_TYPES = [CSV, PARQUET, JSON]

NVIDIA_SMI_QUERY_XML_CMD = "nvidia-smi -q -x"


# Exception definitions
class CMDException(Exception):
    """ Throw when CMD fails to execute. """
    pass


class FileTypeException(Exception):
    """ File Type Validation Exception. """
    pass
