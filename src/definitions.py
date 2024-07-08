#!/usr/bin/env python3

# Variable definitions
DEFAULT_DATA_FILE = "memory_usage_gpus.csv"

CSV = "csv"
PARQUET = "parquet"
JSON = "json"

FILE_TYPES = [CSV, PARQUET, JSON]

# Exception definitions
class FileTypeException(Exception):
    """ File Type Validation Exception. """
    pass
