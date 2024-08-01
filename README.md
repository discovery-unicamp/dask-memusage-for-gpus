# Dask Memory Usage Plugin for GPUs

[![Continuous Tests](https://github.com/discovery-unicamp/dask-memusage-gpus/actions/workflows/ci.yaml/badge.svg)](https://github.com/discovery-unicamp/dask-memusage-gpus/actions/workflows/ci.yaml)![Interrogate](https://github.com/discovery-unicamp/dask-memusage-gpus/blob/badges/badges/interrogate_badge.svg)![Coverage](https://github.com/discovery-unicamp/dask-memusage-gpus/blob/badges/badges/coverage.svg)

If you familiar with [dask-memusage](https://github.com/itamarst/dask-memusage) plugin, this is an alternative version to profile GPU(s) memory usage by using `nvidia-smi` command and its XML output.

This plugin is a low impact memory tracker, but if you need something more advanced, check [Scalene and other profilers](https://raw.githubusercontent.com/plasma-umass/scalene/master/docs/images/profiler-comparison.png). For pros and cons of this plugin, see FAQ file.

## Code Example

Import some blobs and machine learning models. Also import the Dask Client to connect to the scheduler.

```python
import argparse
import cupy as cp

from cuml.dask.datasets import make_blobs as make_blobs_MGPU
from cuml.dask.cluster import KMeans as KMeans_MGPU

from dask.distributed import Client
```

Now, we can run the main client. Remember that Dask is preferred to be executed inside the main section. This example
uses [RAPIDS AI](https://rapids.ai/), check if the libraries are proper installed. We strongly recommend to use a pre
built container.

```python
def main():
    parser = argparse.ArgumentParser(description="Test KMeans experiment.")

    parser.add_argument('--scheduler-file', type=str, required=True,
                        help='Location of the scheduler file.')

    parser.add_argument('--nodes', type=int, required=True,
                        help='Number of worker nodes.')

    args = parser.parse_args()

    client = Client(scheduler_file=args.scheduler_file)

    client.wait_for_workers(n_workers=args.nodes)

    n_samples = 500000000
    n_bins = 3

    centers = cp.asarray([(-6, -6), (0, 0), (9, 1)])
    X, y = make_blobs_MGPU(n_samples=n_samples, centers=centers, shuffle=False, random_state=42, client=client)

    model = KMeans_MGPU(n_clusters=3, random_state=42, max_iter=100)

    model.fit(X=X)

    pred = model.predict(X=X)

    pred.compute()

    client.close()


if __name__ == '__main__':
    main()
```

This code can be executed by passing the proper parameters to the command line.


## Scheduler CLI usage

To run the scheduler to monitor the GPU memory usage, the scheduler just requires the preloaded plugin module as the
example below.

```bash
$ dask scheduler --preload dask_memusage_gpus_plugin --memusage-gpus-path memusage-gpus.csv --memusage-gpus-record-type csv --memusage-gpus-max
```

This plugin also supports other formats like Parquet and Excel for example. There is no problem with workers and
threads because Dask CUDA worker only executes 1 thread per GPU.

The results of this execution within the plugin enabled inside the cluster can be seen below.

![kmeans](docs/imgs/max_memory_used_per_gpu.png)

## Limitations and Useful Content

For further information hints about this plugin visit the [FAQ](FAQ.md) document.


## Authors

- Julio Faracco
