#!/usr/bin/python3

""" Plot recorded files using Plotly. """

import argparse
import sys

try:
    import pandas as pd
except ImportError:
    print("Do you have `pandas` installed?")
    sys.exit(-1)

try:
    import plotly.express as px
    import plotly.graph_objects as go
except ImportError:
    print("Do you have `plotly` installed?")
    sys.exit(-1)


def check_missing_hits(dataframe):
    """ Function to check if there is no missing function to record. """
    t_array = dataframe[dataframe['max_gpu_memory_mb'] == -1]['time'].values

    if len(t_array) == 0:
        return

    min_time = max(t_array)
    for i in range(1, len(t_array)):
        if (min_time > t_array[i] - t_array[i - 1]):
            min_time = t_array[i] - t_array[i - 1]

    min_time = round(min_time / 2, 5)

    print(f"WARNING: There are some missing hits. We suggest to use the interval of {min_time} (s).")


def plot(dataframe, output_path, title=''):
    """ Plot dataframe in `output_path` with or without a `title`. """
    fig = go.Figure()

    dataframe = dataframe[dataframe["max_gpu_memory_mb"] >= 0]

    # Add traces
    i = 0
    for worker_id in pd.unique(dataframe['worker_id']):
        fig.add_trace(go.Scatter(x=dataframe[dataframe['worker_id'] == worker_id]['time'],
                                 y=dataframe[dataframe['worker_id'] == worker_id]['max_gpu_memory_mb'],
                                 mode='lines+markers',
                                 name='GPU ' + str(i),
                                 marker=dict(color=px.colors.qualitative.G10[i]))
        )

        i += 1

    fig.update_layout(
        height=500,
        width=900,
        template='plotly_dark',
        legend_title='GPU IDs',
        title=title,
        xaxis_title="Time (s)",
        yaxis_title="GPU Memory Usage (MiB)",
    )

    fig.write_image(output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('file', metavar='FILE', type=str, 
                        help='The file to be parsed.')
    parser.add_argument('--output', '-o', type=str, required=True,
                        help='Path and name of the output file.')
    parser.add_argument('--title', type=str, default='',
                        help='Title of the chart.')
    parser.add_argument('--type', type=str, default='csv',
                        help='Type of the file to be parsed.')

    args = parser.parse_args()

    if args.type.upper() == 'CSV':
        dataframe = pd.read_csv(args.file)
    elif args.type.upper() == 'EXCEL':
        dataframe = pd.read_excel(args.file)
    elif args.type.upper() == 'JSON':
        dataframe = pd.read_json(args.file)
    elif args.type.upper() == 'PARQUET':
        dataframe = pd.read_parquet(args.file)
    elif args.type.upper() == 'XML':
        dataframe = pd.read_xml(args.file)
    else:
        print('ERROR: file type {args.type} is not supported.')
        sys.exit(-2)

    check_missing_hits(dataframe)

    plot(dataframe, args.output, title=args.title)
