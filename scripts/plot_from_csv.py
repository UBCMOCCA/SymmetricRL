import pandas as pd
import numpy as np
from scipy.ndimage.filters import gaussian_filter1d
import argparse
import math
import csv
import re
import os

from common.plots import Plot, LinePlot, plt, ScatterPlot
from common.misc_utils import str2bool


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_paths", type=str, nargs="+")
    parser.add_argument("--columns", type=str, nargs="+")
    parser.add_argument("--row", type=str, default="total_num_steps")
    parser.add_argument("--alpha", type=float, default=0.9)
    parser.add_argument("--smoothing", type=float, default=0)
    parser.add_argument("--log_scale", type=str2bool, default=False)
    parser.add_argument("--xlog_scale", type=str2bool, default=False)
    parser.add_argument("--legend", type=str2bool, default=True)
    parser.add_argument("--name_regex", type=str, default="")
    parser.add_argument("--final", type=str2bool, default=False)
    args = parser.parse_args()

    N = len(args.columns)
    nrows = math.floor(math.sqrt(N))
    title = "Results with smoothing %.1f" % args.smoothing
    if args.log_scale:
        title += " (Log Scale)"
    plot = Plot(nrows=nrows, ncols=math.ceil(N / nrows), title=title)
    plots = []
    for column in args.columns:
        if args.final:
            plots.append(ScatterPlot(parent=plot, ylabel=column, xlabel=args.row))
        else:
            plots.append(
                LinePlot(
                    parent=plot,
                    ylabel=column,
                    xlabel=args.row,
                    ylog_scale=args.log_scale,
                    xlog_scale=args.xlog_scale,
                    alpha=args.alpha,
                    num_scatters=len(args.load_paths),
                )
            )

    if args.name_regex:
        legends = [re.findall(args.name_regex, path)[0] for path in args.load_paths]
        legend_paths = sorted(zip(legends, args.load_paths))
        legends = [x[0] for x in legend_paths]
        args.load_paths = [x[1] for x in legend_paths]
    else:
        common_prefix = os.path.commonprefix(args.load_paths)
        print("Ignoring the prefix (%s) in the legend" % common_prefix)
        legends = [path[len(common_prefix) :] for path in args.load_paths]

    if args.legend:
        plot.fig.legend(legends, loc="upper center")

    for i, path in enumerate(args.load_paths):
        print("Loading ... ", path)
        filename = "evaluate.csv" if args.final else "progress.csv"
        df = pd.read_csv(os.path.join(path, filename))
        for j, column in enumerate(args.columns):
            if args.final:
                y = df[column][-1:].item()
                x = float(legends[i])
                plots[j].add_point([x, y])
            else:
                if args.smoothing > 0.1:
                    df[column] = gaussian_filter1d(df[column], sigma=args.smoothing)
                plots[j].update(df[[args.row, column]].values, line_num=i)

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()
