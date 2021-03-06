import os
from os import path as P

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


fp_map = {
    "GhostDetections": "Ghost Detections",
    "LocalizationErrors": "Localization Errors",
    "ScalingErrors": "Scaling Errors"
}


_plot_path = P.abspath(P.join(
    P.dirname(__file__), "..", "..", "output"
))
_source_folder = sorted(os.listdir(_plot_path))[-1]          # or write desired timestamp
source_path = P.join(_plot_path, _source_folder, "plotting-raw")

OUT_PATH = P.abspath(P.join(
    P.dirname(__file__), "../..", "output", _source_folder, "figures"
))


files = os.listdir(source_path)
settings = set(map(lambda s: s.split("__")[-1].split(".")[0], files))

models = pd.read_csv(P.join(_plot_path, _source_folder, "results.csv"))["model"].unique()
fns = set(filter(lambda x: x != "FPPI" and "Ghost" not in x and "Localization" not in x and "Scaling" not in x
                           and "fp_ratio" not in x and "recall" not in x and "precision" not in x
                           and "scores" not in x,
                 map(lambda s: s.split("__")[1], files)))

fps = set(filter(lambda x: ("Ghost" in x or "Localization" in x or "Scaling" in x) and "fp_ratio" not in x and
                           "recall" not in x and "precision" not in x and "scores" not in x,
                 map(lambda s: s.split("__")[1], files)))


fp_ratios = set(filter(lambda x: "fp_ratio" in x, map(lambda s: s.split("__")[1], files)))

fp_counts = set(filter(lambda x: "fp_counts" in x, map(lambda s: s.split("__")[1], files)))

for setting in settings:
    # FNs
    for y in fns:
        fig, ax = plt.subplots(1)
        for model in models:
            try:
                fppi = np.load(
                    os.path.join(source_path, f"{model}__fppi__{setting}.npy")
                )
                values = np.load(
                    os.path.join(source_path, f"{model}__{y}__{setting}.npy")
                )
                ax.plot(fppi, np.squeeze(values), label=model)

            except FileNotFoundError:
                print(f"File for {model}-{y}-{setting} does not exist!")

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_yticks([1, 0.1, 0.01])
        ax.set_xticks([0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0])
        ax.set_xlabel("FPPI")
        ax.set_ylabel("Filtered Miss Rate" if y != "mr" else "Miss Rate")
        ax.grid(visible=True, which='major', axis='x', linestyle='-', linewidth=1)
        ax.grid(visible=True, which='major', axis='y', linestyle='-', linewidth=1)
        ax.grid(visible=True, which='minor', axis='y', linestyle='--', linewidth=1)
        ax.set_title(f"{y if 'Mixed' not in y else 'OtherOcclusionErrors'}")
        plt.legend()
        plt.savefig(P.join(OUT_PATH, f"fn-mr-{y}-{setting}.pdf"))
        plt.pause(0.0001)       # without this, the plots do not show!

    for y in fps:
        fig, ax = plt.subplots(1)
        for model in models:
            try:
                mrs = np.load(
                    os.path.join(source_path, f"{model}__mr__{setting}.npy")
                )
                fppi = np.load(
                    os.path.join(source_path, f"{model}__{y}__{setting}.npy")
                ) / 500
                ax.plot(np.squeeze(fppi), np.squeeze(mrs), label=model)

            except FileNotFoundError:
                print(f"File for {model}-{y}-{setting} does not exist!")

        ax.set_xscale("log")
        ax.set_yscale("log")
        # ax.set_yticks([1, 0.1, 0.01])
        # ax.set_xticks([0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0])
        ax.set_xlabel(f"{fp_map[y.replace('fp_counts_', '')]} per Image")
        ax.set_ylabel("Miss Rate")
        ax.grid(visible=True, which='major', axis='x', linestyle='-', linewidth=1)
        ax.grid(visible=True, which='major', axis='y', linestyle='-', linewidth=1)
        ax.grid(visible=True, which='minor', axis='y', linestyle='--', linewidth=1)
        ax.set_title(f"Miss Rate over {fp_map[y.replace('fp_counts_', '')]}")
        plt.legend()
        plt.savefig(P.join(OUT_PATH, f"fp-mr-{y}-{setting}.pdf"))
        plt.pause(0.0001)  # without this, the plots do not show!

    for y in fp_ratios:
        fig, ax = plt.subplots(1)
        for model in models:
            try:
                fppi = np.load(
                    os.path.join(source_path, f"{model}__fppi__{setting}.npy")
                )
                values = np.load(
                    os.path.join(source_path, f"{model}__{y}__{setting}.npy")
                )
                ax.plot(fppi, np.squeeze(values), label=model)

            except FileNotFoundError:
                print(f"File for {model}-{y}-{setting} does not exist!")

        ax.set_xlabel("FPPI")
        ax.set_ylabel("FP Class Ratio")
        ax.set_xscale("log")
        ax.set_ylim([0, 1.1])
        # ax.set_yscale("log")
        ax.grid(visible=True, which='major', axis='x', linestyle='-', linewidth=1)
        ax.grid(visible=True, which='major', axis='y', linestyle='-', linewidth=1)
        ax.grid(visible=True, which='minor', axis='y', linestyle='--', linewidth=1)
        ax.set_title(f"Ratio of {fp_map[y.replace('fp_ratio_', '')]}")
        plt.legend()
        plt.savefig(P.join(OUT_PATH, f"{y}-{setting}.pdf"))
        plt.pause(0.0001)  # without this, the plots do not show!

    for y in fp_counts:
        fig, ax = plt.subplots(1)
        for model in models:
            try:
                conf = np.load(
                    os.path.join(source_path, f"{model}__scores__{setting}.npy")
                )
                values = np.load(
                    os.path.join(source_path, f"{model}__{y}__{setting}.npy")
                )
                ax.plot(conf, np.squeeze(values), label=model)

            except FileNotFoundError:
                print(f"File for {model}-{y}-{setting} does not exist!")

        ax.set_xlabel("Confidence")
        ax.set_ylabel("FP Class Count")
        # ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(visible=True, which='major', axis='x', linestyle='-', linewidth=1)
        ax.grid(visible=True, which='major', axis='y', linestyle='-', linewidth=1)
        ax.grid(visible=True, which='minor', axis='y', linestyle='--', linewidth=1)
        ax.set_title(f"Counts of {fp_map[y.replace('fp_counts_', '')]}")
        plt.legend()
        plt.savefig(P.join(OUT_PATH, f"{y}-{setting}.pdf"))
        plt.pause(0.0001)  # without this, the plots do not show!

    for y in [None]:
        fig, ax = plt.subplots(1)
        for model in models:
            try:
                conf = np.load(
                    os.path.join(source_path, f"{model}__scores__{setting}.npy")
                )
                values = np.load(
                    os.path.join(source_path, f"{model}__fppi__{setting}.npy")
                ) * 500
                ax.plot(conf, np.squeeze(values), label=model)

            except FileNotFoundError:
                print(f"File for {model}-{y}-{setting} does not exist!")

        ax.set_xlabel("Confidence")
        ax.set_ylabel("FP Count")
        # ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(visible=True, which='major', axis='x', linestyle='-', linewidth=1)
        ax.grid(visible=True, which='major', axis='y', linestyle='-', linewidth=1)
        ax.grid(visible=True, which='minor', axis='y', linestyle='--', linewidth=1)
        ax.set_title(f"Total FP Count")
        plt.legend()
        plt.savefig(P.join(OUT_PATH, f"fpcount-overall-{setting}.pdf"))
        plt.pause(0.0001)  # without this, the plots do not show!
