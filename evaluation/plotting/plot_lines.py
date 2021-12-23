import os
from os import path as P

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"]
})


MODEL_MAP = {
    'csp_1': 'CSP',
    'parallel_0': 'Elimination',
    'parallel_2': 'Hourglass',
    'parallel_5': 'ResNeXt',
    'parallel_01': 'FusedDNN-1',
    # 'parallel_02': 'FusedDNN-2'
}
fp_map = {
    "GhostDetections": "Ghost Detections",
    "PoorLocalization": "Localization Errors",
    "ScalingErrors": "Scaling Errors"
}


model_str = lambda s: MODEL_MAP[s] if s in MODEL_MAP else s
setting_str = lambda x: "All" if "4" in x else "Reasonable"


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

models = MODEL_MAP.keys() # ["csp_1", "parallel_2", "parallel_0", "parallel_5"]
fns = set(filter(lambda x: x != "fppi" and "Ghost" not in x and "Localization" not in x and "Scaling" not in x
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
                ax.plot(fppi, np.squeeze(values), label=model_str(model))

            except FileNotFoundError:
                print(f"File for {model}-{y}-{setting} does not exist!")

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_yticks([1, 0.1, 0.01])
        ax.set_xticks([0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0])
        ax.set_xlabel("FPPI")
        ax.set_ylabel("Filtered Miss Rate" if y != "mr" else "Miss Rate")
        ax.grid(b=True, which='major', axis='x', linestyle='-', linewidth=1)
        ax.grid(b=True, which='major', axis='y', linestyle='-', linewidth=1)
        ax.grid(b=True, which='minor', axis='y', linestyle='--', linewidth=1)
        # ax.set_title(f"{y if 'Mixed' not in y else 'OtherOcclusionErrors'}, {setting_str(setting)}")
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
                ax.plot(np.squeeze(fppi), np.squeeze(mrs), label=model_str(model))

            except FileNotFoundError:
                print(f"File for {model}-{y}-{setting} does not exist!")

        ax.set_xscale("log")
        ax.set_yscale("log")
        # ax.set_yticks([1, 0.1, 0.01])
        # ax.set_xticks([0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0])
        ax.set_xlabel(f"{y.replace('fp_counts_', '')} per Image")
        ax.set_ylabel("Miss Rate")
        ax.grid(b=True, which='major', axis='x', linestyle='-', linewidth=1)
        ax.grid(b=True, which='major', axis='y', linestyle='-', linewidth=1)
        ax.grid(b=True, which='minor', axis='y', linestyle='--', linewidth=1)
        # ax.set_title(f"Miss Rate over {fp_map[y.replace('fp_counts_', '')]}, {setting_str(setting)}")
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
                ax.plot(fppi, np.squeeze(values), label=model_str(model))

            except FileNotFoundError:
                print(f"File for {model}-{y}-{setting} does not exist!")

        ax.set_xlabel("fppi")
        ax.set_ylabel("FP class ratio")
        ax.set_xscale("log")
        ax.set_ylim([0, 1.1])
        # ax.set_yscale("log")
        ax.grid(b=True, which='major', axis='x', linestyle='-', linewidth=1)
        ax.grid(b=True, which='major', axis='y', linestyle='-', linewidth=1)
        ax.grid(b=True, which='minor', axis='y', linestyle='--', linewidth=1)
        # ax.set_title(f"Ratio of {fp_map[y.replace('fp_ratio_', '')]}, {setting_str(setting)}")
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
                ax.plot(conf, np.squeeze(values), label=model_str(model))

            except FileNotFoundError:
                print(f"File for {model}-{y}-{setting} does not exist!")

        ax.set_xlabel("Confidence")
        ax.set_ylabel("FP class count")
        # ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(b=True, which='major', axis='x', linestyle='-', linewidth=1)
        ax.grid(b=True, which='major', axis='y', linestyle='-', linewidth=1)
        ax.grid(b=True, which='minor', axis='y', linestyle='--', linewidth=1)
        # ax.set_title(f"Counts of {fp_map[y.replace('fp_counts_', '')]}, {setting_str(setting)}")
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
                ax.plot(conf, np.squeeze(values), label=model_str(model))

            except FileNotFoundError:
                print(f"File for {model}-{y}-{setting} does not exist!")

        ax.set_xlabel("Confidence")
        ax.set_ylabel("FP Count")
        # ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(b=True, which='major', axis='x', linestyle='-', linewidth=1)
        ax.grid(b=True, which='major', axis='y', linestyle='-', linewidth=1)
        ax.grid(b=True, which='minor', axis='y', linestyle='--', linewidth=1)
        # ax.set_title(f"Total FP Count, {setting_str(setting)}")
        plt.legend()
        plt.savefig(P.join(OUT_PATH, f"fpcount-overall-{setting}.pdf"))
        plt.pause(0.0001)  # without this, the plots do not show!
