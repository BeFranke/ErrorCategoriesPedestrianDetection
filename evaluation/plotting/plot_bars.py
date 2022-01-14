from os import path as P
import os

import pandas as pd
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"],
    "font.size": 13
})

models = ["csp_1", "parallel_0", "parallel_2", "parallel_5", "parallel_01", "parallel_02"]
order = lambda series: series.map({x: models.index(x) for x in series})

sstr = "Reasonable"

def make_frequency_bar_plot_fn(df, sstr):
    df = pd.melt(df, id_vars=["model"])
    df = df[np.isin(df.variable, ['FLAMR-crowdOcclusion', "FLAMR-envOcclusion", "FLAMR-clearForeground",
                                  "FLAMR-clearBackground", "FLAMR-ambiguousOcclusion"])]
    varmap = {
        'FLAMR-crowdOcclusion': "Crowd Occlusion",
        "FLAMR-envOcclusion": "Environmental Occlusion",
        "FLAMR-clearForeground": "Clear Foreground",
        "FLAMR-clearBackground": "Clear Background",
        "FLAMR-ambiguousOcclusion": "Ambiguous Occlusion"
    }
    df.variable = df.variable.map(lambda x: varmap[x] if x in varmap else x)

    df.columns = ["Model", "Error Type", "Filtered Log-Average Miss Rate"]

    order = df.groupby("Error Type")["Filtered Log-Average Miss Rate"].mean().sort_values(ascending=False).index.values

    sns.set_style("whitegrid")
    g = sns.catplot(
        data=df,
        kind="bar",
        x="Error Type",
        y="Filtered Log-Average Miss Rate",
        hue="Model",
        palette="dark",
        alpha=0.6,
        height=6,
        legend=False,
        order=order
    )
    g.despine(left=True)

    plt.legend(loc='best')
    plt.xticks(rotation=45)
    # plt.title(f"FLAMR on {sstr} Evaluation (False Negatives)")
    plt.tight_layout()
    plt.savefig(P.join(OUT_PATH, f"fn-{sstr}.pdf"))


def make_frequency_bar_plot_fp(df, sstr):
    df = pd.melt(df, id_vars=["model"])
    df = df[np.isin(df.variable, ["FLAMR-ghostDetections", "FLAMR-localizationErrors", "FLAMR-scaleErrors"])]
    varmap = {
        "FLAMR-localizationErrors": "Localization Errors",
        "FLAMR-ghostDetections": "Ghost Detections",
        "FLAMR-scaleErrors": "Scaling Errors"
    }
    df.variable = df.variable.map(lambda x: varmap[x])

    df.columns = ["Model", "Error Type", "Filtered Log-Average Miss Rate"]

    sns.set_style("whitegrid")
    g = sns.catplot(
        data=df,
        kind="bar",
        x="Error Type",
        y="Filtered Log-Average Miss Rate",
        hue="Model",
        palette="dark",
        alpha=0.6,
        height=6,
        legend=False
    )
    g.despine(left=True)

    plt.legend(loc='best')
    plt.xticks(rotation=45)
    # plt.title(f"LAMR over FP class on {sstr} Evaluation (False Positives)")
    plt.tight_layout()
    plt.savefig(P.join(OUT_PATH, f"fp-{sstr}.pdf"))


if __name__ == "__main__":
    _path = P.abspath(P.join(
        P.dirname(__file__), "..", "..", "output"
    ))
    _source_folder = sorted(os.listdir(_path))[-1]  # or write desired timestamp
    source_path = P.join(_path, _source_folder, "results.csv")
    OUT_PATH = P.abspath(P.join(
        P.dirname(__file__), "..", "..", "output", _source_folder, "figures"
    ))


    df = pd.read_csv(source_path, index_col=None)
    mmap = {'csp_1': 'CSP', 'parallel_2': 'Hourglass Fusion', 'parallel_0': 'Elimination',
            'parallel_5': 'ResNeXt Fusion', 'parallel_01': 'FusedDNN-1', 'parallel_02': 'FusedDNN-2'}

    df = df[df["model"] != "parallel_02"]

    df.columns = [s.replace("_", "-") for s in df.columns]

    df.sort_values("model", key=order, inplace=True)
    df.model = df.model.map(lambda x: mmap[x])
    df[set(df.columns) - {"model"}].astype(float)
    make_frequency_bar_plot_fn(df, sstr)
    make_frequency_bar_plot_fp(df, sstr)

