from os import path as P
import os

import pandas as pd
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"]
})

models = ["csp_1", "parallel_0", "parallel_2", "parallel_5", "parallel_01", "parallel_02"]
order = lambda series: series.map({x: models.index(x) for x in series})

OUT_PATH = P.abspath(P.join(
        P.dirname(__file__), "..", "output", "figures"
    ))

def make_frequency_bar_plot_fn(df, sstr):
    df_bar = df[df['iouMatchThrs'] == 0.5]
    df_bar.drop(["iouMatchThrs"] + list(filter(lambda x: "HC-" in x, df_bar.columns)), axis=1, inplace=True)
    df_bar = pd.melt(df_bar, id_vars=["model"])
    df_bar = df_bar[df_bar.variable != "MR"]
    df_bar = df_bar[df_bar.variable != "minMR"]
    df_bar = df_bar[df_bar.variable != "minFPPI"]
    df_bar = df_bar[df_bar.variable != "multiDetectionErrors"]
    df_bar = df_bar[df_bar.variable != "ghostDetectionErrors"]
    df_bar = df_bar[df_bar.variable != "minFPPI"]
    df_bar = df_bar[df_bar.variable != "scalingErrors"]
    df_bar = df_bar[df_bar.variable != "minMR-loc"]
    df_bar = df_bar[df_bar.variable != "minMR-scaling"]
    df_bar = df_bar[df_bar.variable != "minMR-ghost"]
    varmap = {
        'crowdOcclusionErrors': "Crowd Occlusion",
        "envOcclusionErrors": "Environmental Occlusion",
        "foregroundErrors": "Foreground",
        "otherErrors": "Standard Errors",
        "mixedOcclusionErrors": "Ambiguous Occlusion"
    }
    df_bar.variable = df_bar.variable.map(lambda x: varmap[x] if x in varmap else x)

    df_bar.columns = ["Model", "Error Type", "Filtered Log-Average Miss Rate"]

    sns.set_style("darkgrid")
    g = sns.catplot(
        data=df_bar,
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

    plt.legend(loc='upper left')
    plt.xticks(rotation=45)
    plt.title(f"FLAMR on {sstr} Evaluation (False Negatives)")
    plt.tight_layout()
    plt.savefig(P.join(OUT_PATH, f"fn-{sstr}.pdf"))


def make_frequency_bar_plot_fp(df, sstr):
    df_bar = df[df['iouMatchThrs'] == 0.5]
    df_bar.drop(["iouMatchThrs"] + list(filter(lambda x: "HC-" in x, df_bar.columns)), axis=1, inplace=True)
    df_bar = pd.melt(df_bar, id_vars=["model"])
    df_bar = df_bar[np.isin(df_bar.variable, ["ghostDetectionErrors", "multiDetectionErrors", "scalingErrors"])]
    varmap = {
        "multiDetectionErrors": "Poor Localization",
        "ghostDetectionErrors": "Ghost Detections",
        "scalingErrors": "Scaling Errors"
    }
    df_bar.variable = df_bar.variable.map(lambda x: varmap[x] if x in varmap else x)

    df_bar.columns = ["Model", "Error Type", "Filtered Log-Average Miss Rate"]

    sns.set_style("darkgrid")
    g = sns.catplot(
        data=df_bar,
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

    plt.legend(loc='upper right')
    plt.xticks(rotation=45)
    plt.title(f"LAMR over FP class on {sstr} Evaluation (False Positives)")
    plt.tight_layout()
    plt.savefig(P.join(OUT_PATH, f"fp-{sstr}.pdf"))

if __name__ == "__main__":
    _path = P.abspath(P.join(
        P.dirname(__file__), "..", "output"
    ))
    _source_folder = sorted(os.listdir(_path))[-1]  # or write desired timestamp
    source_path = P.join(_path, _source_folder, "results.csv")
    df = pd.read_csv(source_path, index_col=None)
    mmap = {'csp_1': 'CSP', 'parallel_2': 'Hourglass Fusion', 'parallel_0': 'Elimination',
            'parallel_5': 'ResNeXt Fusion', 'parallel_01': 'FusedDNN-1', 'parallel_02': 'FusedDNN-2'}

    df.sort_values("model", key=order, inplace=True)
    df.model = df.model.map(lambda x: mmap[x])
    df["setting_id"] = df["setting_id"].map(lambda x: x.split("[")[1].split("]")[0])
    df[set(df.columns) - {"model"}].astype(float)
    for setting in df["setting_id"].unique():
        df_s = df[df["setting_id"] == setting]
        df_s.drop("setting_id", inplace=True, axis=1)
        sstr = "Reasonable" if setting == '0' else "All"
        make_frequency_bar_plot_fn(df_s, sstr)
        make_frequency_bar_plot_fp(df_s, sstr)
        # make_hc_bar_plots(df_s, sstr)
        # make_scale_plots(df_s, sstr)
