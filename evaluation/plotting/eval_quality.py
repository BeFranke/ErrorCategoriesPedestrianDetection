from os import path as P

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"],
    "font.size": 14
})

values = ["LAMR", "FLAMR_crowdOcclusion", "FLAMR_envOcclusion", "FLAMR_clearForeground",
          "FLAMR_clearBackground", "FLAMR_ambiguousOcclusion", "FLAMR_localizationErrors", "FLAMR_ghostDetections",
          "FLAMR_scaleErrors"]

folder = P.abspath(P.join(
    P.dirname(__file__), "..", "..", "output"
))

timestamps = ["20220106-153536", "20220106-153744", "20220106-153931", "20220106-154118"]

df = pd.concat((pd.read_csv(P.join(folder, ts, "results.csv")) for ts in timestamps), ignore_index=True)

df = df[df["model"] != "parallel_02"]
df.to_csv("multi-iou-eval.csv")

df["model"] = df["model"].map({
    "csp_1": "CSP",
    "parallel_0": "Elimination",
    "parallel_2": "Hourglass",
    "parallel_5": "ResNeXt",
    "parallel_01": "FusedDNN-1"
})

order = {"CSP": 0, "Elimination": 1, "Hourglass": 2, "ResNeXt": 3, "FusedDNN-1": 4}
df.sort_values(by="model", key=lambda x: x.map(order))
plt.tight_layout()
sns.set_theme(style="whitegrid")

for value in values:
    df_val = df[["model", "iouThreshold", value]]
    sns.lineplot(
        data=df_val,
        x="iouThreshold",
        y=value,
        hue="model",
        legend=True,
        marker="o",
    )
    ax = plt.gca()
    ax.set_ylabel("FLAMR" if value != "LAMR" else "LAMR")
    ax.set_xlabel("Matching IoU threshold")
    ax.set_ylim([0, 0.8])
    plt.savefig(P.join(folder, timestamps[-1], "figures", f"iou-{value}.pdf"))
    plt.close()




