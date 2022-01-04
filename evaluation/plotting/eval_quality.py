from os import path as P

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"]
})

values = ["LAMR", "FLAMR_crowdOcclusionErrors", "FLAMR_envOcclusionErrors", "FLAMR_foregroundErrors",
          "FLAMR_otherErrors", "FLAMR_mixedOcclusionErrors", "FLAMR_multiDetectionErrors", "FLAMR_ghostDetectionErrors",
          "FLAMR_scaleErrors"]

folder = P.abspath(P.join(
    P.dirname(__file__), "..", "..", "output"
))

timestamps = ["20211231-134853", "20211231-135111", "20211231-135306", "20211231-135642"]

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
sns.set_theme()

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




