import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"],
    "font.size": 13,
    "axes.axisbelow": True
})

mmap = {'csp_1': 'CSP', 'parallel_2': 'Hourglass Fusion', 'parallel_0': 'Elimination',
        'parallel_5': 'ResNeXt Fusion', 'parallel_01': 'FusedDNN-1', 'parallel_02': 'FusedDNN-2'}

models = ["csp_1", "parallel_0", "parallel_2", "parallel_5", "parallel_01"]
order = lambda series: series.map({x: models.index(x) for x in series})


df = pd.read_csv("../../output/20220107-151706/results.csv")
df = df[["model", "LAMR"]]

df = df[df["model"] != "parallel_02"]

df = df.sort_values("model", key=order)
df["model"] = df["model"].map(mmap)

plt.bar(df["model"], df["LAMR"], width=0.5, color=sns.color_palette("deep", as_cmap=True), alpha=0.85)
plt.grid(axis="y")
plt.xticks(rotation=45)
plt.ylabel("LAMR")
plt.tight_layout()
plt.savefig("lamr.pdf")
