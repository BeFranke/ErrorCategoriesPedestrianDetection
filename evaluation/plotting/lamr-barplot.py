from os import path as P
import os

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


output_folder = P.abspath(P.join(P.dirname(__file__), P.pardir, P.pardir, "output"))

eval = sorted(os.listdir(output_folder))[-1]

path = P.join(output_folder, eval)

df = pd.read_csv(P.join(path, "results.csv"))
df = df[["model", "LAMR"]]

plt.bar(df["model"], df["LAMR"], width=0.5, color=sns.color_palette("deep", as_cmap=True), alpha=0.85)
plt.grid(axis="y")
plt.xticks(rotation=45)
plt.ylabel("LAMR")
plt.ylim([0.1, 0.15])
plt.tight_layout()
plt.savefig(P.join(path, "figures", "lamr.pdf"))
