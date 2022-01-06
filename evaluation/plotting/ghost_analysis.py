import json
import os
import sys

from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from PIL import Image
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import pandas as pd

from cityscapes import Cityscapes

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"],
    "font.size": 14
})

IGNORE_MODELS = ["parallel_02"]

segm_root = os.path.abspath(os.path.join(
    os.path.dirname(__file__),
    "..", "..", "input", "datasets", "cityscapes", "gtFine"
))

ofolder = os.path.abspath(os.path.join(
    os.path.dirname(__file__),
    "..", "..", "output"
))

tfolder = sorted(filter(lambda x: os.path.isdir(os.path.join(ofolder, x)), os.listdir(ofolder)))[0]

order = ["csp_1___0.5___Reasonable", "parallel_0___0.5___Reasonable", "parallel_2___0.5___Reasonable",
         "parallel_5___0.5___Reasonable", "parallel_01___0.5___Reasonable"]

BB_FILES = sorted(
    map(
        lambda x: os.path.join(ofolder, tfolder, "raw", x),
        filter(
            lambda x: all((ig not in x and "index.txt" not in x) for ig in IGNORE_MODELS),
            os.listdir(os.path.join(ofolder, tfolder, "raw"))
        )
    ), key=lambda x: order.index(x.split("/")[-1])
)

NAMES = ["CSP", "Elimination", "Hourglass", "ResNeXt", "FusedDNN-1"]

CONF_CUTOFF = 0.1

mapping = {
        c.id: c.train_id if c.train_id != 255 else 19 for c in Cityscapes.classes
    }


convert_to_train_ids = np.vectorize(mapping.__getitem__)



def process_img(dts: list, segm_path: str, img_w: int, img_h: int):
    segm = np.asarray(Image.open(segm_path))
    segm = convert_to_train_ids(segm)
    res = []
    for dt in filter(lambda x: x["error_type"] == "ghost" and x["score"] >= CONF_CUTOFF, dts):
        assert len(dt["bbox"]) == 4
        x1, y1, w, h = dt["bbox"]
        x1, x2 = (lambda x: np.clip(x, a_min=0, a_max=img_w).round().astype(int))([x1, x1 + w])
        y1, y2 = (lambda y: np.clip(y, a_min=0, a_max=img_h).round().astype(int))([y1, y1 + h])
        bb = segm[y1:y2+1, x1:x2+1]
        counts = np.zeros(20)       # 19 is background
        u, c = np.unique(bb, return_counts=True)
        counts[u] = c
        res.append(np.argmax(counts))

    return res


get_segm_path = lambda s: os.path.join(
    segm_root,
    "val",
    s.split("_")[0],
    s.replace("_leftImg8bit", "_gtFine_labelIds")
)


# exception-raising single item list unpacking, source:
# https://stackoverflow.com/questions/33161448/getting-only-element-from-a-single-element-list-in-python
def get_class_name(c):
    try:
        return (lambda x: x)(
            *list(filter(lambda cls: cls.train_id == c, Cityscapes.classes))
         ).name
    except TypeError:
        return "other"


def main():
    plt.tight_layout()
    x = list(map(get_class_name, list(range(19)) + [255]))
    results = {'class name': [], 'value': [], 'name': []}
    for i, bb_file in enumerate(BB_FILES):
        with open(bb_file, "r") as fp:
            f = json.load(fp)

        dts = f["dts"]
        imgs = f["imgs"]
        result = []
        for key in tqdm(imgs):
            img = imgs[key]
            segm_path = get_segm_path(img["im_name"])
            img_id, im_h, im_w = img["id"], img["height"], img["width"]
            try:
                result += process_img(dts[str(img_id)], segm_path, im_w, im_h)
            except KeyError:
                # no detections for this image
                pass

        result = np.asarray(result)
        counts_total = np.zeros(20)
        counts_total[tup[0]] = (tup := np.unique(result, return_counts=True))[1]
        inds = np.argsort(counts_total)[::-1]

        # plt.bar(x=x, height=counts_total / np.sum(counts_total), label=NAMES[i])
        results['class name'] += x
        results['value'] += (counts_total / np.sum(counts_total)).tolist()
        results['name'] += [NAMES[i] for _ in x]

        print("Percentage of dominated ghost detections per class:")
        for c in inds:
            print(f"{get_class_name(c if c != 19 else 255)}: {counts_total[c] / np.sum(counts_total)}")

    df = pd.DataFrame(results)
    sns.set_theme(style="whitegrid")
    g = sns.catplot(
        data=df,
        kind="bar",
        x="class name",
        y="value",
        hue="name",
        palette="dark",
        alpha=0.6,
        height=6,
        legend=False
    )
    g.despine(left=True)

    ax = plt.gca()
    ax.set_ylabel("Percentage of Ghost Detections")
    ax.set_xlabel(None)
    plt.xticks(rotation=90)
    plt.subplots_adjust(bottom=0.3)
    plt.tight_layout()
    plt.legend(loc='upper left')
    plt.savefig(os.path.join(ofolder, tfolder, "figures", f"ghost-analysis.pdf"))


if __name__ == "__main__":
    main()
