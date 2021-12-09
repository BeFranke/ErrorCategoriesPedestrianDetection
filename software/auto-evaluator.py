import itertools as it
import os
import sys
from io import StringIO

import numpy as np
import torch
import yaml

COLUMNS = ['setting_id', 'model', 'iouMatchThrs', 'MR', 'minMR', 'minFPPI', "crowdOcclusionErrors",
           "envOcclusionErrors", "foregroundErrors", "multiDetectionErrors", "ghostDetectionErrors", "otherErrors",
           "mixedOcclusionErrors", "scalingErrors", "HC-co", "HC-eo", "HC-fg", "HC-md", "HC-gd", "HC-ot",
           "HC-mo", "HC-sc", "minMR-scaling", "minMR-loc", "minMR-ghost"]


SEARCH_KEYS = ["EVALUATION - MR", "EVALUATION - MINMR", "EVALUATION - MINFPPI", "crowdOcclusionErrors",
               "envOcclusionErrors", "foregroundErrors", "multiDetectionErrors", "ghostDetectionErrors", "otherErrors",
               "mixedOcclusionErrors", "scaleErrors", "HC crowdOcclusionErrors",
               "HC envOcclusionErrors", "HC foregroundErrors", "HC multiDetectionErrors", "HC ghostDetectionErrors",
               "HC otherErrors", "HC mixedOcclusionErrors", "HC scaleErrors", "scaleErrors@minMR",
               "multiDetectionErrors@minMR", "ghostDetectionErrors@minMR"]




sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pandas as pd
from API import evaluation

try:
    results = pd.read_csv("results_metric_semseg_big.csv", index_col=None)
except FileNotFoundError:
    results = pd.DataFrame(columns=COLUMNS)


class NetConfig:
    """
    model_id: '20210310_210051'
    iteration: '32000'
    model: 'csp_1'
    backbone: 'resnet_50'
    """

    def __init__(self, model_id, iteration, model, backbone):
        self.model_id = model_id
        self.iteration = iteration
        self.model = model
        self.backbone = backbone

    def append_to_cfg(self, dct):
        dct['model_id'] = self.model_id
        dct['iteration'] = self.iteration
        dct['model'] = self.model
        dct['backbone'] = self.backbone

    def __str__(self):
        return self.model


params = {
    "model": [
        NetConfig('20210310_210051', 32000, 'csp_1', 'resnet_50'),
        NetConfig('20210716_113443', 12000, 'parallel_2', 'None'),
        NetConfig('20210310_210051', 32000, 'parallel_0', 'resnet_50'),
        NetConfig('20210310_210051', 32000, 'parallel_5', 'resnet_50'),
        NetConfig('20210310_210051', 32000, 'parallel_01', 'resnet_50'),
        NetConfig('20210310_210051', 32000, 'parallel_02', 'resnet_50')
    ],
    # "thresholds.iouMatchThrs": [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    "thresholds.iouMatchThrs": [0.5],
    "setting_id": [[0], [4]]
}

result_file = "results_metric_semseg_big.csv"

cfg = {
    'model_id': None,
    'model': None,
    'iteration': None,
    'backbone': None,
    'dataset': 'cityscapes',
    'image_set': {'val': 'val'},
    'img_res': {'train': [640, 1280], 'val': [1024, 2048]},
    'evaltype': ['mr'],
    'save': False,
    'plot': False,
    'thresholds': {
      'envPixelThrs': 0.3,         # previously alpha
      'occPixelThrs': 0.4,         # new
      'crowdPixelThrs': 0.05,     # previously gamma
      "iouMatchThrs": 0.5,         # previously beta
      "foregroundThrs": 200,        # previously delta
      "highConfidenceThrs": 0.5   # previously mu
    },
    'maa_thres': 0.3,
    'batch_size_eval': 1,
    'generate_dataset': False,
    'cache': True,
    'setting_id': [0]
}

myconfig = os.path.join(os.path.dirname(__file__), "..", "..", "patrick", "cfg_auto.yaml")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "patrick"))

keys = params.keys()
for i, tup in enumerate(it.product(*params.values())):
    already_done = True
    for k, v in zip(keys, tup):
        k = k.split(".")[-1]
        already_done = np.asarray((np.asarray(results[k], dtype=str) == str(v)) & already_done)

    print(" | ".join(f"{k}: {v}" for k, v in zip(keys, tup)))
    if already_done.any():
        print("already done!")
        assert already_done.sum() == 1

    else:
        tup[0].append_to_cfg(cfg)
        params = dict()
        for k, v in zip(keys, tup):
            if "." in k:
                grp, key = k.split(".")
                cfg[grp][key] = str(v)
                params[key] = str(v)
            else:
                cfg[k] = str(v)
                params[k] = str(v)
        with open(myconfig, 'w+') as fp:
            yaml.dump(cfg, fp)

        backup = sys.stdout
        sys.stdout = StringIO()
        wd = os.getcwd()

        os.chdir(os.path.join("..", "..", "patrick"))
        evaluation.main(
            "cfg_auto.yaml",
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )

        os.chdir(wd)
        out = sys.stdout.getvalue()
        sys.stdout = backup
        assert len(COLUMNS[3:]) == len(SEARCH_KEYS)
        try:
            res = {
                key: float(
                    list(filter(
                        lambda x: search_key in x, out.split("\n")
                    ))[0].split(":")[1].strip()
                ) for key, search_key in zip(COLUMNS[3:], SEARCH_KEYS)}
            res.update(params)
        except IndexError:
            print("Unexpected format in stdout!")
            print("Output was:")
            print("#################")
            print(out)
            print("#################")
            exit(1)

        results = results.append(res, ignore_index=True)

        results.to_csv(result_file, index=False)
        torch.cuda.empty_cache()

# os.system("shutdown")
