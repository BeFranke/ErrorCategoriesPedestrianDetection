from argparse import ArgumentParser
from os import path as P
import os
from time import strftime

import pandas as pd

from API.evaluator.eval import ErrorTypeEvaluator


def main(args):
    dt_folder = P.abspath(P.join(
        P.dirname(__file__),
        "..", "input", "dt",
        args.dt_folder
    ))
    gt_file = P.abspath(P.join(
        P.dirname(__file__),
        "..", "input", "gt",
        args.gt_file
    ))
    assert P.isdir(dt_folder), f"'{dt_folder}' does not exist!"
    assert P.isfile(gt_file), f"'{gt_file}' does not exist!"

    evaluator = ErrorTypeEvaluator(ap.config)

    results = []

    for dt_json in filter(lambda s: P.splitext(s)[1] == ".json", os.listdir(dt_folder)):
        print(f"Processing {dt_json} ...")
        result = evaluator.evaluate(dt_json, gt_file)
        results.append(result)

    pd.DataFrame(results).to_csv(args.out, index=False)


if __name__ == "__main__":
    ap = ArgumentParser()
    ap.add_argument("dt_folder", help="folder name containing detection files")
    ap.add_argument("gt_file", help="name of ground truth json")
    ap.add_argument("--config", default="API/cfg_eval.cfg", help="Path to config file")
    ap.add_argument("--out", default=f"../output/{strftime('%Y%m%d-%H%M%S')}.csv")
    args = ap.parse_args()
    main(args)

