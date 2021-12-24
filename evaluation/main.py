from argparse import ArgumentParser
from os import path as P
import os
from time import strftime

import pandas as pd

from API.eval import ErrorTypeEvaluator


def make_folder(fpath: str) -> None:
    """
    creates folder with subfolder structure
    @param fpath: path to main folder
    """
    os.mkdir(fpath)
    os.mkdir(P.join(fpath, "raw"))
    os.mkdir(P.join(fpath, "plotting-raw"))
    os.mkdir(P.join(fpath, "figures"))


def main(args):
    # set folder containing one detection-json per evaluated model
    dt_folder = P.abspath(P.join(
        P.dirname(__file__),
        "..", "input", "dt",
        args.dt_folder
    ))
    # set file containing labels
    gt_file = P.abspath(P.join(
        P.dirname(__file__),
        "..", "input", "gt",
        args.gt_file
    ))
    assert P.isdir(dt_folder), f"'{dt_folder}/' does not exist!"
    assert P.isfile(gt_file), f"'{gt_file}' does not exist!"
    # output dir should always be newly created and empty
    make_folder(args.out)

    # run evaluation for each detection-file
    evaluator = ErrorTypeEvaluator(args.config, args.out)
    results = []
    for dt_json in filter(lambda s: P.splitext(s)[1] == ".json", os.listdir(dt_folder)):
        print(f"Processing {dt_json} ...")
        path = P.join(dt_folder, dt_json)
        result = evaluator.evaluate(path, gt_file)
        result["model"] = dt_json.split(".")[0]
        results.append(result)

    if results:
        df = pd.DataFrame(results)
        df.to_csv(os.path.join(args.out, "results.csv"), index=False)
    else:
        raise FileNotFoundError("No detection files could be found!")


if __name__ == "__main__":
    # parse arguments and launch main
    ap = ArgumentParser()
    ap.add_argument("dt_folder", help="folder name containing detection files")
    ap.add_argument("gt_file", help="name of ground truth json")
    ap.add_argument("--config", default="API/cfg_eval.yaml", help="Path to config file")
    ap.add_argument(
        "--out",
        default=P.abspath(P.join(
            P.dirname(__file__), "..", "output", strftime('%Y%m%d-%H%M%S')
        ))
    )
    args = ap.parse_args()
    main(args)
