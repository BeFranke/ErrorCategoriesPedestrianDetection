from os import path as P
import os
from time import strftime
from typing import Optional

import yaml
from easydict import EasyDict
import numpy as np

from .evaluator.coco_api import COCOeval
from .evaluator.coco import COCO


class ErrorTypeEvaluator:
    """
    Thin wrapper around coco-api, handles I/O
    """
    def __init__(self, config_path: str, out_path: Optional[str] = None):
        """
        @param config_path: path to configuration yaml file
        @param out_path: if not None, specifies output path for evaluation results (will be saved as csv)
        if None, evaluation will only be logged to stdout
        """
        with open(config_path, "r") as file:
            self.config = EasyDict(yaml.safe_load(file))

        self.out_path = out_path
        self.last_model = None

    def evaluate(self, dt_json: str, gt_json: str) -> dict[str, float]:
        """
        run evaluation on given ground truth and detections
        @param dt_json: path to json-file containing detections to evaluate
        @param gt_json: path to json-file containing corresponding ground truth
        @return: dict of metrics (name -> score)
        """
        id = self.config.setting_id
        cocoGt = COCO(annotation_file=gt_json)
        cocoDt = cocoGt.loadRes(resFile=dt_json)
        imgIds = sorted(cocoGt.getImgIds())

        d_fname = P.split(dt_json)[-1]
        g_fname = P.split(gt_json)[-1]
        self.last_model = d_fname.split(".")[0]
        output = {
            'meta': {
                'model': self.last_model,
                'dataset': g_fname.split("_")[1].split(".")[0],
                'split': g_fname.split("_")[0],
            }
        }
        for key in self.config.keys():
            output['meta'][key] = self.config[key]

        coco = COCOeval(
            cocoGt=cocoGt,
            cocoDt=cocoDt,
            env_pixel_thrs=self.config.thresholds.envPixelThrs,
            occ_pixel_thr=self.config.thresholds.occPixelThrs,
            crowd_pixel_thrs=self.config.thresholds.crowdPixelThrs,
            iou_match_thrs=self.config.thresholds.iouMatchThrs,
            foreground_thrs=self.config.thresholds.foregroundThrs,
            ambfactor=self.config.ambFactor,
            center_aligned_threshold=self.config.thresholds.centerAlignedThreshold,
            reduced_iou_threshold=self.config.thresholds.reducedIouThreshold,
            output=output,
            output_path=P.join(self.out_path, "raw",
                               d_fname.replace(".json", "") + "___" + str(self.config.thresholds.iouMatchThrs))
        )
        coco.params.imgIds = imgIds
        coco.evaluate(id)
        coco.accumulate()
        coco.summarize(id_setup=id)

        self.save_plotting_data(coco)

        coco.metrics["setting_id"] = id
        coco.metrics["iouThreshold"] = self.config.thresholds.iouMatchThrs
        return coco.metrics

    def save_plotting_data(self, cocoEval: COCOeval) -> None:
        """
        save various numpy arrays that can be used for plotting
        @param cocoEval: cocoEval object that has been used for evaluation
            (evaluate, accumulate and summarize must have been called!)
        """
        PLOT_OUTPUT_PATH = P.join(self.out_path, "plotting-raw")
        np.save(os.path.join(PLOT_OUTPUT_PATH, f"{self.last_model}__fppi__{self.config.setting_id}.npy"),
                cocoEval.eval["fppi"])
        np.save(os.path.join(PLOT_OUTPUT_PATH, f"{self.last_model}__scores__{self.config.setting_id}.npy"),
                cocoEval.eval["dt_scores"])
        np.save(os.path.join(PLOT_OUTPUT_PATH, f"{self.last_model}__mr__{self.config.setting_id}.npy"),
                cocoEval.eval["mr"])
        for i, (err_c, s) in enumerate(zip(cocoEval.eval['fp_ratio'], ['Poor Localization', 'Ghost Detections',
                                                                       'Scaling Errors'])):
            np.save(os.path.join(PLOT_OUTPUT_PATH, f"{self.last_model}__fp_ratio_{s.replace(' ', '')}__"
                                                   f"{self.config.setting_id}.npy"), err_c)

        for i, (err_c, s) in enumerate(zip(cocoEval.eval['error_cumsums_fp'], ['Poor Localization', 'Ghost Detections',
                                                                               'Scaling Errors'])):
            np.save(os.path.join(PLOT_OUTPUT_PATH, f"{self.last_model}__fp_counts_{s.replace(' ', '')}__"
                                                   f"{self.config.setting_id}.npy"), err_c)

        for i, (err_c, s) in enumerate(zip(cocoEval.eval['error_map'],
                                           ['Crowd Occlusion Errors', 'Environmental Occlusion Errors',
                                            'Foreground Errors', 'Standard Errors', 'Mixed Occlusion Errors'])):
            np.save(os.path.join(PLOT_OUTPUT_PATH, f"{self.last_model}__{s.replace(' ', '')}__"
                                                   f"{self.config.setting_id}.npy"), err_c)

        np.save(os.path.join(PLOT_OUTPUT_PATH, f"{self.last_model}__recall__{self.config.setting_id}.npy"),
                cocoEval.eval['recall'])
        for j, (err_c, s) in enumerate(
                zip(cocoEval.eval['cat_precision'], ['All', 'Crowd Occlusion Errors', 'Environmental Occlusion Errors',
                                                     'Foreground Errors', 'Standard Errors', 'Mixed Occlusion Errors',
                                                     'Poor Localization', 'Ghost Detections',
                                                     'Scaling Errors'])):
            np.save(os.path.join(PLOT_OUTPUT_PATH, f"{self.last_model}__precision_{s.replace(' ', '')}__"
                                                   f"{self.config.setting_id}.npy"), err_c)
