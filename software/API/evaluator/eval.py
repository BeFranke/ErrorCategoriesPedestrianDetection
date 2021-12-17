from os import path as P

import yaml
from easydict import EasyDict

from .coco_api import COCOeval
from .coco import COCO


class ErrorTypeEvaluator:
    def __init__(self, config_path, out_path=None):
        with open(config_path, "r") as file:
            self.config = EasyDict(yaml.safe_load(file))

        self.out_path = out_path

    def evaluate(self, dt_json, gt_json):
        id = self.config.setting_id
        cocoGt = COCO(annotation_file=gt_json)
        cocoDt = cocoGt.loadRes(resFile=dt_json)
        imgIds = sorted(cocoGt.getImgIds())

        d_fname = P.split(dt_json)[-1]
        g_fname = P.split(gt_json)[-1]
        output = {
            'meta': {
                'model': d_fname.split(".")[0],
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
            output=output,
            output_path=P.join(self.out_path.replace(".json", ""), d_fname),
        )
        coco.params.imgIds = imgIds
        coco.evaluate(id)
        coco.accumulate()
        coco.summarize(id_setup=id)

        # TODO output for plotting

        return coco.metrics
