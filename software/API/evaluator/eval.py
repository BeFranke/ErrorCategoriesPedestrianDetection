import yaml
from easydict import EasyDict

from coco_api import COCOeval
from software.API.evaluator.coco import COCO


class ErrorTypeEvaluator:
    def __init__(self, config_path):
        with open(config_path, "r") as file:
            self.config = EasyDict(yaml.safe_load(file))

    def evaluate(self, dt_json, gt_json):
        id = self.config.setting_id
        cocoGt = COCO(annotation_file=gt_json)
        cocoDt = cocoGt.loadRes(resFile=dt_json)
        imgIds = sorted(cocoGt.getImgIds())

        coco = COCOeval(
            cocoGt=cocoGt,
            cocoDt=cocoDt,
            env_pixel_thrs=self.config.thresholds.env_pixel_thrs,
            occ_pixel_thr=self.config.thresholds.occ_pixel_thrs,
            crowd_pixel_thrs=self.config.thresholds.crowd_pixel_thrs,
            iou_match_thrs=self.config.thresholds.iou_match_thrs,
            foreground_thrs=self.config.thresholds.foreground_thrs,
            split=self.config.image_set["val"],
            output=None,
            output_filename=None,
        )
        coco.params.imgIds = imgIds
        coco.evaluate(id)
        coco.accumulate()
        coco.summarize(id_setup=id)

        return coco.eval
