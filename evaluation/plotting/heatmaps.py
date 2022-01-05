import os
from os import path as P

import numpy as np
import torch
from torchvision.transforms import Resize, InterpolationMode, Compose
from torchvision.transforms import functional as FT
from matplotlib import pyplot as plt

from evaluation.API.evaluator.coco import COCO
from evaluation.API.evaluator.coco_api import COCOeval


TARGET_RES = 250
titles = ["Environmental Occlusion", "Crowd Occlusion", "Ambiguous Occlusion", "Clear Foreground", "Standard GT"]

class SquarePad:
    def __call__(self, image):
        max_wh = max(image.size())
        _, p_top, p_left = [(max_wh - s) // 2 for s in image.size()]
        _, p_bottom, p_right = [max_wh - (s+pad) for s, pad in zip(image.size(), [0, p_top, p_left])]
        padding = (p_left, p_top, p_right, p_bottom)
        return FT.pad(image, padding, 0, 'constant')


class FixedHeightResize(object):
    def __init__(self, height):
        self.height = height

    def __call__(self, img):
        size = (self.height, self._calc_new_width(img))
        return FT.resize(img[None, :, :], size)

    def _calc_new_width(self, img):
        try:
            old_height,  old_width = img.size()
        except TypeError:
            old_height, old_width = img.shape
        aspect_ratio = old_width / old_height
        return round(self.height * aspect_ratio)


resize = Compose([
    FixedHeightResize(TARGET_RES),
    SquarePad()
])


def save_image(data, filename):
    # source:
    # https://stackoverflow.com/questions/37809697/remove-white-border-when-using-subplot-and-imshow-in-python-matplotlib
    sizes = np.shape(data)
    fig = plt.figure()
    fig.set_size_inches(1. * sizes[0] / sizes[1], 1, forward=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(data, cmap=plt.cm.get_cmap("inferno"))
    plt.savefig(filename, dpi=sizes[0], cmap='hot')
    plt.show()


def mini_eval(coco, imgId):
    gt = coco._gts[imgId, 1]

    for g in gt:
        if g['ignore']:
            g['_ignore'] = 1
        else:
            g['_ignore'] = 0
    # sort dt highest score first, sort gt ignore last
    gtind = np.argsort([g['_ignore'] for g in gt], kind='mergesort')
    gt = [gt[i] for i in gtind]
    return coco.classify_gt(
        gt,
        coco.env_pixel_thrs,
        coco.occ_pixel_thr,
        coco.crowd_pixel_thrs,
        coco.foreground_thrs,
        coco.ambfactor
    )


def main():
    gt_json = P.abspath(P.join(P.dirname(__file__), "../..", "input", "gt", "val_cityscapes.json.old"))
    # this does not really matter, but needs to be passed
    dt_json = P.abspath(P.join(P.dirname(__file__), "../..", "input", "dt", "std", "csp_1.json"))

    cocoGt = COCO(annotation_file=gt_json)
    cocoDt = cocoGt.loadRes(resFile=dt_json)
    imgIds = sorted(cocoGt.getImgIds())

    coco = COCOeval(
        cocoGt=cocoGt,
        cocoDt=cocoDt,
        env_pixel_thrs=0.5,
        occ_pixel_thr=0.45,
        crowd_pixel_thrs=0.35,
        iou_match_thrs=0.5,
        foreground_thrs=200,
        ambfactor=0.75,
        output=None,
        output_path=None
    )
    coco.params.imgIds = imgIds
    coco._prepare(0)

    heatmaps = np.zeros([5, 250, 250])
    N = np.zeros(5)

    for imgId in imgIds:
        cat_map = np.stack(mini_eval(coco, imgId))
        imap = coco._get_instance_seg(imgId)
        # sort gt ignore last
        gtind = np.argsort([g['_ignore'] for g in coco._gts[imgId, 1]], kind='mergesort')
        gt = [coco._gts[imgId, 1][i] for i in gtind]
        for i, g in enumerate(gt):
            if g['ignore']:
                continue

            inst_id = g['instance_id']
            assert g['instance_id'] // 1000 == 24
            x1 = np.clip(g['bbox'][0], a_min=0, a_max=imap.shape[1] - 1).round().astype(int)
            y1 = np.clip(g['bbox'][1], a_min=0, a_max=imap.shape[0] - 1).round().astype(int)
            x2 = np.clip(g['bbox'][0] + g['bbox'][2], a_min=0, a_max=imap.shape[1] - 1).round().astype(int)
            y2 = np.clip(g['bbox'][1] + g['bbox'][3], a_min=0, a_max=imap.shape[0] - 1).round().astype(int)
            bb = imap[y1:y2, x1:x2] == inst_id
            assert np.any(bb)

            # this is where the magic happens
            bb = resize(torch.from_numpy(bb)).numpy()

            assert np.sum(cat_map[:, i]) == 1

            idx = np.nonzero(cat_map[:, i])[0]
            heatmaps[idx] += bb
            N[idx] += 1

    # normalize
    heatmaps /= N[:, None, None]

    for map, sstr in zip(heatmaps, titles):
        save_image(map, f"../../{sstr.replace(' ', '')}_heatmap.png")

if __name__ == "__main__":
    main()
