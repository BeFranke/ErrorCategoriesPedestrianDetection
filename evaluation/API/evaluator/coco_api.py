import warnings

import numpy as np
import datetime
import time
import os

import json

from collections import defaultdict
from easydict import EasyDict as edict
# from . import mask as maskUtils
import copy
import scipy.io as sio
from PIL import Image

from ..utils.utils import update_recursive

OCCLUSION_CLASSES_SEGM = list(set(range(11, 34)).union({4, 5}) - {16, 22, 23, 24, 25})


class COCOeval:
    # Interface for evaluating detection on the Microsoft COCO dataset.
    #
    # The usage for CocoEval is as follows:
    #  cocoGt=..., cocoDt=...       # load dataset and results
    #  E = CocoEval(cocoGt,cocoDt); # initialize CocoEval object
    #  E.params.recThrs = ...;      # set parameters as desired
    #  E.evaluate();                # run per image evaluation
    #  E.accumulate();              # accumulate per image results
    #  E.summarize();               # display summary metrics of results
    # For example usage see evalDemo.m and http://mscoco.org/.
    #
    # The evaluation parameters are as follows (defaults in brackets):
    #  imgIds     - [all] N img ids to use for evaluation
    #  catIds     - [all] K cat ids to use for evaluation
    #  iouThrs    - [.5:.05:.95] T=10 IoU thresholds for evaluation
    #  recThrs    - [0:.01:1] R=101 recall thresholds for evaluation
    #  areaRng    - [...] A=4 object area ranges for evaluation
    #  maxDets    - [1 10 100] M=3 thresholds on max detections per image
    #  iouType    - ['segm'] set iouType to 'segm', 'bbox' or 'keypoints'
    #  iouType replaced the now DEPRECATED useSegm parameter.
    #  useCats    - [1] if true use category labels for evaluation
    # Note: if useCats=0 category labels are ignored as in proposal scoring.
    # Note: multiple areaRngs [Ax2] and maxDets [Mx1] can be specified.
    #
    # evaluate(): evaluates detections on every image and every category and
    # concats the results into the "evalImgs" with fields:
    #  dtIds      - [1xD] id for each of the D detections (dt)
    #  gtIds      - [1xG] id for each of the G ground truths (gt)
    #  dtMatches  - [TxD] matching gt id at each IoU or 0
    #  gtMatches  - [TxG] matching dt id at each IoU or 0
    #  dtScores   - [1xD] confidence of each dt
    #  gtIgnore   - [1xG] ignore flag for each gt
    #  dtIgnore   - [TxD] ignore flag for each dt at each IoU
    #
    # accumulate(): accumulates the per-image, per-category evaluation
    # results in "evalImgs" into the dictionary "eval" with fields:
    #  params     - parameters used for evaluation
    #  date       - date evaluation was performed
    #  counts     - [T,R,K,A,M] parameter dimensions (see above)
    #  precision  - [TxRxKxAxM] precision for every evaluation setting
    #  recall     - [TxKxAxM] max recall for every evaluation setting
    # Note: precision and recall==-1 for settings with no gt objects.
    #
    # See also coco, mask, pycocoDemo, pycocoEvalDemo
    #
    # Microsoft COCO Toolbox.      version 2.0
    # Data, paper, and tutorials available at:  http://mscoco.org/
    # Code written by Piotr Dollar and Tsung-Yi Lin, 2015.
    # Licensed under the Simplified BSD License [see coco/license.txt]
    def __init__(self, cocoGt, cocoDt, env_pixel_thrs, occ_pixel_thr, crowd_pixel_thrs,
            iou_match_thrs, foreground_thrs, ambfactor, center_aligned_threshold, reduced_iou_threshold,
            output, output_path, iouType='bbox', split="val"):
        '''
        Initialize CocoEval using coco APIs for gt and dt
        :param cocoGt: coco object with ground truth annotations
        :param cocoDt: coco object with detection results
        :return: None
        '''
        if not iouType:
            print('iouType not specified. use default iouType segm')
        self.cocoGt   = cocoGt              # ground truth COCO API
        self.cocoDt   = cocoDt              # detections COCO API
        self.evalImgs = defaultdict(list)   # per-image per-category evaluation results [KxAxI] elements
        self.eval     = {}                  # accumulated evaluation results
        self._gts = defaultdict(list)       # gt for evaluation
        self._dts = defaultdict(list)       # dt for evaluation
        self.params = Params(
            iouType=iouType,
            thresholds = edict({
                'envPixelThrs': env_pixel_thrs,
                'occPixelThrs': occ_pixel_thr,
                'crowdPixelThrs': crowd_pixel_thrs,
                'iouMatchThrs': iou_match_thrs,
                'foregroundThrs': foreground_thrs,
                'centerAlignedThreshold': center_aligned_threshold,
                'reducedIouThreshold': reduced_iou_threshold,
            })
        
        ) # parameters
        self.params
        self._paramsEval = {}               # parameters for evaluation
        self.stats = []                     # result summarization
        self.ious = {}                      # ious between all gts and dts
        if not cocoGt is None:
            self.params.imgIds = sorted(cocoGt.getImgIds())
            self.params.catIds = sorted(cocoGt.getCatIds())

        self.reduced_iou_threshold = self.params.thresholds.reducedIouThreshold
        self.center_aligned_threshold = self.params.thresholds.centerAlignedThreshold
        assert self.params.normalization in ["total", "class"]
        self.normalization = self.params.normalization
        self.env_pixel_thrs = self.params.thresholds.envPixelThrs
        self.occ_pixel_thr = self.params.thresholds.occPixelThrs
        self.crowd_pixel_thrs = self.params.thresholds.crowdPixelThrs
        self.iou_match_thrs = self.params.thresholds.iouMatchThrs
        self.foreground_thrs = self.params.thresholds.foregroundThrs
        self.ambfactor = self.params.ambFactor

        self.segm_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "..", "input", "datasets")
        )

        if output_path is not None:
            self.output_json_path = output_path

        self.output = output
        if output is not None:
            update = {
                'meta': dict(
                    env_pixel_thrs=self.env_pixel_thrs,
                    occ_pixel_thrs=self.occ_pixel_thr,
                    crowd_pixel_thrs=self.crowd_pixel_thrs,
                    iou_match_thrs=self.iou_match_thrs,
                    foreground_thrss=self.foreground_thrs,
                ),
                'imgs': cocoGt.imgs,
                'gts': dict(),
                'dts': dict()
            }
            update_recursive(self.output, update)
        
        self.split = split

    def _prepare(self, id_setup):
        '''
        Prepare ._gts and ._dts for evaluation based on params
        :return: None
        '''
        p = self.params
        if p.useCats:
            gts=self.cocoGt.loadAnns(self.cocoGt.getAnnIds(imgIds=p.imgIds, catIds=p.catIds))
            dts=self.cocoDt.loadAnns(self.cocoDt.getAnnIds(imgIds=p.imgIds, catIds=p.catIds))
        else:
            gts=self.cocoGt.loadAnns(self.cocoGt.getAnnIds(imgIds=p.imgIds))
            dts=self.cocoDt.loadAnns(self.cocoDt.getAnnIds(imgIds=p.imgIds))

        # set ignore flag
        for gt in gts:
            gt['ignore'] = gt['ignore'] if 'ignore' in gt else 0
            gt['ignore'] = 1 if (gt['height'] < self.params.HtRng[id_setup][0] or
                                 gt['height'] > self.params.HtRng[id_setup][1]) or \
                                (gt['vis_ratio'] < self.params.VisRng[id_setup][0] or
                                 gt['vis_ratio'] > self.params.VisRng[id_setup][1]) \
                else gt['ignore']

        self._gts = defaultdict(list)       # gt for evaluation
        self._dts = defaultdict(list)       # dt for evaluation
        for gt in gts:
            self._gts[gt['image_id'], gt['category_id']].append(gt)
        for dt in dts:
            self._dts[dt['image_id'], dt['category_id']].append(dt)
        self.evalImgs = defaultdict(list)   # per-image per-category evaluation results
        self.eval     = {}                  # accumulated evaluation results
        
        if self.output is not None:
            self.output['meta']['setup'] = self.params.SetupLbl[id_setup]

    def evaluate(self, id_setup):
        '''
        Run per image evaluation on given images and store results (a list of dict) in self.evalImgs
        :return: None
        '''
        tic = time.time()
        # print('Running per image evaluation...')
        p = self.params
        # add backward compatibility if useSegm is specified in params
        if not p.useSegm is None:
            p.iouType = 'segm' if p.useSegm == 1 else 'bbox'
            print('useSegm (deprecated) is not None. Running {} evaluation'.format(p.iouType))
        # print('Evaluate annotation type *{}*'.format(p.iouType))
        p.imgIds = list(np.unique(p.imgIds))
        if p.useCats:
            p.catIds = list(np.unique(p.catIds))
        p.maxDets = sorted(p.maxDets)
        self.params=p

        self._prepare(id_setup)
        # loop through images, area range, max detection number
        catIds = p.catIds if p.useCats else [-1]

        computeIoU = self.computeIoU

        self.ious = {(imgId, catId): computeIoU(imgId, catId) \
                        for imgId in p.imgIds
                        for catId in catIds}

        evaluateImg = self.evaluateImg
        maxDet = p.maxDets[-1]
        HtRng = self.params.HtRng[id_setup]
        VisRng = self.params.VisRng[id_setup]
        self.evalImgs = [evaluateImg(imgId, catId, HtRng, VisRng, maxDet)
                 for catId in catIds
                 for imgId in p.imgIds
             ]
        self._paramsEval = copy.deepcopy(self.params)
        toc = time.time()
        # print('DONE (t={:0.2f}s).'.format(toc-tic))

    def computeIoU(self, imgId, catId):
        p = self.params
        if p.useCats:
            gt = self._gts[imgId,catId]
            dt = self._dts[imgId,catId]
        else:
            gt = [_ for cId in p.catIds for _ in self._gts[imgId,cId]]
            dt = [_ for cId in p.catIds for _ in self._dts[imgId,cId]]
        if len(gt) == 0 and len(dt) ==0:
            return []
        inds = np.argsort([-d['score'] for d in dt], kind='mergesort')
        dt = [dt[i] for i in inds]
        if len(dt) > p.maxDets[-1]:
            dt=dt[0:p.maxDets[-1]]


        if p.iouType == 'segm':
            g = [g['segmentation'] for g in gt]
            d = [d['segmentation'] for d in dt]
        elif p.iouType == 'bbox':
            g = [g['bbox'] for g in gt]
            d = [d['bbox'] for d in dt]
        else:
            raise Exception('unknown iouType for iou computation')


        # compute iou between each dt and gt region
        iscrowd = [int(o['ignore']) for o in gt]
        ious = self.iou(d,g,iscrowd)
        return ious

    def iou( self, dts, gts, pyiscrowd ):
        dts = np.asarray(dts)
        gts = np.asarray(gts)
        pyiscrowd = np.asarray(pyiscrowd)
        ious = np.zeros((len(dts), len(gts)))
        for j, gt in enumerate(gts):
            gx1 = gt[0]
            gy1 = gt[1]
            gx2 = gt[0] + gt[2]
            gy2 = gt[1] + gt[3]
            garea = gt[2] * gt[3]
            for i, dt in enumerate(dts):
                dx1 = dt[0]
                dy1 = dt[1]
                dx2 = dt[0] + dt[2]
                dy2 = dt[1] + dt[3]
                darea = dt[2] * dt[3]

                unionw = min(dx2,gx2)-max(dx1,gx1)
                if unionw <= 0:
                    continue
                unionh = min(dy2,gy2)-max(dy1,gy1)
                if unionh <= 0:
                    continue
                t = unionw * unionh

                # COCOs way to handle ignored gts:
                # D (detection), GT (ground truth), darea (area of D)
                # D is matched to ignored GT if (intersection / darea) > 0.5
                if pyiscrowd[j]:
                    unionarea = darea
                else:
                    unionarea = darea + garea - t

                ious[i, j] = float(t)/unionarea
        return ious

    @staticmethod
    def contained(dt, gt):
        # warnings.warn("contained is still modified")
        # return False

        gx1 = gt['bbox'][0]
        gy1 = gt['bbox'][1]
        gx2 = gt['bbox'][0] + gt['bbox'][2]
        gy2 = gt['bbox'][1] + gt['bbox'][3]
        dx1 = dt['bbox'][0]
        dy1 = dt['bbox'][1]
        dx2 = dt['bbox'][0] + dt['bbox'][2]
        dy2 = dt['bbox'][1] + dt['bbox'][3]
        return dx1 >= gx1 and dy1 >= gy1 and dx2 <= gx2 and dy2 <= gy2

    @staticmethod
    def find_center_aligned_off_scale(dts, gts, max_center_offset=0.2):
        def get_center_bbox(gt, f):
            x1, y1, w, h = gt['bbox']
            x_center, y_center = x1 + w / 2, y1 + h / 2
            x1_c, y1_c = x_center - w * f / 2, y_center - h * f / 2
            x2_c, y2_c = x_center + w * f / 2, y_center + h * f / 2
            return np.array([x1_c, y1_c, x2_c, y2_c])

        center_aligned_bitmap = np.zeros(len(dts), dtype=bool)
        for gt in gts:
            # x1, y1, x2, y2
            gt_center_box = get_center_bbox(gt, max_center_offset)
            h_gt = gt["height"]
            for i, dt in enumerate(dts):
                dt_center = np.array([
                    dt['bbox'][0] + dt['bbox'][2] / 2,
                    dt['bbox'][1] + dt['bbox'][3] / 2
                ])
                center_aligned_bitmap[i] = center_aligned_bitmap[i] or \
                                           (gt_center_box[0] <= dt_center[0] <= gt_center_box[2] and
                                            gt_center_box[1] <= dt_center[1] <= gt_center_box[3])

        return center_aligned_bitmap

    def _get_instance_seg(self, imgId):
        img_dct = self.cocoGt.imgs[imgId]
        assert img_dct['id'] == imgId
        img_name = img_dct['im_name']
        isegm_path = os.path.join(
            self.segm_path, "cityscapes", "gtFine", self.split,
            # exploiting the fact that the folder name (the city) is contained in the file name
            img_name.split("_")[0],
            img_name.replace("_leftImg8bit.png", "_gtFine_instanceIds.png")
        )
        return np.asarray(Image.open(isegm_path))

    def _get_semantic_seg(self, imgId):
        img_dct = self.cocoGt.imgs[imgId]
        assert img_dct['id'] == imgId
        img_name = img_dct['im_name']
        isegm_path = os.path.join(
            self.segm_path, "cityscapes", "gtFine", self.split,
            # exploiting the fact that the folder name (the city) is contained in the file name
            img_name.split("_")[0],
            img_name.replace("_leftImg8bit", "_gtFine_labelIds.png")
        )
        return np.asarray(Image.open(isegm_path))

    def classify_gt(self, gts, env_thrs, occ_thrs, crowd_thrs, foreground_thrs, amb_factor, imgId, crowd_classes=None):
        """
        determine which GTs are occluded, and which kind of occlusion it is
        :param instance_seg: np.ndarray that gives the images instance datasets
        :param gts: list of GTs
        :param env_thrs: threshold for environmental occlusion
        :param crowd_thrs: threshold for crowd occlusion
        :param pedestrian_class: id for person-class, best left unchanged
        :return numpy arrays 'env', 'crd', 'mxd' each a bit map for the respective occlusion kind
        """
        # split into instance and class ids, source:
        # https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/preparation/json2instanceImg.py
        # class_id_map = instance_seg // 1000
        if crowd_classes is None:
            crowd_classes = {24, 25}
        env, crd, mxd, fgd, oth = [np.zeros(len(gts), dtype=bool) for _ in range(5)]
        for i, gt in enumerate(gts):
            if gt['ignore']:
                continue
            try:
                inst_vis_ratio = gt["inst_vis_ratio"]
                env_occl_ratio = gt["env_occl_ratio"]
                crowd_occl_ratio = gt["crowd_occl_ratio"]
            except KeyError as e:
                if os.environ.get("COCO_FALLBACK"):
                    warnings.warn("key inst_vis_ratio not found, trying to get segmentation masks...")

                    semantic_seg = self._get_semantic_seg(imgId)
                    instance_seg = self._get_instance_seg(imgId)

                    # Update 2021-12-22: This clipping is necessary to get correct boolean masks, but it is now ensured to not
                    # distort the occlusion ratios
                    x1 = np.clip(gt['bbox'][0], a_min=0, a_max=instance_seg.shape[1] - 1).round().astype(int)
                    y1 = np.clip(gt['bbox'][1], a_min=0, a_max=instance_seg.shape[0] - 1).round().astype(int)
                    x2 = np.clip(gt['bbox'][0] + gt['bbox'][2], a_min=0,
                                 a_max=instance_seg.shape[1] - 1).round().astype(int)
                    y2 = np.clip(gt['bbox'][1] + gt['bbox'][3], a_min=0,
                                 a_max=instance_seg.shape[0] - 1).round().astype(int)
                    id = gt['instance_id']
                    assert np.isin(gt['instance_id'] // 1000, crowd_classes)
                    pedestrian_map = np.isin(semantic_seg[y1:y2, x1:x2], crowd_classes)
                    instance_map = instance_seg[y1:y2, x1:x2] == id
                    env_map = np.isin(semantic_seg[y1:y2, x1:x2], OCCLUSION_CLASSES_SEGM)
                    inst_vis_ratio = np.sum(instance_map) / ((y2 - y1) * (x2 - x1))
                    env_occl_ratio = np.sum(env_map) / ((y2 - y1) * (x2 - x1))
                    crowd_occl_ratio = 1 - (np.sum(instance_map) / np.sum(pedestrian_map))
                else:
                    raise e

            if inst_vis_ratio < occ_thrs:
                # assert np.sum(instance_map) > 0
                env_occluded = env_occl_ratio > env_thrs
                # crowd occlusion is measured by (area_instance / area_pedestrian) \in [0, 1]
                crowd_occluded = crowd_occl_ratio > crowd_thrs
                # assert np.sum(pedestrian_map) > 0
                # assert np.sum(np.logical_not(pedestrian_map)) > 0
                ambiguous = env_occl_ratio > amb_factor * env_thrs and crowd_occl_ratio > amb_factor * crowd_thrs

            else:
                crowd_occluded = env_occluded = False
                ambiguous = False

            mxd[i] = ambiguous
            env[i] = env_occluded and not mxd[i]
            crd[i] = crowd_occluded and not mxd[i]
            assert not (env[i] and crd[i])
            if not (mxd[i] or env[i] or crd[i]):
                if gt['height'] >= foreground_thrs:
                    fgd[i] = True
                else:
                    oth[i] = True

            assert gt["ignore"] or (env[i] + crd[i] + mxd[i] + fgd[i] + oth[i] == 1)

        return env, crd, mxd, fgd, oth

    def force_match(self, ec_gt, ec_dt, dt, gt, gtIg, ious, crowd_occ_dts, gtm, dtm, t, tind=0):
        iou = min([t, 1 - 1e-10])
        for gind, g in enumerate(gt):
            if not ec_gt[gind] or gtIg[gind] == 1:
                continue
            dtScores = [d['score'] for d in dt]
            dtIds = [d['id'] for d in dt]
            # Get old match - dind
            _dind = int(gtm[tind, gind])
            # Already matched
            if _dind != 0:
                _dtid = dtIds.index(_dind)
                # Get old best score
                score = _score = dtScores[_dtid]
            else:
                score = _score = 0.0
            betterid = None
            for dind, d in enumerate(dt):
                if ious[dind, gind] >= iou and crowd_occ_dts[tind, dind] and dtScores[dind] > score:
                    score = dtScores[dind]
                    betterid = dind
            # if g['id'] == 1020:
            #     print('boo')
            if betterid is not None:
                ec_dt[tind, betterid] = 1
                gtm[tind, gind] = dtIds[betterid]
                # Del old detection
                if _dind != 0:
                    dtm[tind, _dtid] = 0
                    ec_dt[tind, _dtid] = 0
                # print('Upgrade gtId {} score-wise from {} to {}'.format(g['id'], _score, score))

    def evaluateImg(self, imgId, catId, hRng, vRng, maxDet):
        '''
        perform evaluation for single category and image
        :return: dict (single image results)
        '''
        p = self.params
        if p.useCats:
            gt = self._gts[imgId,catId]
            dt = self._dts[imgId,catId]
        else:
            gt = [_ for cId in p.catIds for _ in self._gts[imgId,cId]]
            dt = [_ for cId in p.catIds for _ in self._dts[imgId,cId]]
        if len(gt) == 0 and len(dt) ==0:
            return None

        for g in gt:
            if g['ignore']:
                g['_ignore'] = 1
            else:
                g['_ignore'] = 0
        # sort dt highest score first, sort gt ignore last
        gtind = np.argsort([g['_ignore'] for g in gt], kind='mergesort')
        gt = [gt[i] for i in gtind]
        dtind = np.argsort([-d['score'] for d in dt], kind='mergesort')
        # dtind = np.argsort([-d['score'] for d in dt])
        dt = [dt[i] for i in dtind[0:maxDet]]

        dtind = [i for i in range(len(dt)) if
                 hRng[0] / self.params.expFilter <= dt[i]['height'] < hRng[1] * self.params.expFilter]
        dt = [d for d in dt if
              hRng[0] / self.params.expFilter <= d['height'] < hRng[1] * self.params.expFilter]

        # load computed ious
        if len(dtind) > 0:
            ious = self.ious[imgId, catId][dtind, :] if len(self.ious[imgId, catId]) > 0 else self.ious[imgId, catId]
            ious = ious[:, gtind]
        else:
            ious = []

        T = len(p.iouThrs)
        G = len(gt)
        D = len(dt)
        gtm  = np.zeros((T,G))
        dtm  = np.zeros((T,D))

        iscrowd, g, foreground, env_occluded, crowd_occluded, mixed_occluded = [
            np.zeros_like(gt) for _ in range(6)
        ]

        env_occluded, crowd_occluded, mixed_occluded, foreground, background = self.classify_gt(
            gt,
            self.env_pixel_thrs,
            self.occ_pixel_thr,
            self.crowd_pixel_thrs,
            self.foreground_thrs,
            self.ambfactor,
            imgId
        )

        for i, o in enumerate(gt):
            iscrowd[i] = int(o['ignore'])
            g[i] = o['bbox']

        assert len(crowd_occluded.shape) == 1

        gtIg = np.array([g['_ignore'] for g in gt])
        dtIg = np.zeros((T,D))

        assert np.sum(
            env_occluded + crowd_occluded + mixed_occluded + foreground + background
        ) == len(gt) - np.sum(gtIg)

        # error category analysis
        # GT
        crowd_occ_dts = np.zeros([len(p.iouThrs), len(dt)], dtype=bool)
        env_oc_dts = np.zeros([len(p.iouThrs), len(dt)], dtype=bool)
        foreground_dts = np.zeros([len(p.iouThrs), len(dt)], dtype=bool)
        background_dts = np.zeros([len(p.iouThrs), len(dt)], dtype=bool)
        mixed_occ_dts = np.zeros([len(p.iouThrs), len(dt)], dtype=bool)

        #FP
        scaling_dts = np.zeros([len(p.iouThrs), len(dt)], dtype=bool)
        localization_dts = np.zeros([len(p.iouThrs), len(dt)], dtype=bool)
        ghost_detection_dts = np.zeros([len(p.iouThrs), len(dt)], dtype=bool)
        unmatched_gt_fn = np.zeros([5, len(p.iouThrs)], dtype=bool)

        intersects_gt_nums = np.zeros([len(p.iouThrs), len(dt)], dtype=int) - 1
        
        if self.output is not None:
            self.output['dts'][int(imgId)] = dt
            self.output['gts'][int(imgId)] = gt

            for i, g in enumerate(self.output['gts'][imgId]):
                if foreground[i]:
                    self.output['gts'][imgId][i]['error_type'] = "foreground"
                elif env_occluded[i]:
                    self.output['gts'][imgId][i]['error_type'] = "env_occ"
                elif crowd_occluded[i]:
                    self.output['gts'][imgId][i]['error_type'] = "crd_occ"
                elif mixed_occluded[i]:
                    self.output['gts'][imgId][i]['error_type'] = "mxd_occ"
                else:
                    assert background[i] or g['ignore']
                    self.output['gts'][imgId][i]['error_type'] = "background"

        # ious = np.round(ious, 2)  # Yes, we could recover some matches this way: Valid?
        if not len(ious) == 0:
            for tind, t in enumerate(p.iouThrs):
                last_score = 1
                for dind, d in enumerate(dt):
                    if self.output is not None:
                        self.output['dts'][imgId][dind]['matched_gt_ignore'] = 0  # this can be overwritten later
                    assert last_score >= d['score'], f"{last_score} > {d['score']}"
                    last_score = d['score']
                    # information about best match so far (m=-1 -> unmatched)
                    iou = min([t, 1 - 1e-10])
                    bstOa = iou
                    bstg = -2
                    bstm = -2
                    for gind, g in enumerate(gt):
                        if self.output is not None and 'matched_dt' not in self.output['gts'][imgId][gind]:
                            self.output['gts'][imgId][gind]['matched_dt'] = 0
                        m = gtm[tind, gind]
                        if ious[dind, gind] > t:
                            intersects_gt_nums[tind, dind] = gind

                        if m > 0:
                            # if this gt already matched, and not a crowd, continue
                            continue
                        # if dt matched to reg gt, and on ignore gt, stop
                        if bstm != -2 and gtIg[gind] == 1:
                            break
                        # continue to next gt unless better match made
                        if ious[dind, gind] < bstOa:
                            continue
                        # if match successful and best so far, store appropriately
                        bstOa = ious[dind, gind]
                        bstg = gind
                        if gtIg[gind] == 0:
                            bstm = 1
                        else:
                            bstm = -1

                    if bstg == -2:
                        continue

                    dtIg[tind, dind] = gtIg[bstg]
                    dtm[tind, dind] = gt[bstg]['id']
                    
                    if bstm != -2 and self.output is not None:
                        self.output['dts'][imgId][dind]['matched_gt'] = gt[bstg]['id']
                        self.output['dts'][imgId][dind]['matched_gt_ignore'] = gt[bstg]['ignore']
                        self.output['gts'][imgId][bstg]['matched_dt'] = d['id']

                  
                    if bstm == 1:
                        gtm[tind, bstg] = d['id']

                    # save maximum confidence dt to all relevant error categories
                    # each detection indexed here is by definition a TP
                    # if we arrive here, bstg has not been matched yet
                    if mixed_occluded[bstg] and bstm == 1:
                        mixed_occ_dts[tind, dind] = 1
                    elif crowd_occluded[bstg] and bstm == 1:
                        crowd_occ_dts[tind, dind] = 1
                    elif env_occluded[bstg] and bstm == 1:
                        env_oc_dts[tind, dind] = 1
                    elif foreground[bstg] and bstm == 1:
                        foreground_dts[tind, dind] = 1
                    elif background[bstg] and bstm == 1:
                        background_dts[tind, dind] = 1

                # Force Match and Re-Match for Foreground with Crowd Occlusion: Share detections!
                if self.params.force_foreground_match:
                    self.force_match(foreground, foreground_dts, dt, gt, gtIg, ious, crowd_occ_dts, gtm, dtm, p.iouThrs)
                if self.params.force_background_match:
                    self.force_match(background, background_dts, dt, gt, gtIg, ious, crowd_occ_dts, gtm, dtm, p.iouThrs)

                # new FP definitions
                center_aligned = self.find_center_aligned_off_scale(dt, gt, self.center_aligned_threshold)
                assert len(center_aligned) == len(dtm[tind]) == len(intersects_gt_nums[tind])
                scaling_dts[tind] = np.logical_and.reduce((
                    np.logical_not(dtm[tind]),
                    center_aligned,
                ))
                # IMPORTANT NOTE:
                # this definition does not care about scores, so a highly confident, but too small detection might get
                #  classified as a multi detection if a low confidence detection can be matched to the same GT!
                # helper array for checking if a contained gt is also matched
                ious_matched = ious.copy()
                # CHANGED: multi-detctions are now "poor localization" error, so the "GT is matched" condition is no
                # longer required
                # ious_matched[:, np.logical_not(gtm[tind].astype(bool))] = 0
                localization_dts[tind] = np.logical_and.reduce((
                    np.logical_not(dtm[tind]),
                    np.max(ious_matched, axis=1, initial=0) > self.reduced_iou_threshold,
                    ~ scaling_dts[tind]
                ))
                ghost_detection_dts[tind] = np.logical_and.reduce((
                    np.logical_not(dtm[tind]),
                    np.logical_not(localization_dts[tind]),
                    np.logical_not(scaling_dts[tind])
                ))

                # ignore flags
                crowd_occ_dts[tind] = np.logical_and(
                    crowd_occ_dts[tind],
                    np.logical_not(dtIg),
                )
                env_oc_dts[tind] = np.logical_and(
                    env_oc_dts[tind],
                    np.logical_not(dtIg)
                )
                foreground_dts[tind] = np.logical_and(
                    foreground_dts[tind],
                    np.logical_not(dtIg)
                )
                background_dts[tind] = np.logical_and(
                    background_dts[tind],
                    np.logical_not(dtIg)
                )
                mixed_occ_dts[tind] = np.logical_and(
                    mixed_occ_dts[tind],
                    np.logical_not(dtIg)
                )
                localization_dts[tind] = np.logical_and(
                    localization_dts,
                    np.logical_not(dtIg)
                )
                ghost_detection_dts[tind] = np.logical_and(
                    ghost_detection_dts,
                    np.logical_not(dtIg)
                )
                scaling_dts[tind] = np.logical_and(
                    scaling_dts,
                    np.logical_not(dtIg)
                )
                
                if self.output is not None:
                    for i, d in enumerate(self.output['dts'][imgId]):
                        if localization_dts[tind][i]:
                            d["error_type"] = "localization"
                        elif ghost_detection_dts[tind][i]:
                            d["error_type"] = "ghost"
                        elif scaling_dts[tind][i]:
                            d["error_type"] = "scale"
                        else:
                            d["error_type"] = "none"

        # store results for given image and category
        return {
            'image_id': imgId,
            'category_id': catId,
            'hRng': hRng,
            'vRng': vRng,
            'maxDet': maxDet,
            'dtIds': [d['id'] for d in dt],
            'gtIds': [g['id'] for g in gt],
            'dtMatches': dtm,
            'gtMatches': gtm,
            'dtScores': [d['score'] for d in dt],
            'gtIgnore': gtIg,
            'dtIgnore': dtIg,

            'dtHeights': [d['height'] for d in dt],
            'dtWidths': [d['width'] for d in dt],
            'dtBoxes': [d['bbox'] for d in dt],
            'gtBoxes': [g['bbox'] for g in gt],
            'gtVisRatios': [g['vis_ratio'] for g in gt],
            'gtHeights': [g['height'] for g in gt],

            # GT
            'crowdOcclusionDts': crowd_occ_dts,  # co
            'envOcclusionDts': env_oc_dts,  # eo
            'foregroundDts': foreground_dts,  # fg
            'mixedOcclusionDts': mixed_occ_dts,
            'backgroundDts': background_dts,

            'foreground': foreground,  # fg
            'background': background,
            'env_occluded': env_occluded,
            'crowd_occluded': crowd_occluded,
            'amb_occluded': mixed_occluded,

            # FP
            'localizationDts': localization_dts,  # md
            'ghostDetectionDts': ghost_detection_dts,  # gd
            'scalingDts': scaling_dts,

            'unmatched_gt_fn': unmatched_gt_fn,

            'numCrowdOcclusionGT': np.sum(np.logical_and(crowd_occluded, np.logical_not(gtIg))),
            'numEnvOcclusionGT': np.sum(np.logical_and(env_occluded, np.logical_not(gtIg))),
            'numMixedOcclusionGT': np.sum(np.logical_and(mixed_occluded, np.logical_not(gtIg))),
            'numForegroundGT': np.sum(np.logical_and(foreground, np.logical_not(gtIg))),
            'numBackgroundGT': np.sum(np.logical_and(background, np.logical_not(gtIg)))
        }

    def accumulate(self, p=None, verbose=False):
        '''
        Accumulate per image evaluation results and store the result in self.eval
        :param p: input params for evaluation
        :return: None
        '''
        # print('Accumulating evaluation results...')
        tic = time.time()
        if not self.evalImgs:
            print('Please run evaluate() first')
        # allows input customized parameters
        if p is None:
            p = self.params
        p.catIds = p.catIds if p.useCats == 1 else [-1]
        T = len(p.iouThrs)
        R = len(p.fppiThrs)
        K = len(p.catIds) if p.useCats else 1
        M = len(p.maxDets)
        ys = -np.ones((T, R, K, M))  # -1 for the precision of absent categories
        sampled_precision = -np.ones((9, T, len(p.recThrs), K, M))
        sampled_recall = -np.ones((T, len(p.recThrs), K, M))
        sampled_mr_fp = -np.ones([3, T, R, K, M])

        # create dictionary for future indexing
        _pe = self._paramsEval
        catIds = [1]  # _pe.catIds if _pe.useCats else [-1]
        setK = set(catIds)
        setM = set(_pe.maxDets)
        setI = set(_pe.imgIds)
        # get inds to evaluate
        k_list = [n for n, k in enumerate(p.catIds) if k in setK]

        m_list = [m for n, m in enumerate(p.maxDets) if m in setM]
        i_list = [n for n, i in enumerate(p.imgIds) if i in setI]
        I0 = len(_pe.imgIds)

        error_freqs = np.zeros([5, T, R, len(k_list), len(m_list)])
        error_cat_h = np.zeros([5, T, R, len(k_list), len(m_list)])
        conf_thrs_fppi = np.zeros([T, len(k_list), len(m_list), 9])
        fp_mrs = np.zeros([3, T, len(k_list), len(m_list), 9])

        # retrieve E at each category, area range, and max number of detections
        for k, k0 in enumerate(k_list):
            Nk = k0 * I0
            for m, maxDet in enumerate(m_list):
                E = [self.evalImgs[Nk + i] for i in i_list]
                E = [e for e in E if not e is None]
                if len(E) == 0:
                    continue

                dtScores = np.concatenate([e['dtScores'][0:maxDet] for e in E])

                # different sorting method generates slightly different results.
                # mergesort is used to be consistent as Matlab implementation.

                inds = np.argsort(-dtScores, kind='mergesort')
                dtScores = dtScores[inds]

                dtm = np.concatenate([e['dtMatches'][:, 0:maxDet] for e in E], axis=1)[:, inds]
                dtIg = np.concatenate([e['dtIgnore'][:, 0:maxDet] for e in E], axis=1)[:, inds]
                gtIg = np.concatenate([e['gtIgnore'] for e in E])
                n_gtIg = np.sum(gtIg)
                n_gt = gtIg.size
                n_crowd_gt = np.sum([e['numCrowdOcclusionGT'] for e in E])
                n_env_gt = np.sum([e['numEnvOcclusionGT'] for e in E])
                n_foreground_gt = np.sum([e['numForegroundGT'] for e in E])
                n_mixed_gt = np.sum([e['numMixedOcclusionGT'] for e in E])
                n_other_gt = np.sum([e['numBackgroundGT'] for e in E])
                assert n_other_gt >= 0

                # here we extract all error class arrays
                crowd_occ_dts = np.concatenate(
                    [e['crowdOcclusionDts'][:, 0:maxDet] for e in E], axis=1
                )[:, inds]
                env_occ_dts = np.concatenate(
                    [e['envOcclusionDts'][:, 0:maxDet] for e in E], axis=1
                )[:, inds]
                foreground_dts = np.concatenate(
                    [e['foregroundDts'][:, 0:maxDet] for e in E], axis=1
                )[:, inds]
                loc_dts = np.concatenate(
                    [e['localizationDts'][:, 0:maxDet] for e in E], axis=1
                )[:, inds]
                ghost_detection_dts = np.concatenate(
                    [e['ghostDetectionDts'][:, 0:maxDet] for e in E], axis=1
                )[:, inds]
                scaling_dts = np.concatenate(
                    [e['scalingDts'][:, 0:maxDet] for e in E], axis=1
                )[:, inds]
                background_dts = np.concatenate(
                    [e['backgroundDts'][:, 0:maxDet] for e in E], axis=1
                )[:, inds]
                amb_occ_dts = np.concatenate(
                    [e['mixedOcclusionDts'][:, 0:maxDet] for e in E], axis=1
                )[:, inds]

                npig = np.count_nonzero(gtIg == 0)
                if npig == 0:
                    continue
                tps = np.logical_and(dtm, np.logical_not(dtIg))
                fps = np.logical_and(np.logical_not(dtm), np.logical_not(dtIg))
                inds = np.where(dtIg == 0)[1]
                tps = tps[:, inds]
                fps = fps[:, inds]
                dtScores = dtScores[inds]
                crowd_occ_dts = crowd_occ_dts[:, inds]
                env_occ_dts = env_occ_dts[:, inds]
                foreground_dts = foreground_dts[:, inds]
                background_dts = background_dts[:, inds]
                loc_dts = loc_dts[:, inds]
                ghost_detection_dts = ghost_detection_dts[:, inds]
                scaling_dts = scaling_dts[:, inds]
                amb_occ_dts = amb_occ_dts[:, inds]

                # _ = np.cumsum(np.squeeze(foreground_dts))
                tp_sum = np.cumsum(tps, axis=1).astype(dtype=np.float)
                fp_sum = np.cumsum(fps, axis=1).astype(dtype=np.float)
                tp_cumsums_errcat = np.stack([
                    np.cumsum(crowd_occ_dts, axis=1),
                    np.cumsum(env_occ_dts, axis=1),
                    np.cumsum(foreground_dts, axis=1),
                    np.cumsum(background_dts, axis=1),
                    np.cumsum(amb_occ_dts, axis=1),
                ])
                error_cumsums_fp = np.stack([
                    np.cumsum(loc_dts, axis=1),
                    np.cumsum(ghost_detection_dts, axis=1),
                    np.cumsum(scaling_dts, axis=1)
                ])
                tp_cumsum_clear = np.cumsum(foreground_dts + background_dts, axis=1)
                assert np.isin(
                    np.all(crowd_occ_dts + env_occ_dts + foreground_dts + background_dts + amb_occ_dts),
                    [0, 1]
                )
                # assert np.all(tp_cumsums_errcat <= np.array([n_crowd_gt, n_env_gt, n_foreground_gt,
                #                                              n_other_gt, n_mixed_gt])[:, None, None])

                # this loop iterates over iouThresholds
                for t, (tp, fp) in enumerate(zip(tp_sum, fp_sum)):
                    # np.testing.assert_array_equal(np.squeeze(np.sum(tp_cumsums_errcat, axis=0)), tp)
                    tp = np.array(tp)
                    fppi = np.array(fp) / I0
                    nd = len(tp)
                    recall = tp / npig
                    # 1st axis: [all, crowd, env, foreground, other, mixed, loc, ghost, scaling]
                    # cat_precision = np.squeeze(np.stack([
                    #     tp_cumsums_errcat[0] / (fp + tp_cumsums_errcat[0] + np.spacing(1)),
                    #     tp_cumsums_errcat[1] / (fp + tp_cumsums_errcat[1] + np.spacing(1)),
                    #     tp_cumsums_errcat[2] / (fp + tp_cumsums_errcat[2] + np.spacing(1)),
                    #     tp_cumsums_errcat[3] / (fp + tp_cumsums_errcat[3] + np.spacing(1)),
                    #     tp_cumsums_errcat[4] / (fp + tp_cumsums_errcat[4] + np.spacing(1)),
                    #     (tp / (fp + tp + np.spacing(1)))[None, :],
                    #     (tp / (error_cumsums_fp[0] + tp + np.spacing(1))),
                    #     (tp / (error_cumsums_fp[1] + tp + np.spacing(1))),
                    #     (tp / (error_cumsums_fp[2] + tp + np.spacing(1)))
                    # ], axis=0))
                    q = np.zeros((R,))

                    # numpy is slow without cython optimization for accessing elements
                    # use python array gets significant speed improvement
                    # recall = recall.tolist()
                    q = q.tolist()
                    for i in range(nd - 1, 0, -1):
                        if recall[i] < recall[i - 1]:
                            recall[i - 1] = recall[i]
                        # # flatten the precision-recall curve
                        # for j in range(len(cat_precision)):
                        #     if cat_precision[j, i] > cat_precision[j, i - 1]:
                        #         cat_precision[j, i - 1] = cat_precision[j, i]
                    mr = 1 - recall

                    # Sorting for fppi between 10^-2 and 10^0
                    inds = np.searchsorted(fppi, p.fppiThrs, side='right') - 1
                    inds_fp = np.zeros([3, 9])
                    for i in range(3):
                        inds_fp[i] = np.searchsorted(np.squeeze(error_cumsums_fp[i] / I0), p.fppiThrs, side='right') - 1
                    inds_fp = inds_fp.astype(int)
                    inds_rec = np.searchsorted(recall, p.recThrs, side='right') - 1

                    # try:
                    for ri, pi in enumerate(inds):
                        q[ri] = recall[pi]
                        sampled_mr_fp[0, t, ri, k, m] = recall[inds_fp[0, ri]]
                        sampled_mr_fp[1, t, ri, k, m] = recall[inds_fp[1, ri]]
                        sampled_mr_fp[2, t, ri, k, m] = recall[inds_fp[2, ri]]
                    # except:
                    #     pass
                    # sampled_precision[:, t, :, k, m] = cat_precision[:, inds_rec]
                    sampled_recall[t, :, k, m] = recall[inds_rec]
                    conf_thrs_fppi[t, k, m, :] = dtScores[inds]

                    # compute frequencies of errors relative to number of regarded detections, fixed for 0
                    ys[t, :, k, m] = np.array(q)

                    if self.normalization == "total":
                        # npig as normalization constant for all
                        error_freqs[:, t, :, k, m] = 1 - tp_cumsums_errcat[:, t, inds] / npig
                    else:
                        # normalize by number of GTs with that error class, FPs still get noirmalize dby npig
                        norm = np.array([n_crowd_gt, n_env_gt, n_foreground_gt, n_other_gt, n_mixed_gt])
                        error_freqs[:, t, :, k, m] = 1 - tp_cumsums_errcat[:, t, inds] / norm[:, None]
                        error_cat_h[:, t, :, k, m] = 1 - tp_cumsums_errcat[:, t, inds_fp[1, :]] / norm[:, None]
                        tp_mrs_clear = 1 - tp_cumsum_clear[t, inds_fp[1]] / (n_foreground_gt + n_other_gt)
                        lamr_check = 1 - np.sum(tp_cumsums_errcat[:, t, inds], axis=0) / npig

                    if verbose:
                        print("INFO:")
                        print(f"n_crowd_gt={n_crowd_gt}")
                        print(f"n_env_gt={n_env_gt}")
                        print(f"n_foreground_gt={n_foreground_gt}")
                        print(f"n_background_gt={n_other_gt}")
                        print(f"n_mixed_gt={n_mixed_gt}")
                        assert npig == n_crowd_gt + n_env_gt + n_foreground_gt + n_other_gt + n_mixed_gt
                        print(f"n_gt={npig}")

                    # _tp_cumsums_errcat = tp_cumsums_errcat[:, t, :] / norm[:, None]
                    mr_errcat = 1.0 - tp_cumsums_errcat[:, t, :] / norm[:, None]
                    minmr_idx = np.argmin(mr)
                    
        if self.output is not None and self.output_json_path is not None:
            self.output_json_path += "___" + self.output['meta']['setup']
            with open(self.output_json_path, "w+") as fp:
                json.dump(self.output, fp, indent=4, sort_keys=True)

        # assert cat_precision.shape[-1] == recall.shape[-1]
        recall_errcat = np.max(np.squeeze(tp_cumsums_errcat) / norm[:, None], axis=1)
        
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            fp_ratio = np.nan_to_num(error_cumsums_fp[:, 0, :] / (fppi * I0))
        
        self.eval = {
            'params': p,
            'counts': [T, R, K, M],
            'date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'TP': ys,
            'fppi': fppi.tolist(),
            'mr': mr.tolist(),
            'error_freqs': error_freqs,
            'error_cat_h': error_cat_h,
            # 'confThrs': conf_thrs_fppi,
            'lamr_check': lamr_check,
            'error_map': 1 - tp_cumsums_errcat / norm[:, None, None],
            'mr_errcat': mr_errcat,
            'recall_errcat': recall_errcat,
            'max_fppi': np.max(fppi),
            # 'cat_precision': cat_precision,
            'recall': recall,
            # 'sampled_precision': sampled_precision,
            'sampled_recall': sampled_recall,
            'error_cumsums_fp': error_cumsums_fp,
            'sampled_mr_fp': sampled_mr_fp,
            'class_fppi_minmr': error_cumsums_fp[:, 0, minmr_idx] / I0,
            'dt_scores': dtScores,
            'tp_mrs_clear': tp_mrs_clear,
            'fp_ratio': fp_ratio,
            'counts_error_categories': {'F': n_foreground_gt, 'B': n_other_gt, 'C': n_crowd_gt, 'E': n_env_gt,
                                        'A': n_mixed_gt, 'total': n_gt, 'I': n_gtIg},
        }

        assert n_gt == n_foreground_gt+n_other_gt+n_crowd_gt+n_env_gt+n_mixed_gt+n_gtIg
        toc = time.time()
        # print('DONE (t={:0.2f}s).'.format( toc-tic))

    def summarize(self, id_setup, verbose):
        '''
        Compute and display summary metrics for evaluation results.
        Note this functin can *only* be applied on the default parameter setting
        '''
        def _summarize(iouThr=None, maxDets=100, verbose=False):
            p = self.params
            iStr = ' {:<18} {} @ {:<18} [ IoU={:<9} | height={:>6s} | visibility={:>6s} ] = {:0.2f}%'
            titleStr = 'Average Miss Rate'
            typeStr = '(COCO)'
            setupStr = p.SetupLbl[id_setup]
            iouStr = '{:0.2f}:{:0.2f}'.format(p.iouThrs[0], p.iouThrs[-1]) \
                if iouThr is None else '{:0.2f}'.format(iouThr)
            heightStr = '[{:0.0f}:{:0.0f}]'.format(p.HtRng[id_setup][0], p.HtRng[id_setup][1])
            occlStr = '[{:0.2f}:{:0.2f}]'.format(p.VisRng[id_setup][0], p.VisRng[id_setup][1])

            mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]

            # dimension of precision: [TxRxKxAxM]
            s = self.eval['TP']
            e = self.eval['error_freqs']
            e_h = self.eval['error_cat_h']
            # _e = np.squeeze(e)
            la = self.eval['lamr_check']
            fp_freqs_mr = self.eval['sampled_mr_fp']


            # IoU
            if iouThr is not None:
                t = np.where(iouThr == p.iouThrs)[0]
                s = s[t]
            mrs = 1-s[:,:,:,mind]
            fp_freqs_mr = 1 - fp_freqs_mr[..., mind]
            e = e[:, :, :, :, mind]
            e_h = e_h[:, :, :, :, mind]
            # ap = self.eval['sampled_precision']
            # ap = ap[:, 0, :, 0, 0]  # quickfix: ignore other categories
            # ap = np.mean(ap, axis=1)
            # assert len(ap.shape) == 1

            if len(mrs[mrs<2])==0:
                mean_s = -1
            else:
                mean_s = np.log(mrs[mrs<2])
                mean_s = np.mean(mean_s)
                mean_s = np.exp(mean_s)

            # if verbose: print(iStr.format(titleStr, typeStr,setupStr, iouStr, heightStr, occlStr, mean_s*100))
            # res_file.write(iStr.format(titleStr, typeStr,setupStr, iouStr, heightStr, occlStr, mean_s*100))
            # res_file.write(str(abs(mean_s) * 100))
            # res_file.write('\n')

            self.metrics = {}

            self.metrics["LAMR"] = mean_s

            la = np.exp(np.mean(np.log(la)))
            mean_e = np.exp(np.mean(np.log(e + 1e-6), axis=(1, 2, 3, 4)))
            mean_e_h = np.exp(np.mean(np.log(e_h + 1e-6), axis=(1, 2, 3, 4)))
            fp_mrs = np.exp(np.mean(np.log(fp_freqs_mr[:, 0, :, 0, 0]), axis=1))
            minmr_idx = np.argmin(self.eval['mr'])
            self.metrics["minMR"] = self.eval['mr'][minmr_idx]
            self.metrics["FPPI"] = self.eval['fppi'][minmr_idx]

            if verbose:
                print(iStr.format(titleStr, typeStr, setupStr, iouStr, heightStr, occlStr, mean_s * 100))
                print("Filtered Log-Average Miss Rates")

            assert len(mean_e) == 5
            assert len(fp_mrs) == 3
            for e_i, s in zip(mean_e, ['crowdOcclusion', 'envOcclusion', 'clearForeground', 'clearBackground',
                                       'ambiguousOcclusion', 'FB']):
                if verbose: print(f"{s}: {e_i:.5f}")
                self.metrics[f"FLAMR_{s}"] = e_i
            for e_i, s in zip(mean_e_h, ['crowdOcclusion', 'envOcclusion', 'clearForeground', 'clearBackground',
                                       'ambiguousOcclusion', 'FB']):
                if verbose: print(f"{s}: {e_i:.5f}")
                self.metrics[f"FLAMRH_{s}"] = e_i
            for e_i, s in zip(fp_mrs, ["localizationErrors", "ghostDetections", "scaleErrors"]):
                if verbose: print(f"{s}: {e_i:.5f}")
                self.metrics[f"FLAMR_{s}"] = e_i

            # if verbose: print("Category-aware Average Precision:")
            # for e_i, s in zip(ap, ["Overall", 'Crowd Occlusion', 'Environmental Occlusion',
            #                        'Clear Foreground', 'Clear Background', 'Ambiguous Occlusion',
            #                        "localizationErrors", "ghostDetections", "scaleErrors"]):
            #     if verbose: print(f"Precision-{s}: {e_i:.5f}")
            #     self.metrics[f"FAP_{s}"] = e_i

            if verbose: print("Category-aware FPPI @ minMR:")
            for e_i, s in zip(self.eval['class_fppi_minmr'],
                              ["localizationErrors", "ghostDetections", "scaleErrors"]):
                if verbose: print(f"{s}@minMR: {e_i:.5f}")
                self.metrics[f"CatFPPI_{s}"] = e_i

            if verbose:
                print("Other metrics:")
                print("FPPI Thrs -> Conf Thrs:")
                print(p.fppiThrs)
                print(np.squeeze(self.eval['confThrs']))
                print(f"LAMR-check value: {la}")
                # print(f"Clear FLAMR over Ghost Detections: {flamr_clear_ghost}")

            # res_file.write(iStr.format(titleStr, typeStr,setupStr, iouStr, heightStr, occlStr, mean_s*100))
            # res_file.write(str(abs(mean_s) * 100))
            # res_file.write('\n')
            return mean_s, mean_e

        if not self.eval:
            raise Exception('Please run accumulate() first')
        self.mean_s = _summarize(iouThr=.5, maxDets=1000, verbose=verbose)

    def __str__(self):
        self.summarize()

class Params:
    '''
    Params for coco evaluation api
    '''
    def setDetParams(self, thresholds):
        self.imgIds = []
        self.catIds = []

        # np.arange causes trouble.  the data point on arange is slightly larger than the true value
        # np. linspace causes also trouble with numpy version 1.19.2 - num needs to be int!
        num = int(np.round((1.00 - .0) / .01) + 1)  # 101 Stützstellen
        self.recThrs = np.linspace(start=.0, stop=1.00,
                                   num=int(np.round((1.00 - .0) / .01) + 1),
                                   endpoint=True)
        self.fppiThrs = np.array([0.0100, 0.0178, 0.0316, 0.0562, 0.1000, 0.1778, 0.3162, 0.5623, 1.0000])

        # print('Use inverse ffpiThrs for Params in eval_MR_multisetup_extended!!!!!')
        # self.fppiThrs = (1.0 - self.fppiThrs)[::-1]

        self.maxDets = [1000]
        self.expFilter = 1.25
        self.useCats = 1

        self.iouThrs = np.array([0.5])  # np.linspace(.5, 0.95, np.round((0.95 - .5) / .05) + 1, endpoint=True)

        self.HtRng = [
            [50, 1e5 ** 2],
            [50, 1e5 ** 2],
        ]
        self.VisRng = [
            [0.0, 1e5 ** 2],
            [0.65, 1e5 ** 2],
        ]
        self.SetupLbl = [
            'All',
            'Reasonable',
        ]

        self.ambFactor = 0.75

        # # Original: Benedikts masterthesis
        # self.thresholds = edict({
        #     'envPixelThrs': 0.5,
        #     'occPixelThrs': 0.45,
        #     'crowdPixelThrs': 0.35,
        #     'iouMatchThrs': 0.5,
        #     'foregroundThrs': 200,
        #     'centerAlignedThreshold': 0.2,
        #     'reducedIouThreshold': 0.25,
        # })
        if thresholds is None:
            self.thresholds = edict({
                'envPixelThrs': 0.7,
                'occPixelThrs': 0.6,
                'crowdPixelThrs': 0.5,
                'iouMatchThrs': 0.5,
                'foregroundThrs': 190,  # from evaluator/COCO/tools/enrich_ground_truth/calc_foreground_height.py
                'centerAlignedThreshold': 0.2,
                'reducedIouThreshold': 0.25,
            })
        else:
            self.thresholds = thresholds

        self.force_foreground_match = True
        self.force_background_match = True

        self.normalization = 'class'

    def __init__(self, iouType='segm', thresholds=None):
        if iouType == 'segm' or iouType == 'bbox':
            self.setDetParams(thresholds)
        else:
            raise Exception('iouType not supported')
        self.iouType = iouType
        # useSegm is deprecated
        self.useSegm = None
