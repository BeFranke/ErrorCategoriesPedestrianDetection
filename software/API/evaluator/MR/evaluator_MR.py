import json
import os
import pickle
import warnings
from copy import deepcopy
import ast

import numpy as np
from easydict import EasyDict
from matplotlib import pyplot as plt

from benedikt.utils import function_attribute
from ..MR.coco import COCO
from ..MR.eval_MR_multisetup import COCOeval
from ..base_evaluator import BaseEvaluator

PLOT_OUTPUT_PATH = os.path.join(
    os.path.dirname(__file__),
    "..", "..", "..", "benedikt", "error_class_utils", "plotting-raw"
)


class EvaluatorMR(BaseEvaluator):
    """
    This class is responsible for generating the COCO evaluation metrics. To
    generate the mAP results we are currently using the COCO API provided in the
    pycocotools library. Later we want to build a similar library by ourselves
    to mitigate some issues with the library. Furthermore, this class will plot
    some graphs for better understanding of the evaluation metrics.
    """

    def __init__(self, config, device, dataset, mode, plotter=None) -> None:
        super().__init__(dataset=dataset, config=config, device=device, mode=mode, plotter=plotter)
        self.make_dirs(dirs=['mr'])

        self._root = './evaluator/MR'
        self.annFile = '{}_gt.json'.format(self.dataset.image_set)

        self.resFile = '{}_dt'.format(self.dataset.image_set)
        # self.resFile = 'mini_dt.json'

        self.evalFile = 'results.txt'
        if os.path.exists(os.path.join(self._root, self.evalFile)):
            os.remove(os.path.join(self._root, self.evalFile))

        self.annType = 'bbox'
        # self.id_setup = [0, 1, 2, 3]       # Reasonable, Large, ...
        self.id_setup = self.config.setting_id
        if not isinstance(self.id_setup, list):
            self.id_setup = ast.literal_eval(self.id_setup)  # convert string repr of list to list

        # self.plotId = 3
        self.plotId = 1

        # Make new GT Anno if ImageSet is not 'val'
        if self.config.dataset == 'kia' or (self.config.dataset == 'cityscapes'
                # and self.dataset.image_set != 'val'
        ):
            self.gt_dataset = deepcopy(self.dataset)
            # self.gt_dataset.extract_reasonable_height_and_width(height=50, width=None)
            self._generate_gt_json(dataset=self.gt_dataset)

        # sanity check so I don't shoot myself in the foot
        if config.generate_dataset:
            assert config.model == 'csp_1', "generate_dataset is True, but model is not CSP_1!"

    def _generate_det_json(self, detections, iteration, cache="./cache") -> None:
        """
        This method generates an annotation file which contains all the found
        detections. The annotations meet the COCO annotation dict structure.
        The annotations are saved as a .json-file in the checkpoints folder.

        :param detections: Absolute output of the detector.
        """

        det_list = []
        for i in range(0, len(detections)):
            det = detections[i]['det']

            if det.shape[0] > 0:
                for j in range(0, det.shape[0]):
                    _dict = EasyDict()

                    x1 = round(float(det[j, 0]), 2)
                    y1 = round(float(det[j, 1]), 2)
                    w = round(float(det[j, 2] - det[j, 0]), 2)
                    h = round(float(det[j, 3] - det[j, 1]), 2)

                    # PREDEFINED KEYS
                    #   - IMAGE_IDS has to start with 1, not 0!!!!!
                    _dict.image_id = detections[i]['image_id']
                    _dict.category_id = 1  # All pedestrians
                    _dict.bbox = [x1, y1, w, h]
                    _dict.score = round(float(det[j, 4]), 2)

                    det_list.append(_dict)

        # Generate the json file with all detections
        path = os.path.join(self.root, 'mr', '{}.json'.format(self.resFile))
        with open(path, 'w') as json_file:
            json.dump(det_list, json_file)

    def _generate_gt_json(self, dataset):
        annotations = []
        images = []
        categories = set()
        annotation_id = 1

        for img_index, image_id in enumerate(dataset.image_ids):
            for ann_index, annotation_class in enumerate(dataset.classes[img_index]):
                coords = dataset.coordinates[img_index][ann_index]
                width = int(abs(coords[2] - coords[0]))
                height = int(abs(coords[3] - coords[1]))

                new_ann = {}
                new_ann["id"] = annotation_id
                new_ann["image_id"] = img_index + 1
                new_ann["category_id"] = 1

                new_ann["height"] = height
                new_ann["iscrowd"] = 0
                new_ann["ignore"] = 1 - int(annotation_class == 1)
                new_ann["vis_bbox"] = dataset.vis_bbox[img_index][ann_index]
                new_ann["bbox"] = dataset.bbox[img_index][ann_index]
                new_ann["vis_ratio"] = round(dataset.vis_ratio[img_index][ann_index], ndigits=12)
                if new_ann["vis_ratio"] == 1.0 or new_ann["vis_ratio"] == 0.0:
                    new_ann["vis_ratio"] = int(new_ann["vis_ratio"])

                new_ann["instance_id"] = dataset.instance_ids[img_index][ann_index]

                annotations.append(new_ann)
                annotation_id += 1

                categories.add(int(annotation_class))

            new_img = {}
            new_img["id"] = img_index + 1
            new_img["im_name"] = os.path.split(image_id)[-1]
            new_img["width"] = int(dataset.img_widths[img_index])
            new_img["height"] = int(dataset.img_heights[img_index])

            images.append(new_img)

        final_json = {}
        final_json["categories"] = [{"id": cat_id, "name": dataset.class_names[cat_id]} for cat_id in categories]
        final_json["images"] = images
        final_json["annotations"] = annotations

        path = os.path.join(self._root, self.annFile)
        with open(path, 'w') as json_file:
            json.dump(final_json, json_file, indent=4, sort_keys=True)

        print('CREATED COCO GROUNDTRUTH .json!')

    def plot_detections(self, dataset, detections, iteration, scale=1.0):

        if self.plotId is not None:
            plotId = self.plotId
            # for plotId in range(len(dataset)):
            _, target = dataset.__getitem__(index=plotId)

            image = target['image']
            classes = target['classes']
            coords = target['coords']

            fig, ax = self.get_subplots(num_subplots=1, img_res_h_w=(image.height, image.width))
            ax.imshow(image)
            ax.axis('off')

            self.plot_gt_with_height(classes=classes.cpu().numpy(),
                                     coords=coords.cpu().numpy(),
                                     ax=ax)

            if detections[plotId]['det'].shape[0] > 0:
                self.plot_output(output=detections[plotId]['det'], ax=ax)

            image_path = target['image_path']
            img_name = os.path.split(image_path)[-1][:-4]
            fig_name = '{}_{}'.format(img_name, iteration)

            if self.save:
                self.saveFig(fig=fig, file='mr', name='inf_' + fig_name, iteration=iteration)

            figures = {'Inf': fig}
            self.logging_figures(figures=figures, iteration=iteration, tag='MR/')

    def plot_mr_fppi(self, cocoEval, lamr, iteration, fig=None, ax=None):
        '''
        :param s: [num_iouThrs, num_fppiThrs, num_cat (2), maxDets]
        '''

        # Plot MissRate over FPPI
        if fig is None:
            fig, ax = self.get_subplots(num_subplots=1, img_res_h_w=(1000, 1000), margin=80, sharey="all")
            ax.set_title('log-average Miss Rate: {:.2f}%'.format(100 * lamr))
        ax.plot(cocoEval.eval['fppi'], cocoEval.eval['mr'], label=self.config.model)
        np.save(os.path.join(PLOT_OUTPUT_PATH, f"{self.config.model}__fppi__{self.config.setting_id}.npy"),
                cocoEval.eval["fppi"])
        np.save(os.path.join(PLOT_OUTPUT_PATH, f"{self.config.model}__mr__{self.config.setting_id}.npy"),
                cocoEval.eval["mr"])
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_yticks([1, 0.1, 0.01])
        ax.set_xticks([0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0])
        ax.set_xlabel("FPPI")
        ax.set_ylabel("Miss Rate")
        ax.grid(b=True, which='major', axis='x', linestyle='-', linewidth=1)
        ax.grid(b=True, which='major', axis='y', linestyle='-', linewidth=1)
        ax.grid(b=True, which='minor', axis='y', linestyle='--', linewidth=1)

        if self.save:
            self.saveFig(fig=fig, file='mr', name='mr_fppi', iteration=iteration)

        figures = {'MR': fig}
        self.logging_figures(figures=figures, iteration=iteration, tag='MR/')
        # plt.legend()

        # Dump MissRate and FPPI
        _dict = {'mr': cocoEval.eval['mr'],
                 'fppi': cocoEval.eval['fppi'],
                 'lamr': lamr}
        path = os.path.join(self.root, 'mr', 'mr_fppi_{}_{}.json'.format(iteration, self.dataset.image_set))
        with open(path, 'w') as json_file:
            json.dump(_dict, json_file)

    def dumpEvalImgs(self, cocoEval, iteration):
        # Dump evalImgs
        path = os.path.join(self.root, 'mr', 'evalImgs_{}_{}.json'.format(self.dataset.image_set, iteration))
        _evalImgs = []
        for evalImg in cocoEval.evalImgs:
            if evalImg is not None:
                for k, v in evalImg.items():
                    if isinstance(v, np.ndarray):
                        evalImg[k] = v.reshape(-1).astype(np.int).tolist()
                    if isinstance(v, np.integer):
                        evalImg[k] = int(v)

                _evalImgs.append(evalImg)

        evalImgs = []
        # Add Boxes for Detection and Groundtruth
        keys = [('dtBoxes', 'dtIds'), ('gtBoxes', 'gtIds')]

        for evalImg in _evalImgs:
            imgId = evalImg['image_id']

            for key in keys:
                evalImg[key[0]] = []
                anns = cocoEval.cocoDt.anns if key[0] == 'dtBoxes' else cocoEval.cocoGt.anns

                for Id in evalImg[key[1]]:
                    assert imgId == anns[Id]['image_id'], 'ImageId missmatch'

                    box = anns[Id]['bbox']
                    evalImg[key[0]].append(box)

            evalImgs.append(evalImg)

        with open(path, 'w') as json_file:
            json.dump(cocoEval.evalImgs, json_file)

    def evaluate(self, model, iteration=None, figlist=None, axlist=None) -> None:
        """
        Generates the COCO evaluation metrics using the official COCO API.
        :param model: The model that should be evaluated.
        """
        detections = self.load_cache() if self.config.cache else False
        if not detections:
            detections = self.get_detections(model)
            self.cache_detections(detections)

        # Save the json file with the detections to the checkpoints folder
        if self.num_detections > 0:

            if self.dataset.image_set != 'val':
                self.plot_detections(dataset=self.dataset, detections=deepcopy(detections), iteration=iteration)

            self._generate_det_json(detections=detections, iteration=iteration)

            for id in self.id_setup:
                cocoGt = COCO(annotation_file=os.path.join(self._root, self.annFile))
                cocoDt = cocoGt.loadRes(resFile=os.path.join(self.root, 'mr', '{}.json'.format(self.resFile)))
                imgIds = sorted(cocoGt.getImgIds())
                if self.config.image_set["val"] == "val":
                    output = {
                        'meta': {
                            'model': self.config.model,
                            'dataset': self.config.dataset,
                            'split': self.config.image_set['val'],
                        }
                    }
                    for key in filter(lambda x: 'root' in x, self.config.keys()):
                        output['meta'][key] = self.config[key]
                else:
                    output = None

                cocoEval = COCOeval(cocoGt, cocoDt, self.annType,
                                    env_pixel_thrs=float(self.config.thresholds.envPixelThrs),
                                    occ_pixel_thr=float(self.config.thresholds.occPixelThrs),
                                    crowd_pixel_thrs=float(self.config.thresholds.crowdPixelThrs),
                                    iou_match_thrs=float(self.config.thresholds.iouMatchThrs),
                                    foreground_thrs=float(self.config.thresholds.foregroundThrs),
                                    high_confidence_thrs=float(self.config.thresholds.highConfidenceThrs),
                                    maa_thres=float(self.config.maa_thres),
                                    generate_dataset=self.config.generate_dataset,
                                    split=self.config.image_set["val"], output=output,
                                    output_filename=self.config.model + "___" + str(
                                        self.config.thresholds.iouMatchThrs))

                # get inst-seg file names
                inst_seg_files = map(lambda k, v: {v['id']: v['im_name']}, cocoGt.imgs.items())

                cocoEval.params.imgIds = imgIds
                cocoEval.evaluate(id)
                cocoEval.accumulate()
                cocoEval.summarize(id_setup=id)

                # Reasonable (or error_eval)
                if True:  # id == 0 :
                    # Save LAMR in results.txt
                    with open(os.path.join(self._root, self.evalFile), "a") as results:
                        results.write('{}\n'.format(abs(cocoEval.mean_s) * 100))

                    if int(abs(cocoEval.mean_s) * 100) < 100:
                        self.dumpEvalImgs(cocoEval=cocoEval, iteration=iteration)

                        metrics = {'MR': cocoEval.mean_s,
                                   'minMR': min(cocoEval.eval['mr']),
                                   'MinFPPI': max(cocoEval.eval['fppi'])}
                        self.logging_metrics(metrics=metrics,
                                             iteration=iteration,
                                             type='Metric/MR')

                        # Plot and dump MissRate over FPPI
                        self.plot_mr_fppi(cocoEval=cocoEval,
                                          lamr=cocoEval.mean_s,
                                          iteration=iteration)

                        self.plot_err_fppi(cocoEval, iteration=iteration)

                    else:
                        print('No MREval due to no reasonable Detection - Iter: {}'.format(iteration))

        else:
            print('No MREval due to no Detections - Iter: {}'.format(iteration))

    def plot_err_fppi(self, cocoEval, iteration):
        # save some things for later (accumulated) plotting
        np.save(os.path.join(PLOT_OUTPUT_PATH, f"{self.config.model}__fppi__{self.config.setting_id}.npy"),
                cocoEval.eval["fppi"])
        np.save(os.path.join(PLOT_OUTPUT_PATH, f"{self.config.model}__scores__{self.config.setting_id}.npy"),
                cocoEval.eval["dt_scores"])
        for i, (err_c, s) in enumerate(zip(cocoEval.eval['fp_ratio'], ['Poor Localization', 'Ghost Detections',
                                                                       'Scaling Errors'])):
            np.save(os.path.join(PLOT_OUTPUT_PATH, f"{self.config.model}__fp_ratio_{s.replace(' ', '')}__"
                                                   f"{self.config.setting_id}.npy"),
                    err_c)

        for i, (err_c, s) in enumerate(zip(cocoEval.eval['error_cumsums_fp'], ['Poor Localization', 'Ghost Detections',
                                                                               'Scaling Errors'])):
            np.save(os.path.join(PLOT_OUTPUT_PATH, f"{self.config.model}__fp_counts_{s.replace(' ', '')}__"
                                                   f"{self.config.setting_id}.npy"),
                    err_c)

        # Plot error count over FPPI
        for i, (err_c, s) in enumerate(zip(cocoEval.eval['error_map'],
                                           ['Crowd Occlusion Errors', 'Environmental Occlusion Errors',
                                            'Foreground Errors', 'Standard Errors', 'Mixed Occlusion Errors'])):
            np.save(os.path.join(PLOT_OUTPUT_PATH, f"{self.config.model}__{s.replace(' ', '')}__"
                                                   f"{self.config.setting_id}.npy"),
                    err_c)
            fig, ax = self.get_subplots(num_subplots=1, img_res_h_w=(1000, 1000), margin=80)
            ax.plot(cocoEval.eval['fppi'], np.squeeze(err_c), label=self.config.model)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_yticks([1, 0.1, 0.01])
            ax.set_xticks([0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0])
            ax.set_xlabel("FPPI")
            ax.set_ylabel("Error Frequency")
            ax.set_title(s)
            ax.grid(b=True, which='major', axis='x', linestyle='-', linewidth=1)
            ax.grid(b=True, which='major', axis='y', linestyle='-', linewidth=1)
            ax.grid(b=True, which='minor', axis='y', linestyle='--', linewidth=1)
            # plt.legend()

        np.save(os.path.join(PLOT_OUTPUT_PATH, f"{self.config.model}__recall__{self.config.setting_id}.npy"),
                cocoEval.eval['recall'])
        # Plot ratio of FP classes over FPPI
        for j, (err_c, s) in enumerate(
                zip(cocoEval.eval['cat_precision'], ['All', 'Crowd Occlusion Errors', 'Environmental Occlusion Errors',
                                                     'Foreground Errors', 'Standard Errors', 'Mixed Occlusion Errors',
                                                     'Poor Localization', 'Ghost Detections',
                                                     'Scaling Errors'])):
            np.save(os.path.join(PLOT_OUTPUT_PATH, f"{self.config.model}__precision_{s.replace(' ', '')}__"
                                                   f"{self.config.setting_id}.npy"),
                    err_c)
            fig, ax = self.get_subplots(num_subplots=1, img_res_h_w=(1000, 1000), margin=80)

            ax.plot(cocoEval.eval['recall'], np.squeeze(err_c))
            # ax.set_xscale("log")
            # ax.set_yscale("log")
            # ax.set_yticks([1, 0.1, 0.01])
            # ax.set_xticks([0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0])
            ax.set_xlabel("Recall")
            ax.set_ylabel("Category-aware Precision" if s != "All" else "Precision")
            ax.set_title(s if s != "All" else f"Precision-Recall Curve")
            ax.grid(b=True, which='major', axis='x', linestyle='-', linewidth=1)
            ax.grid(b=True, which='major', axis='y', linestyle='-', linewidth=1)
            ax.grid(b=True, which='minor', axis='y', linestyle='--', linewidth=1)
            # plt.legend()

        if self.save:
            self.saveFig(fig=fig, file='mr', name='mr_fppi', iteration=iteration)

        figures = {f'error_counts': fig}
        self.logging_figures(figures=figures, iteration=iteration, tag='error_counts/')

    @function_attribute("CACHE_PATH", os.path.join(os.path.dirname(__file__), "cache"))
    def load_cache(self):
        model = self.config.model
        if not os.path.isdir(self.load_cache.CACHE_PATH):
            return False
        else:
            files = os.listdir(self.load_cache.CACHE_PATH)
            # TODO add some sort of check by hash or file modification data
            fname = f"{model}_det.pck"
            if fname in files:
                warnings.warn(
                    "CACHE is enabled and was hit! Please manually ensure that cached inference is up-to-date!")
                with open(os.path.join(self.load_cache.CACHE_PATH, fname), 'rb') as fp:
                    detections = pickle.load(fp)

                self.num_detections = sum(d['det'].shape[0] for d in detections)
                print("cache hit!")
                return detections
            else:
                return False

    def cache_detections(self, detections):
        model = self.config.model
        if not os.path.isdir(self.load_cache.CACHE_PATH):
            os.mkdir(self.load_cache.CACHE_PATH)

        fname = f"{model}_det.pck"
        with open(os.path.join(self.load_cache.CACHE_PATH, fname), 'wb+') as fp:
            pickle.dump(detections, fp)
