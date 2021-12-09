import time
import json
import torch
import os

from abc import ABC
from copy import deepcopy
from easydict import EasyDict
from typing import List

import torch.utils.data as data
import pandas as pd

import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms.functional as FT
from tqdm import tqdm

from patrick.inferencer.utils import *


class BaseEvaluator(ABC):
    """
    The 'Evaluator' class serves as a base class for future evaluators to make
    sure that the training scripts can be reused for any other dataset and
    evaluator. By hiding the COCOEvaluator and VOCEvaluator classes behind the
    Evaluator class we can make sure that both classes implement the same
    methods that are required for running the training script.

    To create an Evaluator please check out the EvaluatorBuilder class. This
    class will allow you to parametrize your required evaluator and build your
    desired Evaluator.
    """

    def __init__(self, config, device, dataset, mode=None, plotter=None) -> None:

        self.config = config
        self.device = device
        self.dataset = dataset
        self.mode = mode
        self.verbose = True if self.mode == 'offline' else False
        self.plotter = plotter

        # ConfThreshold Issue: https://github.com/liuwei16/CSP/issues/54
        # Adapted CSP also uses score=.1 in their Code!
        self.conf_threshold = 0.1           # threshold for final detections
        self.nms_threshold = 0.5            # threshold for IoU in NMS (higher -> more detections)

        # Make eval Directory
        if self.mode is not None:
            if self.mode == 'online':
                self.root = os.path.join(self.config.checkpoints_model_folder, 'eval')
            else:
                self.root = os.path.join(self.config.trained_model_path, 'eval')
            if not os.path.exists(self.root): os.mkdir(self.root)

            # Make 'online' or 'offline' Directory under Eval
            self.root = os.path.join(self.root, self.mode)
            if not os.path.exists(self.root): os.mkdir(self.root)

        # Data Loader
        self.batch_size = self.config.batch_size_eval
        self.data_loader = data.DataLoader(dataset=self.dataset,
                                           batch_size=self.batch_size,
                                           num_workers=0,
                                           collate_fn=self.dataset.collate_fn,
                                           shuffle=False)
        self.run_style_sheet()
        self.dpi = 100  # dots per inch

        if self.mode == 'offline':
            self.save = self.config.save
            self.plot = self.config.plot
        else:
            self.save = False
            self.plot = False
            # https://www.dkrz.de/up/help/faq/mistral/python-matplotlib-fails-with-qxcbconnection-could-not-connect-to-display
            if self.config.evaluation_step >= 100: mpl.use('Agg')

        self.distinct_colors = ['#e6194b', '#3cb44b', '#ffe119', '#0082c8', '#f58231',
                                '#911eb4', '#46f0f0', '#f032e6', '#d2f53c', '#fabebe',
                                '#008080', '#000080', '#aa6e28', '#fffac8', '#800000',
                                '#aaffc3', '#808000', '#ffd8b1', '#e6beff', '#808080',
                                '#FFFFFF', '#e6194b', '#3cb44b', '#ffe119', '#0082c8',
                                '#f58231', '#911eb4', '#46f0f0', '#f032e6', '#d2f53c',
                                '#fabebe', '#008080', '#000080', '#aa6e28', '#fffac8',
                                '#800000', '#aaffc3', '#808000', '#ffd8b1', '#e6beff',
                                '#808080', '#FFFFFF', '#e6194b', '#3cb44b', '#ffe119',
                                '#0082c8', '#f58231', '#911eb4', '#46f0f0', '#f032e6',
                                '#d2f53c', '#fabebe', '#008080', '#000080', '#aa6e28',
                                '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1',
                                '#e6beff', '#808080', '#FFFFFF', '#e6194b', '#3cb44b',
                                '#ffe119', '#0082c8', '#f58231', '#911eb4', '#46f0f0',
                                '#f032e6', '#d2f53c', '#fabebe', '#008080', '#000080',
                                '#aa6e28', '#fffac8', '#800000', '#aaffc3', '#808000',
                                '#ffd8b1', '#e6beff', '#808080', '#FFFFFF', '#e6194b',
                                '#3cb44b', '#ffe119', '#0082c8', '#f58231', '#911eb4',
                                '#46f0f0', '#f032e6', '#d2f53c', '#fabebe', '#008080',
                                '#000080', '#aa6e28', '#fffac8', '#800000', '#aaffc3',
                                '#808000', '#ffd8b1', '#e6beff', '#808080', '#FFFFFF',
                                '#e6194b', '#3cb44b', '#ffe119', '#0082c8', '#f58231',
                                '#911eb4', '#46f0f0', '#f032e6', '#d2f53c', '#fabebe',
                                '#008080', '#000080', '#aa6e28', '#fffac8', '#800000',
                                '#aaffc3', '#808000', '#ffd8b1', '#e6beff', '#808080',
                                '#FFFFFF', '#e6194b', '#3cb44b', '#ffe119', '#0082c8',
                                '#f58231', '#911eb4', '#46f0f0', '#f032e6', '#d2f53c',
                                '#fabebe', '#008080', '#000080', '#aa6e28', '#fffac8',
                                '#800000', '#aaffc3', '#808000', '#ffd8b1', '#e6beff',
                                '#808080', '#FFFFFF']

    def run_style_sheet(self):
        mpl.rcParams['axes.titlesize'] = 18
        mpl.rcParams['axes.labelsize'] = 14
        mpl.rcParams['xtick.labelsize'] = 14
        mpl.rcParams['ytick.labelsize'] = 14

        mpl.rcParams['legend.fontsize'] = 14
        mpl.rcParams['legend.loc'] = 'upper right'

        mpl.rcParams['font.family'] = 'serif'
        mpl.rcParams['font.sans-serif'] = 'Times New Roman'
        mpl.rcParams['lines.linewidth'] = 2
        mpl.rcParams['lines.markersize'] = 10

        self.FigSize = (400, 700)
        self.margin = 80
        self.spacing = 40

        self.markers = ['o', 'd', 'x', 'v']
        self.colors = dict(TP='green', FP='orange', FN='red', GT='blue', GTIG='magenta', DTIG='white')

    def make_dirs(self, dirs):
        for dir in dirs:
            if not os.path.exists(os.path.join(self.root, dir)):
                os.mkdir(os.path.join(self.root, dir))

    def get_detections(self, model) -> List:
        """
        This function should calculate the evaluation metrics for the specific
        dataset. To log information every n iterations you also need to pass in
        the current iteration of training. You need to inherit from this class
        to create your own specific 'Evaluator'. Furthermore, you should always
        add a super call to this function because it computes the detections for
        the model and the dataset at hand.

        :param model: The model that should be evaluated.
        :param iteration: The current iteration of your training loop.
        """

        detections = []

        if not self.conf_threshold:
            raise AttributeError("The confidence threshold cannot be None.")

        if not self.nms_threshold:
            raise AttributeError("The nms threshold cannot be None.")

        self.times = []
        self.num_detections = 0
        with torch.no_grad():
            for iter, loaded_data in tqdm(enumerate(self.data_loader, start=0),
                                          total=len(self.dataset) // self.batch_size):
                image,_ = loaded_data
                input = image.to(self.device)

                if isinstance(model, torch.nn.DataParallel):
                    start_time = time.time()
                    dts = model.module.inference(input, self.conf_threshold, self.nms_threshold)
                else:
                    start_time = time.time()
                    dts = model.inference(input, self.conf_threshold, self.nms_threshold)

                self.times.append(time.time() - start_time)

                for i in range(len(dts)):
                    # Don't save empty detections
                    detections.append({'image_id': iter * self.batch_size + i + 1,
                                       'det': dts[i],
                                       # 'posCells': posCells
                                       })

                    # Absolute Detections for Original Image Size!
                    self.num_detections += dts[i].shape[0]

        self.avg_time = sum(self.times) / len(self.times)
        self.fps = 1 / self.avg_time

        return detections

    def loadEvalImgs(self, iteration):
        evalImgsPath = os.path.join(self.root, 'mr', 'evalImgs_{}_{}.json'.format(self.dataset.image_set, iteration))

        if os.path.exists(evalImgsPath):
            with open(evalImgsPath, "r") as json_file:
                evalImgs = json.load(json_file)

            # Sort out None elements for class 0
            evalImgs = [evalImg for evalImg in evalImgs if evalImg is not None]

            return evalImgs

        else:
            return None

    def logging_metrics(self, metrics, iteration, type) -> None:

        print('----------------------------------------------------------------------------------------')
        for key, value in metrics.items():
            print('EVALUATION - {}: {:.4f}'.format(key.upper(), value))
        print('----------------------------------------------------------------------------------------')

        if self.plotter is not None:
            self.plotter.write_scalars(scalars_dict=metrics, iteration=iteration, type=type)

    def logging_figures(self, figures, iteration, tag) -> None:

        if self.plotter is not None:
            self.plotter.write_figures(figures=figures, iteration=iteration, tag=tag)

    def logging_hist(self, hists, iteration, tag) -> None:

        if self.plotter is not None:
            self.plotter.write_hists(hists=hists, iteration=iteration, tag=tag)

    @staticmethod
    def applyPosNegMask(model, coords, classes, tensor):
        """
        :param tensor: [num_prototypes, num_cells]
        """

        # Use original coords to calculate neg_mask
        if not isinstance(model, torch.nn.DataParallel):
            pos_mask, neg_mask = model.matching(coords=coords, classes=classes, mode='val')
        else:
            pos_mask, neg_mask = model.module.matching(coords=coords, classes=classes,
                                                       mode='val')  # [num_gts, num_cells]

        _pos_tensor = pos_mask[:, :, None] * tensor.permute(1, 0)[None, :, :]  # [_num_gts, num_cells, num_prototypes]
        pos_tensor = _pos_tensor.permute(2, 1, 0)  # [num_prototypes, num_cells, num_gts]

        _neg_tensor = neg_mask[:, :, None] * tensor.permute(1, 0)[None, :, :]  # [_num_gts, num_cells, num_prototypes]
        neg_tensor = _neg_tensor.permute(2, 1, 0)  # [num_prototypes, num_cells, num_gts]

        pos_tensor = pos_tensor.detach().cpu().numpy()
        neg_tensor = neg_tensor.detach().cpu().numpy()

        return pos_tensor, neg_tensor

    def get_subplots(self, num_subplots=None, nrows=None, ncols=None,
                     scale=1.0,
                     img_res_h_w=None,
                     margin=None,
                     flatten=True,
                     sharey="none"):

        if num_subplots is not None:
            assert nrows is None
            nrows = ncols = int(np.ceil(np.sqrt(num_subplots)))

        spacing = 0 if nrows < 2 and ncols < 2 else self.spacing
        margin = margin if margin is not None else self.margin
        img_res_h_w = img_res_h_w if img_res_h_w is not None else self.FigSize

        width = (ncols * img_res_h_w[1] + 2 * margin + (ncols-1) * spacing) / self.dpi * scale  # inches
        height = (nrows * img_res_h_w[0] + 2 * margin + (nrows-1) * (3 * spacing)) / self.dpi * scale
        left = margin / self.dpi / width  # axes ratio
        bottom = margin / self.dpi / height
        wspace = spacing / float(200)
        fig, axes = plt.subplots(nrows, ncols, figsize=(width, height), dpi=self.dpi, sharey=sharey)
        fig.subplots_adjust(left=left, bottom=bottom, right=1. - left, top=1. - bottom,
                            wspace=wspace, hspace=wspace)

        if num_subplots != 1 and flatten:
            axes = axes.flatten()

        return fig, axes

    def plot_image(self, image):
        imgResHW = (image.height, image.width)
        fig, ax = self.get_subplots(num_subplots=1, img_res_h_w=imgResHW, margin=0)
        ax.imshow(image)
        ax.axis('off')

        return fig, ax

    def plot_gt_with_height(self, classes, coords, ax):
        labels = np.concatenate((np.expand_dims(classes, 1), coords), axis=1)

        coords = labels[:, -4:]
        for i in range(coords.shape[0]):
            xmin = coords[i, 0]
            ymin = coords[i, 1]
            xmax = coords[i, 2]
            ymax = coords[i, 3]
            w = xmax - xmin
            h = ymax - ymin

            ax.add_patch(plt.Rectangle((xmin, ymin), w, h,
                                       color=self.distinct_colors[int(labels[i, 0])],
                                       fill=False, linewidth=1))
            # ax.text(xmin, ymax, '{:.2f}'.format(h),
            #         horizontalalignment='left', verticalalignment='bottom',
            #         fontsize='x-small', color='black',
            #         bbox=dict(boxstyle='square,pad=0',
            #                   color=self.distinct_colors[int(labels[i, 0])]))

    def plot_output(self, output, ax):
        coords = output[:, :4]
        for i in range(coords.shape[0]):
            xmin = coords[i, 0]
            ymin = coords[i, 1]
            xmax = coords[i, 2]
            ymax = coords[i, 3]
            w = xmax - xmin
            h = ymax - ymin

            # print(xmin, ymin, xmax, ymax)
            ax.add_patch(plt.Rectangle((xmin, ymin), w, h,
                                       color='white',
                                       fill=False, linewidth=1))
            # ax.text(xmin, ymax,
            #         # '{}-{:.2f}'.format(int(h), output[i, -1]),
            #         '{:.2f}'.format(output[i, -1]),
            #         horizontalalignment='left', verticalalignment='bottom',
            #         fontsize='small', color='black',
            #         bbox=dict(boxstyle='square,pad=0', color='white'))

    def plotDtGtConfProto(self, image, imgId, iteration, dt=None, dtRaw=None, confusion=None, protoIds=None, gt=None):

        imgResHW = (image.height, image.width)
        fig, ax = self.get_subplots(num_subplots=1, img_res_h_w=imgResHW, margin=0)
        ax.imshow(image)
        ax.axis('off')

        if dtRaw is not None:
            for box in dtRaw:
                xmin, ymin, xmax, ymax, score = box
                w = xmax - xmin
                h = ymax - ymin
                ax.add_patch(plt.Rectangle((xmin, ymin), w, h,
                                           color='white',
                                           fill=False, linewidth=1))

        if gt is not None and len(gt) > 1:
            for box, ignore in zip(gt[0], gt[1]):
                xmin, ymin, w, h = box
                ax.add_patch(plt.Rectangle((xmin, ymin), w, h,
                                           color='blue' if ignore == 0 else 'magenta',
                                           fill=False, linewidth=1))

        if dt is not None and confusion is not None and protoIds is not None:
            for box, key, protoId in zip(dt, confusion, protoIds):
                if key != 'FN':
                    xmin, ymin, xmax, ymax, score = box
                else:
                    xmin, ymin, xmax, ymax = box
                    score = 0

                w = xmax - xmin
                h = ymax - ymin

                ax.add_patch(plt.Rectangle((xmin, ymin), w, h,
                                           color=self.colors[key],
                                           fill=False, linewidth=1))
                ax.text(xmin, ymax,
                        # '{}-{:.2f}-{}'.format(int(h), score, protoId),
                        '{}-{:.2f}'.format(protoId, score),
                        horizontalalignment='left', verticalalignment='bottom',
                        fontsize='small', color='black',
                        bbox=dict(boxstyle='square,pad=0', color=self.colors[key]))

        self.saveFig(fig=fig, file='confusion', name='{}_det'.format(imgId), iteration=iteration,
                     types=['.png', '.eps'])

    def plotConfusion(self, image, dt, key, protoId):

        colors = dict(TP='green', FP='orange', FN='red')

        fig, ax = self.get_subplots(num_subplots=1, img_res_h_w=(image.height, image.width))
        ax.imshow(image)
        ax.axis('off')

        coords = dt[:, :4]
        for i in range(coords.shape[0]):
            xmin = coords[i, 0]
            ymin = coords[i, 1]
            xmax = coords[i, 2]
            ymax = coords[i, 3]
            w = xmax - xmin
            h = ymax - ymin

            # print(xmin, ymin, xmax, ymax)
            ax.add_patch(plt.Rectangle((xmin, ymin), w, h,
                                       color=colors[key],
                                       fill=False, linewidth=2))
            ax.text(xmin, ymax, '{}-{:.2f}-{}'.format(int(h), dt[i, -1], protoId),
                    horizontalalignment='left', verticalalignment='bottom',
                    fontsize='x-small', color='black',
                    bbox=dict(boxstyle='square,pad=0', color=colors[key]))

    def plotConfusionCrop(self, image, dtDict, similarity, imgId, iteration, gtBoxes, gtIgn, dtIgnBoxes):

        boxes = dict(GT=[np.array(box) for n, box in enumerate(gtBoxes) if not gtIgn[n]],
                     GTIG=[np.array(box) for n, box in enumerate(gtBoxes) if gtIgn[n]],
                     TP=dtDict['TP'],
                     FP=dtDict['FP'],
                     FN=dtDict['FN'],
                     DTIG=dtIgnBoxes)

        H, W = 350, 250
        image = np.array(image)

        for dtId, (ids, box) in enumerate(zip(boxes['FP'][0], boxes['FP'][1])):
            fig, ax = self.get_subplots(num_subplots=1, img_res_h_w=(H, W), margin=0)
            ax.axis('off')

            # Crop according to False Positives
            xminFp, yminFp, xmaxFp, ymaxFp = box[0], box[1], box[2], box[3]
            wFp = xmaxFp - xminFp
            hFp = ymaxFp - yminFp
            scoreFp = box[-1]
            cxFP = int(xminFp) + wFp // 2
            cyFP = int(yminFp) + hFp // 2

            xminImg, yminImg = max(0, cxFP - W // 2), max(0, cyFP - H // 2)
            xmaxImg, ymaxImg = min(2048, cxFP + W // 2), min(1024, cyFP + H // 2)
            if xminImg == 0: xmaxImg = W
            if xmaxImg == 2048: xminImg = 2048 - W

            imageCrop = deepcopy(image)[int(yminImg):int(ymaxImg), int(xminImg):int(xmaxImg), :]

            # Plot every else than False Positives
            for key in ['GT', 'GTIG', 'TP', 'FN', 'DTIG']:
                if key in ['TP', 'FN']:
                    for n, (ids, box) in enumerate(zip(boxes[key][0], boxes[key][1])):
                        xmin, ymin, xmax, ymax = box[0], box[1], box[2], box[3]
                        ax.add_patch(plt.Rectangle((xmin - xminImg, ymin - yminImg), xmax - xmin, ymax - ymin,
                                                   color=self.colors[key],
                                                   fill=False, linewidth=1))
                        if key in ['TP']:
                            scoreTp = box[4]
                            ax.text(xmin - xminImg, ymax - yminImg,
                                    # '{}-{:.2f}-{}'.format(int(h), score, protoId),
                                    # '{} | {:.2f}'.format(protoId, score),
                                    '{:.2f}'.format(scoreTp),
                                    horizontalalignment='left', verticalalignment='bottom',
                                    fontsize='x-small', color='black',
                                    bbox=dict(boxstyle='square,pad=0', color=self.colors[key]))

                elif key in ['GT', 'GTIG']:
                    for box in boxes[key]:
                        xmin, ymin, xmax, ymax = box[0], box[1], box[0]+box[2], box[1]+box[3]
                        ax.add_patch(plt.Rectangle((xmin - xminImg, ymin - yminImg), xmax - xmin, ymax - ymin,
                                                   color=self.colors[key],
                                                   fill=False, linewidth=1))

                elif key in ['DTIG']:
                    for n, box in enumerate(boxes[key]):
                        xmin, ymin, xmax, ymax = box[0], box[1], box[2], box[3]
                        ax.add_patch(plt.Rectangle((xmin - xminImg, ymin - yminImg), xmax - xmin, ymax - ymin,
                                                   color=self.colors[key],
                                                   fill=False, linewidth=1))
                        scoreDtIg = box[4]
                        ax.text(xmin - xminImg, ymax - yminImg,
                                '{:.2f}'.format(scoreDtIg),
                                horizontalalignment='left', verticalalignment='bottom',
                                fontsize='x-small', color='black',
                                bbox=dict(boxstyle='square,pad=0', color=self.colors[key]))

            # Determine the Cluster with highest similarity
            centerSimilarity = similarity[:, ids[0], ids[1]]  # [numPrototypes]
            _, protoId = torch.max(centerSimilarity.cpu(), dim=0)

            # Plot False Positives
            ax.add_patch(plt.Rectangle((xminFp-xminImg, yminFp-yminImg), wFp, hFp,
                                       color=self.colors['FP'],
                                       fill=False, linewidth=1))
            ax.text(xminFp-xminImg, ymaxFp-yminImg,
                    # '{}-{:.2f}-{}'.format(int(h), score, protoId),
                    # '{:.2f}({})'.format(score, protoId+1),
                    '{:.2f}'.format(scoreFp),
                    horizontalalignment='left', verticalalignment='bottom',
                    fontsize='x-small',
                    color='black',
                    # color=self.colors['FP'],
                    bbox=dict(boxstyle='square,pad=0', color=self.colors['FP'], alpha=0.7)
            )

            ax.imshow(imageCrop)
            self.saveFig(fig=fig, file='confusion', name='{}_{}_detConf'.format(imgId, dtId), iteration=iteration,
                     types=['.png', '.pdf'])

    @staticmethod
    def plot_heatmap(array, image, ax, px=None, py=None, min_quantile=None, colorbar=False):
        '''
        :param array: in Shape (h, w)
        :param img_res: in Shape (h, w)
        '''
        h, w = array.shape

        # Confidence interval for receptive field heatmap
        ar_min, ar_max = np.amin(array), np.amax(array)

        if min_quantile is not None:
            ar_min = np.quantile(array, min_quantile)
            array = np.clip(array, a_min=ar_min, a_max=ar_max)

        # Normalization is necessary
        array = (array - ar_min) / (ar_max - ar_min)

        # Heatmap has to be scaled in [0, 1] at this point
        cmap = cv2.COLORMAP_JET
        heatmap = cv2.applyColorMap(np.uint8(255 * array), cmap)
        heatmap = np.float32(heatmap) / 255
        heatmap = heatmap[..., ::-1]

        posMask = np.expand_dims(heatmap[:, :, 0] > 0, axis=2)

        image = image.resize(size=(w, h))
        image = np.array(image) / 255

        # image = np.where(heatmap>0, image, image * 0.5)

        overlayed_img = image + heatmap * posMask

        ax.imshow(overlayed_img)
        ax.axis('off')

    @staticmethod
    def plot_interpolated_heatmap(array, image, ax, px=None, py=None, min_quantile=None, colorbar=False,
                                  method='cubic'):
        '''
        :param array: in Shape (h, w)
        :param img_res: in Shape (h, w)
        '''
        h, w = image.height, image.width
        cbar_min, cbar_max = np.amin(array), np.amax(array)

        if (h, w) != array.shape:
            inter_method = {'linear': cv2.INTER_LINEAR,
                            'cubic': cv2.INTER_CUBIC}
            array = cv2.resize(array, dsize=(w, h), interpolation=inter_method[method])

        # assert array.any() < 0, "Problem with interpolation method!"
        # if np.amin(array) < 0:
        #     # ar += -np.amin(ar)
        #     ar = np.clip(array, a_min=0.0, a_max=np.amax(array))
        #     # print('Questionable interpolation result')

        # Confidence interval for receptive field heatmap
        ar_min, ar_max = np.amin(array), np.amax(array)

        if min_quantile is not None:
            ar_min = np.quantile(array, min_quantile)
            array = np.clip(array, a_min=ar_min, a_max=ar_max)

        # Normalization is necessary
        array = (array - ar_min) / (ar_max - ar_min)

        # Heatmap has to be scaled in [0, 1] at this point
        cmap = cv2.COLORMAP_JET
        heatmap = cv2.applyColorMap(np.uint8(255 * array), cmap)
        heatmap = np.float32(heatmap) / 255
        heatmap = heatmap[..., ::-1]

        image = image.resize(size=(w, h))
        image = np.array(image) / 255

        overlayed_img = 0.6 * image + 0.2 * heatmap

        ax.imshow(overlayed_img)
        ax.axis('off')
        # ax.axis('equal')

        if px is not None:
            ax.plot(px, py, 'wo', markersize=3.)

        if colorbar:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('bottom', size='5%', pad=0.05)

            norm = mpl.colors.Normalize(vmin=cbar_min, vmax=cbar_max)
            cb1 = mpl.colorbar.ColorbarBase(ax=cax, cmap=mpl.cm.jet,
                                            norm=norm,
                                            orientation='horizontal')
            # plt.colorbar(overlayed_img, cax=cax, orientation='horizontal', cmap)

    def get_y_x(self, cell, W=512):
        # CONVENTION - INDEX IN GENERAL WITH [y, x]
        y = cell // W
        x = cell - y * W
        return (int(y), int(x))

    def saveFig(self, fig, file, name, iteration, extra_artists=None, types=['.png', '.eps', '.pdf']):
        if self.save:
            for type in types:
                if extra_artists is None:
                    fig.savefig(os.path.join(self.root, file, '{}_{}_{}{}'
                                             .format(name, iteration, self.dataset.image_set, type)), dpi=self.dpi)

                else:
                    fig.savefig(os.path.join(self.root, file, '{}_{}_{}{}'
                                             .format(name, iteration, self.dataset.image_set, type)), dpi=self.dpi,
                                bbox_extra_artists=extra_artists)

    def saveCSV(self, dataFrame, file, name, iteration):
        if self.save:
            dataFrame.to_csv(path_or_buf=os.path.join(self.root, file,
                                                      '{}_{}_{}.csv'.format(name, iteration, self.dataset.image_set)))

    #------------------- Old --------------------------------
    def getProtoInfo(self, model, iteration):
        protoInfo = None

        # Check if already exists
        if self.mode == 'offline':
            onlineExist = os.path.exists(os.path.join(self.config.trained_model_path, 'eval', 'online',
                                                      'proto',
                                                      'proto_info_{}_{}.json'.format(self.dataset.image_set,
                                                                                     iteration)))
            offlineExist = os.path.exists(os.path.join(self.config.trained_model_path, 'eval', 'offline',
                                                       'proto',
                                                       'proto_info_{}_{}.json'.format(self.dataset.image_set,
                                                                                      iteration)))

            if onlineExist:
                protoInfo = self.loadProtoInfo(mode='online', iteration=iteration)
            elif offlineExist:
                protoInfo = self.loadProtoInfo(mode='offline', iteration=iteration)

        if protoInfo is None:
            # Load information about TPs and FNs from MR Evaluation
            evalImgs = self.loadEvalImgs(iteration=iteration)
            if self.mode == 'online' and evalImgs is None:
                print('No ProtoEval due to no evalImgs')
                return None
            else:
                assert evalImgs is not None, 'No ProtoEval due to no evalImgs - Run MREvaluator first!'

            protoInfo = self.makeProtoInfo(model=model, evalImgs=evalImgs)

            path = os.path.join(self.root, 'proto', 'proto_info_{}_{}.json'.format(self.dataset.image_set, iteration))
            with open(path, 'w') as json_file:
                json.dump(protoInfo, json_file)

        # No TP after all
        if len(protoInfo['scores']) == 0:
            print('No ProtoEval due to no TPs')
            return None

        return protoInfo

    def makeProtoInfo(self, model, evalImgs):
        # Extract Variable from DataParallel or not
        modelPrototypes = model.prototypes if not isinstance(model, torch.nn.DataParallel) else model.module.prototypes
        protoInfo = {'prototype': [modelPrototypes[i, :].squeeze().detach().cpu().numpy().tolist() for i in
                                   range(self.config.num_prototypes)],
                     'prototypeFeaturesPosTp': [[] for i in range(self.config.num_prototypes)],
                     # 'prototypeFeaturesFn': [[] for i in range(self.config.num_prototypes)],
                     'prototypeFeaturesNeg': [[] for i in range(self.config.num_prototypes)],
                     'simPosTp': [[] for i in range(self.config.num_prototypes)],
                     # 'simFn': [[] for i in range(self.config.num_prototypes)],
                     'simNeg': [[] for i in range(self.config.num_prototypes)],
                     'scores': [],
                     'posCells': [],
                     'featCenter': [],
                     'similarity': [],
                     }

        model.eval()
        dts = self.get_detections(model)

        with torch.no_grad():
            for i, evalImg in enumerate(evalImgs):

                imgId = evalImg['image_id'] - 1
                x, classes, coords, _ = self.dataset.__getitem__(index=imgId)

                x = x.unsqueeze(dim=0).to(self.device)
                coords = coords.to(self.device)
                classes = classes.to(self.device)

                if self.verbose: print('makeProtoInfo() - Image #{}...'.format(i + 1))

                out = model(x)
                similarity = out['similarity'].squeeze().view(self.config.num_prototypes, -1)
                origSimilarity = out['similarity'].squeeze().cpu().numpy()
                latent_space = out['latent_space'].squeeze().detach().cpu().numpy()
                featCenter = out['featCenter'].squeeze().detach().cpu().numpy()

                # No TPs for this images
                if classes.sum() == 0 or classes.size(0) == 0: continue

                # Extract TP and FN for this Image
                tpMask, fnMask, scores, posCells = self.makeTpFnMask(evalImg=evalImg, dt=dts[imgId])
                classes *= tpMask.long()

                # No TPs for this images
                if classes.sum() == 0: continue

                protoInfo['scores'].append(scores[tpMask].tolist())
                posCells = posCells[tpMask, :].tolist()
                protoInfo['posCells'].append(posCells)
                protoInfo['featCenter'].append([featCenter[:, y, x].tolist() for y, x in posCells])
                protoInfo['similarity'].append([origSimilarity[:, y, x].tolist() for y, x in posCells])

                # Apply Pos and Neg Mask
                posSim, negSim = self.applyPosNegMask(model=model,
                                                      coords=coords,
                                                      classes=classes,
                                                      tensor=similarity)  # [num_prototypes, num_cells, num_gts]

                numTps = posSim.shape[2]
                for p in range(self.config.num_prototypes):
                    # posProto = posSim[p, :]  # [num_cells, num_gts]
                    negProto = negSim[p, :].squeeze()  # [num_cells]

                    # --------------------------------- BB-In-TP --------------------------
                    # For complete Pos Area
                    # maxProtoSimIdx = np.argmax(posProto, axis=0)
                    # maxProtoSim = posProto[maxProtoSimIdx, np.arange(numTps)]
                    # maxProtoSimYX = [self.get_y_x(cell=idx) for idx in maxProtoSimIdx.tolist()]

                    # According to final Detection
                    protoSim, _latent_space = [], []
                    for y, x in protoInfo['posCells'][-1]:
                        protoSim.append(float(origSimilarity[p, y, x]))
                        _latent_space.append(latent_space[:, y, x].tolist())

                    protoInfo['simPosTp'][p].append(protoSim)
                    protoInfo['prototypeFeaturesPosTp'][p].append(_latent_space)

                    # --------------------------------- BB-Out ----------------------------
                    # bb_out_max = bb_out[np.argpartition(bb_out, -num_gts)[-num_gts:]].tolist()
                    maxNegProtoSimIdx = np.argpartition(negProto, -numTps)[-numTps:]
                    maxNegProtoSim = negProto[maxNegProtoSimIdx]
                    maxNegProtoSimYX = [self.get_y_x(cell=idx) for idx in maxNegProtoSimIdx.tolist()]

                    protoInfo['simNeg'][p].append(maxNegProtoSim.tolist())
                    _latent_space = []
                    for y, x in maxNegProtoSimYX:
                        _latent_space.append(latent_space[:, y, x].tolist())
                    protoInfo['prototypeFeaturesNeg'][p].append(_latent_space)

                    assert len(protoSim) == len(maxNegProtoSim)

        return protoInfo

    def loadProtoInfo(self, mode, iteration):
        json_path = os.path.join(self.config.trained_model_path, 'eval', mode,
                                 'proto', 'proto_info_{}_{}.json'.format(self.dataset.image_set, iteration))

        with open(json_path, "r") as json_file:
            protoInfo = json.load(json_file)

        print('Loaded ProtoInfo.')
        return protoInfo

    def readProtoInfo(self, protoInfo, model):
        # X and y from dict
        X = []  # [num_samples, num_features]
        xInfo = {'featType': [],  # p: prototype or tp: True Positive (BB-In) or neg: Negative (BB-Out)
                 'score': [],  # score only for TPs
                 'cluster': [],  # 1, 2, 3, 4 according to numPrototypes [num_samples]
                 'similarity': [],
                 'decMaker': [],
                 'influence': [],
                 'influenceAll': [],
                 'posInfluence': [],
                 }

        # Softmax needs to include the final Classification Weights!!
        if isinstance(model, torch.nn.DataParallel):
            clWeight = model.module.center.weight.data.squeeze().cpu().numpy()
        else:
            clWeight = model.center.weight.data.squeeze().cpu().numpy()

        # Adding prototypes
        for protoId, prototype in enumerate(protoInfo['prototype']):
            X += [prototype]
            xInfo['cluster'] += [protoId + 1]
            xInfo['featType'] += ['Prototype']
            xInfo['score'] += [None]
            xInfo['similarity'] += [None]
            xInfo['decMaker'] += [None]
            xInfo['influence'] += [None]
            xInfo['influenceAll'] += [None]
            xInfo['posInfluence'] += [None]

        # Adding and Sorting PrototypeFeatures
        for protoId in range(self.config.num_prototypes):
            # TP and BB-In
            for imgId in range(len(protoInfo['prototypeFeaturesPosTp'][protoId])):
                for gtId, prototype_feature in enumerate(protoInfo['prototypeFeaturesPosTp'][protoId][imgId]):
                    X += [prototype_feature]
                    xInfo['cluster'] += [protoId + 1]
                    xInfo['featType'] += ['TP']
                    xInfo['score'] += [protoInfo['scores'][imgId][gtId]]
                    xInfo['similarity'] += [protoInfo['similarity'][imgId][gtId][protoId]]

                    featCenter = protoInfo['featCenter'][imgId][gtId]
                    featCenterAct = np.array(featCenter) * clWeight

                    softmax = np.exp(featCenterAct) / np.sum(np.exp(featCenterAct))
                    mostInfluentialId = int(np.argmax(softmax))
                    # Centerness has always ID 0
                    if featCenterAct.size == self.config.num_prototypes:
                        mostInfluentialId += 1

                    xInfo['decMaker'] += [mostInfluentialId]
                    xInfo['influenceAll'] += [softmax]
                    xInfo['influence'] += [softmax[protoId]]
                    xInfo['posInfluence'] += [softmax[protoId] > 0]

            # Neg and BB-Out
            for imgId in range(len(protoInfo['prototypeFeaturesNeg'][protoId])):
                for gtId, prototype_feature in enumerate(protoInfo['prototypeFeaturesNeg'][protoId][imgId]):
                    X += [prototype_feature]
                    xInfo['cluster'] += [protoId + 1]
                    xInfo['featType'] += ['N']
                    xInfo['score'] += [None]
                    xInfo['similarity'] += [protoInfo['simNeg'][protoId][imgId][gtId]]

                    xInfo['decMaker'] += [None]
                    xInfo['influence'] += [None]
                    xInfo['influenceAll'] += [None]
                    xInfo['posInfluence'] += [None]

        X = np.array(X)  # [num_samples, num_features]
        xInfo = pd.DataFrame(xInfo)

        return X, xInfo
