import json
import os

from easydict import EasyDict
from pycocotools.cocoeval import COCOeval

from atasets.evaluator import BaseEvaluator


class COCOEvaluator(BaseEvaluator):
    """
    This class is responsible for generating the COCO evaluation metrics. To
    generate the mAP results we are currently using the COCO API provided in the
    pycocotools library. Later we want to build a similar libarry by ourselves
    to mitigate some issues with the library. Furthermore, this class will plot
    some graphs for better understanding of the evaluation metrics.
    """

    def __init__(self, dataset, config, device, plotter=None,
                 verbose=True) -> None:
        super().__init__(dataset, config, device, plotter, verbose)
        self.conf_threshold = 0.3
        self.nms_threshold = 0.6

    def evaluate(self, model, iteration=None) -> None:
        """
        Generates the COCO evaluation metrics using the official COCO API.
        After that the function plots the results to visdom for better
        understanding.

        :param model: The model that should be evaluated.
        :param iteration: The current iteration when the evaluation happened.
        """
        super().evaluate(model, iteration)

        # Save the json file with the detections to the checkpoints folder
        self._generate_json(self.absolute_detections)

        detections_file = os.path.join(self.config.checkpoints_model_folder,
                                       'coco_detections.json')
        annotation_type = 'bbox'

        try:
            coco_detections = self.dataset.coco_annotations.loadRes(
                detections_file)
            coco_evaluation = COCOeval(self.dataset.coco_annotations,
                                       coco_detections)
            coco_evaluation.params.iouType = annotation_type
            coco_evaluation.params.imgIds = self.dataset.image_ids
            coco_evaluation.evaluate()
            coco_evaluation.accumulate()
            coco_evaluation.summarize()

            if self.config.training_vis:
                self._plot_precision_stats(coco_evaluation, iteration)
                self._plot_recall_stats(coco_evaluation, iteration)

        except IndexError:
            print("Could not generate COCO stats because of no detections!")

        print(
            '----------------------------------------------------------------------------------------')

    def _plot_precision_stats(self, coco_evaluation, iteration) -> None:
        """
        Only plots the given precision statistics that have been generated.

        :param coco_evaluation: The COCOeval Object that contains the current
            precision statistics.
        :param iteration: The current iteration.
        """
        title_name = '%s_%s_%s - %s' % (
            self.config.model, self.config.backbone, self.config.dataset,
            'mAP Precision')
        self.plotter.plot(var_name='mAP Precision',
                                split_name='[ IoU=0.50:0.95 | area=   all | maxDets=100 ]',
                                title_name=title_name, x=iteration,
                                y=coco_evaluation.stats[0])
        self.plotter.plot(var_name='mAP Precision',
                                split_name='[ IoU=0.50      | area=   all | maxDets=100 ]',
                                title_name=title_name, x=iteration,
                                y=coco_evaluation.stats[1])
        self.plotter.plot(var_name='mAP Precision',
                                split_name='[ IoU=0.75      | area=   all | maxDets=100 ]',
                                title_name=title_name, x=iteration,
                                y=coco_evaluation.stats[2])
        self.plotter.plot(var_name='mAP Precision',
                                split_name='[ IoU=0.50:0.95 | area= small | maxDets=100 ]',
                                title_name=title_name, x=iteration,
                                y=coco_evaluation.stats[3])
        self.plotter.plot(var_name='mAP Precision',
                                split_name='[ IoU=0.50:0.95 | area=medium | maxDets=100 ]',
                                title_name=title_name, x=iteration,
                                y=coco_evaluation.stats[4])
        self.plotter.plot(var_name='mAP Precision',
                                split_name='[ IoU=0.50:0.95 | area= large | maxDets=100 ]',
                                title_name=title_name, x=iteration,
                                y=coco_evaluation.stats[5])

    def _plot_recall_stats(self, coco_evaluation, iteration) -> None:
        """
        Only plots the given recall statistics that have been generated.

        :param coco_evaluation: The COCOeval Object that contains the current
            recall statistics.
        :param iteration: The current iteration.
        """
        title_name = '%s_%s_%s - %s' % (
            self.config.model, self.config.backbone, self.config.dataset,
            'mAP Recall')
        self.plotter.plot(var_name='mAP Recall',
                                split_name='[ IoU=0.50:0.95 | area=   all | maxDets=  1 ]',
                                title_name=title_name, x=iteration,
                                y=coco_evaluation.stats[6])
        self.plotter.plot(var_name='mAP Recall',
                                split_name='[ IoU=0.50:0.95 | area=   all | maxDets= 10 ]',
                                title_name=title_name, x=iteration,
                                y=coco_evaluation.stats[7])
        self.plotter.plot(var_name='mAP Recall',
                                split_name='[ IoU=0.50:0.95 | area=   all | maxDets=100 ]',
                                title_name=title_name, x=iteration,
                                y=coco_evaluation.stats[8])
        self.plotter.plot(var_name='mAP Recall',
                                split_name='[ IoU=0.50:0.95 | area= small | maxDets=100 ]',
                                title_name=title_name, x=iteration,
                                y=coco_evaluation.stats[9])
        self.plotter.plot(var_name='mAP Recall',
                                split_name='[ IoU=0.50:0.95 | area=medium | maxDets=100 ]',
                                title_name=title_name, x=iteration,
                                y=coco_evaluation.stats[10])
        self.plotter.plot(var_name='mAP Recall',
                                split_name='[ IoU=0.50:0.95 | area= large | maxDets=100 ]',
                                title_name=title_name, x=iteration,
                                y=coco_evaluation.stats[11])

    def _generate_json(self, detections) -> None:
        """
        This method generates an annotation file which contains all the found
        detections. The annotations meet the COCO annotation dict structure.
        The annotations are saved as a .json-file in the checkpoints folder.

        :param detections: Absolute output of the detector.
        """

        # Reverse the label mapping that was applied by the dataset.
        label_mapping = self.dataset.get_label_map()
        # Keys: Intermediate Ids, Values: COCO ids
        backmap = {y: x for x, y in label_mapping.items()}

        det_list = []
        for i in range(0, len(detections)):

            class_ids = detections[i][:, 0]

            for j in range(0, class_ids.size):
                # Backmapping to original COCO class_ids
                class_ids[j] = backmap[class_ids[j]]

                dic = EasyDict()
                bbox_list = []

                score = float(detections[i][j, 1])
                x1 = float(detections[i][j, 2])
                x2 = float(detections[i][j, 4])
                y1 = float(detections[i][j, 3])
                y2 = float(detections[i][j, 5])
                width = x2 - x1
                height = y2 - y1
                bbox_list.append(x1)
                bbox_list.append(y1)
                bbox_list.append(width)
                bbox_list.append(height)
                dic.image_id = self.dataset.image_ids[i]
                dic.category_id = int(class_ids[j])
                dic.bbox = bbox_list
                dic.score = score
                det_list.append(dic)

        # Generate the json file with all detections
        folder = self.config.checkpoints_model_folder
        path = os.path.join(folder, 'coco_detections.json')
        with open(path, 'w') as json_file:
            json.dump(det_list, json_file)
