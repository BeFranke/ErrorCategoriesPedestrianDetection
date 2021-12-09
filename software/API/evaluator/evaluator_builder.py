import sys

import yaml
import importlib
import torch
import json
import os

import torch.utils.data as data

from easydict import EasyDict
from copy import deepcopy

# from datasets.pascalvoc.voc_adapter import VOCAdapter
# from datasets.fusion_dataset import FusionDataset
# from datasets.inria.inria_dataset import INRIADataset
# from datasets.ecp.ecp_dataset import ECPDataset
from patrick.datasets.cityscapes.cityscapes_dataset import CityscapesDataset
from patrick.datasets.kia.kia_dataset import KIADataset

# from evaluator._archive.evaluator_proto import EvaluatorProto
# from patrick.evaluator.evaluator_inference import EvaluatorInference
from patrick.evaluator.MR.evaluator_MR import EvaluatorMR
# from evaluator._archive.evaluator_decision import EvaluatorDecision
# from evaluator.evaluator_model import EvaluatorModel
# from evaluator.evaluator_cluster import EvaluatorCluster
from patrick.evaluator.evaluator_mr_benchmark import EvaluatorMrBenchmark
# from evaluator._archive.evaluator_latent_space import EvaluatorLatentSpace
# from evaluator.evaluator_reconstruction import EvaluatorReconstruction
# from evaluator.evaluator_latent_space import EvaluatorLatentSpace
# from evaluator.evaluator_quality import EvaluatorQuality


class EvaluatorBuilder(object):
    """
    The 'EvaluatorBuilder' allows you to create any kind of 'Evaluator'.
    """

    def __init__(self) -> None:
        self.dataset = None
        self.config = None
        self.plotter = None
        self.device = None

    def online(self, config: EasyDict, device: torch.device,
               dataset: data.Dataset, plotter=None) -> 'EvaluatorBuilder':
        self.config = config
        self.device = device
        self.dataset = dataset
        self.plotter = plotter
        self.mode = 'online'
        self.evaltype = self.config.evaltype
        return self

    def offline(self, eval_config_root: str, device: torch.device) -> 'EvaluatorBuilder':
        self.eval_config_root = eval_config_root
        self._load_eval_config()
        self.device = device
        self._load_config()
        self._overwrite_config()
        self._load_model()
        self._load_dataset()
        self.mode = 'offline'
        self.evaltype = self.eval_config.evaltype
        return self

    def _load_eval_config(self):
        with open(self.eval_config_root, "r") as file:
            self.eval_config = EasyDict(yaml.safe_load(file))

    def _load_config(self):
        self.trained_model_path = '../../input/model_zoo/{}_models/{}/'.format(self.eval_config.model.split("_")[0],
                                                                               self.eval_config.model_id)
        self.model_name = '{}_{}'.format(self.eval_config.model, self.eval_config.backbone)

        with open(self.trained_model_path + self.model_name + '.txt') as json_file:
            self.config = EasyDict(json.load(json_file))
        self.config['trained_model_path'] = self.trained_model_path

    def _overwrite_config(self):
        for key, value in self.eval_config.items():
            self.config[key] = value

    def _load_model(self):
        # Load/ import models dynamically
        module = importlib.import_module(
            '.' + self.config.model,
             package=f'patrick.models.{self.config.model.split("_")[0]}.{self.config.model}'
        )
        model = getattr(module, self.config.model.upper())
        self.model = model(self.config, self.device)

        # Load Model StateDict
        dict_path = os.path.join(self.trained_model_path,
                                 '{}_{}.pth'.format(self.model_name, self.eval_config.iteration))

        assert os.path.exists(dict_path), 'No Model State Dict was loaded!'

        pretrained_dict = torch.load(dict_path, map_location=self.device)
        self.model.load_state_dict(pretrained_dict, strict=False)
        self.model.to(self.device)
        self.model.eval()

    def _load_dataset(self):
        _dataset = self.eval_config.dataset

        if _dataset == 'cityscapes':
            dataset = CityscapesDataset(config=self.config,
                                        image_set=self.config.image_set['val'],
                                        mode='val',
                                        augmentation=False)
        elif _dataset == 'kia':
            dataset = KIADataset(config=self.config,
                                 image_set=self.config.image_set['val'],
                                 mode='val',
                                 augmentation=False)

        dataset.calc_basic_stats()
        dataset.print_stats(dataset=(self.eval_config.dataset, 'val'))
        self.dataset = dataset

    def get_evaluators(self):
        """
        Creates a new object whose super type is 'Evaluator'.
        """

        if not self.config:
            raise EvaluatorException("You need to specify the config of your script.")

        if not self.device:
            raise EvaluatorException("You need to specify the device for your tensors.")

        evaluators = []

        if 'model' in self.evaltype:
            _evaluator = EvaluatorModel(dataset=deepcopy(self.dataset), config=self.config,
                                        device=self.device, plotter=self.plotter,
                                        mode=self.mode)
            evaluators += [_evaluator]

        if 'mrBenchmark' in self.evaltype:
            _evaluator = EvaluatorMrBenchmark(dataset=deepcopy(self.dataset), config=self.config,
                                              device=self.device, plotter=self.plotter,
                                              mode=self.mode)
            evaluators += [_evaluator]

        if 'mr' in self.evaltype:
            _evaluator = EvaluatorMR(dataset=deepcopy(self.dataset), config=self.config,
                                     device=self.device, plotter=self.plotter,
                                     mode=self.mode)
            evaluators += [_evaluator]

        if 'cluster' in self.evaltype:
            _evaluator = EvaluatorCluster(dataset=deepcopy(self.dataset), config=self.config,
                                          device=self.device, plotter=self.plotter,
                                          mode=self.mode)
            evaluators += [_evaluator]

        if 'decision' in self.evaltype:
            _evaluator = EvaluatorDecision(dataset=deepcopy(self.dataset), config=self.config,
                                           device=self.device, plotter=self.plotter,
                                           mode=self.mode)
            evaluators += [_evaluator]

        if 'proto' in self.evaltype:
            _evaluator = EvaluatorProto(dataset=deepcopy(self.dataset), config=self.config,
                                        device=self.device, plotter=self.plotter,
                                        mode=self.mode)
            evaluators += [_evaluator]

        if 'inference' in self.evaltype:
            _evaluator = EvaluatorInference(dataset=deepcopy(self.dataset), config=self.config,
                                            device=self.device, plotter=self.plotter,
                                            mode=self.mode)
            evaluators += [_evaluator]

        if 'latent_space' in self.evaltype:
            _evaluator = EvaluatorLatentSpace(dataset=deepcopy(self.dataset), config=self.config,
                                              device=self.device, plotter=self.plotter,
                                              mode=self.mode)
            evaluators += [_evaluator]

        if 'reconstruction' in self.evaltype:
            _evaluator = EvaluatorReconstruction(dataset=deepcopy(self.dataset), config=self.config,
                                                 device=self.device, plotter=self.plotter,
                                                 mode=self.mode)
            evaluators += [_evaluator]

        if 'quality' in self.evaltype:
            _evaluator = EvaluatorQuality(dataset=deepcopy(self.dataset), config=self.config,
                                          device=self.device, plotter=self.plotter,
                                          mode=self.mode)
            evaluators += [_evaluator]

        return evaluators


class EvaluatorException(BaseException):
    """
    Serves as base exception type for exceptions that might occur in the
    'EvaluatorBuilder' or in a specific 'Evaluator' implementation like the
    'COCOEvaluator'. You can specify an error message to make it easier for your
    user to understand the problem of the code.
    """

    def __init__(self, message) -> None:
        """
        Allows you to specify the message for the raised Exception. The message
        should appear on the user's CLI.

        :param message: The message for the user.
        """
        super().__init__(message)
