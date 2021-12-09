import os
import json

from itertools import cycle

from patrick.evaluator.base_evaluator import BaseEvaluator

class EvaluatorMrBenchmark(BaseEvaluator):
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

        self.modelDict = {
            'CSP': {'model_name': '20201112_175026', 'iter': '34500', 'dir': 'csp_models'},
            'CSPP': {'model_name': '20201206_125558', 'iter': '44000', 'dir': 'cspp_models'},
        }

        self.jsonRoot = '../../input/model_zoo'

        lines = ["-", "--", "-.", ":"]
        self.linecycler = cycle(lines)

    def evaluate(self, model, iteration=None):

        # Plot MissRate over FPPI
        fig, ax = self.get_subplots(num_subplots=1)

        for model, info in self.modelDict.items():
            filePath = os.path.join(self.jsonRoot, info['dir'], info['model_name'], 'eval', 'offline', 'mr',
                                    'mr_fppi_{}_val.json'.format(info['iter']))

            assert os.path.exists(filePath), 'No .json available for {}'.format(info['model_name'])

            with open(filePath, "r") as read_file:
                data = json.load(read_file)

            f = ax.plot(data['fppi'], data['mr'],
                        # color='grey',
                        label='{:.2f}% {}'.format(data['lamr'] * 100, model),
                        # linestyle=next(self.linecycler),
                        linewidth=2,
                        )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_yticks([1, 0.1, 0.01])
        ax.set_xticks([0.0001, 0.001, 0.01, 0.1, 1.0, 10.0])
        ax.set_xlabel("FPPI")
        ax.set_ylabel("Miss rate")
        ax.grid(b=True, which='major', axis='x', linestyle='-', linewidth=1)
        ax.grid(b=True, which='major', axis='y', linestyle='-', linewidth=1)
        ax.grid(b=True, which='minor', axis='y', linestyle='--', linewidth=1)
        ax.legend()

        if self.save:
            self.saveFig(fig=fig, file='mr', name='mr_benchmark', iteration=iteration)
