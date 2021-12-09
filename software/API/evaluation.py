import argparse
import torch
import sys
import os
sys.path.insert(0, os.getcwd())

import matplotlib.pyplot as plt

from patrick.evaluator.evaluator_builder import EvaluatorBuilder


def main(eval_config_root, device):

    evaluator_offline = EvaluatorBuilder().offline(eval_config_root=eval_config_root,
                                                   device=torch.device(device))
    evaluators = evaluator_offline.get_evaluators()

    # Actual evaluation of multipls evaluators
    with torch.no_grad():
        [evaluator.evaluate(model=evaluator_offline.model,
                            iteration=evaluator.config.iteration) for evaluator in evaluators]

    plt.show() if evaluator_offline.config.plot and evaluator_offline.mode == 'offline' else plt.close('all')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download and extract')
    parser.add_argument('--eval_config_root', type=str, default='./cfg_eval.yaml', required=False)
    parser.add_argument('--device', type=str, default='cuda:0', required=False)
    args = parser.parse_args()

    main(args.eval_config_root, args.device)
