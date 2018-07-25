import argparse
import json
import os
import shutil

from meta_learning import CONFIG
from meta_learning.a2c import train


def parse_flags():

    parser = argparse.ArgumentParser()

    parser.add_argument('--env', default='binary', help='Name of environment.')
    parser.add_argument('--model_dir', default='./saved', help='Directory to save run information in.')
    parser.add_argument('--seed', default=3, type=int, help='Random seed.')
    parser.add_argument('--n_iter', default=1000, type=int, help='Number iterations.')
    parser.add_argument('--dont_normalize_advantages', '-dna', action='store_true',
                        help='Advantage normalization can help with stability. Turns that feature off.')
    parser.add_argument('--mix_halflife', default=10000.0, type=float, help='Halflife of gradient/action mixture.')
    parser.add_argument('--mix_start', default=1.0, type=float,
                        help='Starting weight for magnitude of gradient contributions.')
    parser.add_argument('--batch_size', default=3, type=int, help='Batch size when collecting gradients for policy.')
    parser.add_argument('--discount', default=0.99, type=float, help='Discount rate for rewards.')
    parser.add_argument('--ep_len', default=10, type=int, help='Number of steps before performing an update.')
    parser.add_argument('--num_lstm_units', default=3, type=int, help='Number LSTM units in policy hidden layer.')
    parser.add_argument('--grad_reg', type=int, help='Cap for l2 norm of gradients.')
    parser.add_argument('--no_xover', action='store_true', help='Treat each gradient as separate input to policy.')
    parser.add_argument('--render', action='store_true', help='Not implemented.')
    parser.add_argument('--load_params_torch', help='Load policy parameters from a pytorch model.')
    parser.add_argument('--histogram_parameters', action='store_true', help='Record histogram of parameters.')
    parser.add_argument('--reset', action='store_true',
                        help='Delete the existing model directory and start training from scratch.')
    parser.add_argument('--debug', action='store_true', help='Run session in a CLI debug session.')
    parser.add_argument('--run_name', default='default', help='Name of run.')

    parser.parse_args(namespace=CONFIG)


def proc_flags():

    run_name_components = [CONFIG.run_name,
                           CONFIG.env]
    CONFIG.dpath_model = os.path.join(CONFIG.model_dir, '_'.join(run_name_components))

    # Optionally empty model directory
    if CONFIG.reset:
        shutil.rmtree(CONFIG.dpath_model, ignore_errors=True)

    CONFIG.dpath_checkpoint = os.path.join(CONFIG.dpath_model, 'checkpoints')
    if not os.path.exists(CONFIG.dpath_checkpoint):
        os.makedirs(CONFIG.dpath_checkpoint)

    with open(os.path.join(CONFIG.dpath_model, 'config.txt'), 'w') as f:
        json.dump(vars(CONFIG), f, indent=4)


if __name__ == '__main__':

    parse_flags()
    proc_flags()
    train()
