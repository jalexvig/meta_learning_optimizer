import argparse
import json
import os
import shutil

from meta_learning import CONFIG
from meta_learning.a2c import train


def parse_flags():

    parser = argparse.ArgumentParser()

    parser.add_argument('--env', default='binary', help='Name of environment')
    parser.add_argument('--model_dir', default='./saved')
    parser.add_argument('--seed', default=3, type=int, help='Random seed')
    parser.add_argument("--n_iter", default=1000, help="Number iterations", type=int)
    parser.add_argument('--dont_normalize_advantages', '-dna', action='store_true')
    parser.add_argument("--mix_halflife", default=10000.0, help="Halflife of gradient/action mixture", type=float)
    parser.add_argument("--mix_start", default=1.0, help="Starting weight for gradient", type=float)
    parser.add_argument("--batch_size", default=3, help="Batch size", type=int)
    parser.add_argument("--discount", default=0.99, help="Discount rate for rewards", type=float)
    parser.add_argument("--ep_len", default=10, help="Number of steps before performing an update", type=int)
    parser.add_argument("--num_lstm_units", default=4, help="Number LSTM units", type=int)
    parser.add_argument("--grad_reg", help="Cap for l2 norm of gradients", type=int)
    parser.add_argument('--no_xover', action='store_true')
    parser.add_argument('--render', action='store_true')
    parser.add_argument("--histogram_parameters", action='store_true')
    parser.add_argument("--reset", action='store_true',
                        help="If set, delete the existing model directory and start training from scratch.")
    parser.add_argument('--debug', action='store_true')
    parser.add_argument("--run_name", default="default", help="Name of run.")

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
