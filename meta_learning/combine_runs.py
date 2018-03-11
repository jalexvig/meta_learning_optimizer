import os
from collections import defaultdict

import numpy as np
import tensorflow as tf
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

import shutil
import glob


def _tabulate_events(dpath):

    summary_iterators = [EventAccumulator(os.path.join(dpath, dname, 'train')).Reload() for dname in os.listdir(dpath)]

    tags = summary_iterators[0].Tags()['scalars']

    for it in summary_iterators:
        assert set(it.Tags()['scalars']) == set(tags)

    out = defaultdict(list)

    for tag in tags:
        for events in zip(*[acc.Scalars(tag) for acc in summary_iterators]):
            assert len(set(e.step for e in events)) == 1

            out[tag].append([e.value for e in events])

    return out


def _write_combined_events(dpath, d_combined, dname='combined'):

    fpath = os.path.join(dpath, dname)
    writer = tf.summary.FileWriter(fpath)

    tags, values = zip(*d_combined.items())

    timestep_mean = np.array(values).mean(axis=-1)

    for tag, means in zip(tags, timestep_mean):
        for i, mean in enumerate(means):
            summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=mean)])
            writer.add_summary(summary, global_step=i)

        writer.flush()


def _combine_dirs(dpath, chop, sep='_'):

    s = set(sep.join(dname.split(sep)[:-chop]) for dname in os.listdir(dpath))

    for exp_name in s:
        dpath_prefix = os.path.join(dpath, exp_name)
        if not os.path.isdir(dpath_prefix):
            os.mkdir(dpath_prefix)
        for dpath_seed in glob.glob(dpath_prefix + '*'):
            if dpath_seed != dpath_prefix:
                shutil.move(dpath_seed, dpath_prefix)


def combine_events(args):

    if not args.combined:
        _combine_dirs(args.dpath, args.chop)

    for dpath in glob.glob(os.path.join(args.dpath, '*')):
        d = _tabulate_events(dpath)

        _write_combined_events(dpath, d)


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('dpath', help='Directory path to runs.')
    parser.add_argument('-c', '--combined', action='store_true')
    parser.add_argument('--chop', help='Number of directory name parameters to ignore.', type=int, default=2)

    args = parser.parse_args()

    combine_events(args)
