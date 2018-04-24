import itertools
import os
import time

import gym
import numpy as np
import tensorflow as tf
from tensorflow.python import debug as tf_debug

from meta_learning import CONFIG
from meta_learning.a2c.estimators import PolicyEstimator, ValueEstimator
from meta_learning.environments import MNIST, BinaryClassifier

STABILITY = 1e-8


def train():

    env, inc_gs, policy_net, value_net = _setup()

    writer = tf.summary.FileWriter(os.path.join(CONFIG.dpath_model, 'train'))
    saver = tf.train.Saver(keep_checkpoint_every_n_hours=2.0, max_to_keep=10)

    if CONFIG.histogram_parameters:
        _add_histograms()

    with tf.Session() as sess:

        if CONFIG.debug:
            sess = tf_debug.LocalCLIDebugWrapperSession(sess)

        tf.global_variables_initializer().run()

        latest_checkpoint = tf.train.latest_checkpoint(CONFIG.dpath_checkpoint)

        if latest_checkpoint:
            print("Loading model checkpoint: {}".format(latest_checkpoint))
            saver.restore(sess, latest_checkpoint)

        env.set_session(sess)

        for batch_idx in range(CONFIG.n_iter):

            paths = []
            for ep_idx in range(CONFIG.batch_size):
                obs = env.reset()
                observations, actions, rewards = [], [], []
                render_this_episode = (not paths and (batch_idx % 10 == 0) and CONFIG.render)
                steps = 0

                recurrent_state = np.zeros((2, 1, policy_net.state_placeholders[0].shape[-1]))

                for _ in itertools.count():
                    if render_this_episode:
                        env.render()
                        time.sleep(0.05)
                    observations.append(obs)

                    feed = {
                        policy_net.observations: obs[None, None],
                        policy_net.state_placeholders[0]: recurrent_state[0],
                        policy_net.state_placeholders[1]: recurrent_state[1]
                    }

                    sampled_action, recurrent_state, gs_env_val = sess.run(
                        [policy_net.sampled, policy_net.new_state, inc_gs],
                        feed_dict=feed)

                    actions.append(sampled_action[0][0])
                    obs, rew, done, _ = env.step(sampled_action)
                    rewards.append(rew)
                    steps += 1
                    if done:
                        break
                path = {"observation": np.array(observations),
                        "reward": np.array(rewards),
                        "action": np.array(actions),
                        "last_observation": obs,
                        }
                paths.append(path)

            # Build arrays for observation, action for the policy gradient update by concatenating
            # across paths
            observations = np.array([path["observation"] for path in paths])
            actions = np.array([path["action"] for path in paths])

            qs, final_obs = [], []

            # Multiply each step in path by appropriate discount
            for path in paths:
                n = path["reward"].shape[0]
                discounts = CONFIG.discount ** np.arange(n)
                discounted_rew_seq = discounts * path["reward"]
                q_path = np.cumsum(discounted_rew_seq[::-1])[::-1] / discounts

                qs.append(q_path)
                final_obs.append(path['last_observation'])

            qs = np.array(qs)
            final_obs = np.array(final_obs)

            observations_w_final = np.concatenate([observations, final_obs[:, None]], axis=1)

            feed = {
                value_net.observations: observations_w_final
            }

            vals = sess.run(value_net.predicted_values, feed)
            vals, vals_final = np.split(vals, [-1], axis=1)

            n_timesteps = qs.shape[1]
            vals_final_disc = vals_final * (np.arange(n_timesteps, 0, -1) ** CONFIG.discount)
            qs = qs + vals_final_disc

            vals_norm = _normalize(vals, qs.mean(), qs.std())
            advs = qs - vals_norm

            if not CONFIG.dont_normalize_advantages:
                advs = _normalize(advs)

            recurrent_state = np.zeros((2, observations.shape[0], policy_net.state_placeholders[0].shape[-1]))

            feed_dict = {
                policy_net.observations: observations,
                policy_net.actions: actions,
                policy_net.targets: advs,
                value_net.observations: observations,
                value_net.targets: _normalize(qs),
                policy_net.state_placeholders[0]: recurrent_state[0],
                policy_net.state_placeholders[1]: recurrent_state[1]
            }

            summaries = tf.summary.merge_all(tf.GraphKeys.SUMMARIES)

            policy_loss, _, value_loss, _, summaries_all = sess.run([
                policy_net.loss,
                policy_net.update_op,
                value_net.loss,
                value_net.update_op,
                summaries,
            ], feed_dict=feed_dict)

            writer.add_summary(summaries_all, global_step=batch_idx + 1)

            add_path_summaries(batch_idx, paths, writer)

            writer.flush()

        saver.save(sess, os.path.join(CONFIG.dpath_model, 'checkpoints', 'model'))


def _setup():
    """
    Initialize environment, policy model, and value model.
    Args:

    Returns:
        (gym.Environment, policy model, value model)
    """

    # Set random seeds
    tf.set_random_seed(CONFIG.seed)
    np.random.seed(CONFIG.seed)

    gs_env = tf.Variable(0, trainable=False, name='global_step_env')
    inc_gs = tf.assign_add(gs_env, 1)

    # Make the gym environment
    if CONFIG.env == 'mnist':
        env = MNIST()
    elif CONFIG.env == 'binary':
        env = BinaryClassifier(done_every=CONFIG.ep_len)
    else:
        raise ValueError('Do not recognize environment ', CONFIG.env)

    discrete = isinstance(env.action_space, gym.spaces.Discrete)

    # Observation and action sizes
    ob_dim = env.observation_space.shape[0]
    ac_dim = env.action_space.n if discrete else env.action_space.shape[0]

    policy_net = PolicyEstimator(ob_dim, ac_dim, gs_env)
    value_net = ValueEstimator(ob_dim)

    return env, inc_gs, policy_net, value_net


def _add_histograms(scope=None):

    trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)

    for var in trainable_vars:
        tf.summary.histogram(var.name, var)


def add_path_summaries(itr, paths, writer):

    data = {
        'returns': [path["reward"].sum() for path in paths],
    }

    funcs = {
        'mean': np.mean,
        'std': np.std,
        'max': np.max,
        'min': np.min,
    }

    summary_values = []
    for data_name, func_name in [
        ('returns', 'mean'),
        ('returns', 'std'),
        ('returns', 'max'),
        ('returns', 'min'),
    ]:
        name = 'path/%s/%s' % (data_name, func_name)
        val = funcs[func_name](data[data_name])
        value = tf.Summary.Value(tag=name, simple_value=val)
        summary_values.append(value)

    summary = tf.Summary(value=summary_values)
    writer.add_summary(summary, global_step=itr + 1)


def _normalize(a: np.ndarray, u: float=0, s: float=1) -> np.ndarray:
    """
    Normalize to a new mean/std. This defaults to producing z-scores.

    Args:
        a: Array.
        u: Mean.
        s: Standard deviation.

    Returns:
        Array that has been renormalized.
    """
    a_norm = (a - np.mean(a)) / (np.std(a) + STABILITY)
    a_rescaled = a_norm * s + u

    return a_rescaled
