import tensorflow as tf
import numpy as np
from tensorflow.contrib.rnn import LSTMCell
from tensorflow.contrib.layers import fully_connected


from meta_learning import CONFIG

STABILITY = 1e-8


def build_policy_net(
    input_placeholder,
    ac_dim,
):

    with tf.variable_scope('hidden'):
        lstm1 = LSTMCell(CONFIG.num_lstm_units)

        # batch_size, num_units
        c = tf.placeholder(tf.float32, shape=[None, CONFIG.num_lstm_units], name='c_state')
        h = tf.placeholder(tf.float32, shape=[None, CONFIG.num_lstm_units], name='h_state')
        state_placeholders = tf.contrib.rnn.LSTMStateTuple(c, h)

    h1, new_state = tf.nn.dynamic_rnn(lstm1, input_placeholder, initial_state=state_placeholders)

    with tf.variable_scope('output'):
        output_layer = fully_connected(h1, ac_dim)

    return output_layer, state_placeholders, new_state


def build_value_net(
    input_placeholder,
):

    with tf.variable_scope('output'):
        output_layer = fully_connected(input_placeholder, 1)

    output_layer = tf.squeeze(output_layer, axis=-1)

    return output_layer


class Estimator(object):

    def __init__(self, obs_dim):

        # Batch, time, actions
        self.observations = tf.placeholder(shape=[None, None, obs_dim], name="observations", dtype=tf.float32)
        self.targets = tf.placeholder(shape=[None, None], name='targets', dtype=tf.float32)


class PolicyEstimator(Estimator):

    def __init__(self, obs_dim, ac_dim, gs):

        super(PolicyEstimator, self).__init__(obs_dim)

        self.gs = gs

        # Batch, time, actions
        self.actions = tf.placeholder(shape=[None, None, ac_dim], name="actions", dtype=tf.float32)

        with tf.variable_scope('policy_net'):

            action_means, self.state_placeholders, self.new_state = build_policy_net(self.observations, ac_dim)
            action_means_reg = self._regularize_action_means(action_means)
            action_std = tf.exp(tf.Variable(tf.zeros([ac_dim])))

            # Draw an action
            self.sampled = action_means_reg + tf.random_normal(tf.shape(action_means_reg)) * action_std

            if CONFIG.grad_reg:
                self.sampled = tf.clip_by_norm(self.sampled, CONFIG.grad_reg)

            # Calculate logprobability of multivariate gaussian
            z_scores = (self.actions - action_means_reg) / (action_std + STABILITY)
            consts = tf.log(tf.cast(ac_dim, tf.float32)) - 1/2 * tf.log(2 * np.pi)
            logprobs = consts - 1/2 * tf.reduce_sum(tf.square(z_scores), axis=-1)

            # Targets are advantages
            self.loss = -tf.reduce_mean(tf.multiply(logprobs, self.targets), name='loss')
            self.update_op = tf.train.AdamOptimizer().minimize(self.loss)

            tf.summary.scalar('loss', self.loss)

    def _regularize_action_means(self, action_means):

        self.mix_grad = tf.train.inverse_time_decay(
            learning_rate=1.0,
            global_step=self.gs,
            decay_steps=CONFIG.mix_halflife,
            decay_rate=0.5,
            name='policy_mix_rate')

        self.lr = tf.subtract(1.0, self.mix_grad, name='policy_learning_rate')

        grads = self.observations[:, :, 1:]

        tf.summary.scalar('lr', self.lr)
        tf.summary.scalar('mix_grad_rate', self.mix_grad)

        t_dict = {'grads': grads, 'actionus': action_means, 'diff_grad_actionus': grads - action_means}
        dist_dict = {'l1': 1, 'l2': 2, 'inf': np.inf}

        for t_name, t in t_dict.items():
            for norm_name, norm in dist_dict.items():
                d = tf.norm(t, ord=norm)
                tf.summary.scalar('_'.join([t_name, norm_name]), d)

        # This uses observations for the norm - this means that policy action vector norm doesn't make a difference
        # action_means_norm = tf.nn.l2_normalize(action_means, axis=-1) * tf.norm(grads, axis=-1, keepdims=True)
        # action_means_mix = self.mix_grad * grads + (1 - self.mix_grad) * action_means_norm

        # This uses the net prediction for the norm and doesn't perform well
        # grads_norm = tf.nn.l2_normalize(grads, axis=-1) * tf.norm(action_means, axis=-1, keepdims=True)
        # action_means_mix = self.mix_grad * grads_norm + (1 - self.mix_grad) * action_means

        norm = tf.add(self.mix_grad * tf.norm(grads, axis=-1, keepdims=True),
                      (1 - self.mix_grad) * tf.norm(action_means, axis=-1, keepdims=True),
                      name='action_l2_norm')

        action_means_norm = tf.nn.l2_normalize(action_means, axis=-1) * norm

        action_means_mix = self.mix_grad * grads + (1 - self.mix_grad) * action_means_norm

        action_means_mix *= self.lr

        return action_means_mix


class ValueEstimator(Estimator):

    def __init__(self, obs_dim):

        super(ValueEstimator, self).__init__(obs_dim)

        with tf.variable_scope('value_net'):

            self.predicted_values = build_value_net(self.observations)

            self.loss = tf.nn.l2_loss(self.predicted_values - self.targets)

            self.update_op = tf.train.AdamOptimizer().minimize(self.loss)

            tf.summary.scalar('loss', self.loss)
