import tensorflow as tf
import numpy as np
from tensorflow.contrib.rnn import LSTMCell
from tensorflow.contrib.layers import fully_connected


from meta_learning import CONFIG, saved_weights

STABILITY = 1e-8


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

        if CONFIG.no_xover:
            observations = self._create_obs_no_xover()
            num_outputs = 1
        else:
            observations = self.observations
            num_outputs = ac_dim

        with tf.variable_scope('policy_net'):

            self.action_means = self._build(observations, num_outputs)

            if CONFIG.no_xover:
                dims = tf.shape(self.action_means)
                self.action_means = tf.reshape(self.action_means, [-1, dims[1], ac_dim])

            self.action_means_reg = self._regularize_action_means(self.action_means)
            self.action_stds = tf.exp(tf.Variable(tf.zeros([ac_dim])))

            # Draw an action
            self.sampled = self.action_means_reg + tf.random_normal(tf.shape(self.action_means_reg)) * self.action_stds

            if CONFIG.grad_reg:
                self.sampled = tf.clip_by_norm(self.sampled, CONFIG.grad_reg)

            # Calculate logprobability of multivariate gaussian
            self.z_scores = (self.actions - self.action_means_reg) / (self.action_stds + STABILITY)
            consts = tf.log(tf.cast(ac_dim, tf.float32)) - 1/2 * tf.log(2 * np.pi)
            self.logprobs = consts - 1/2 * tf.reduce_sum(tf.square(self.z_scores), axis=-1)

            # Targets are advantages
            self.loss = -tf.reduce_mean(tf.multiply(self.logprobs, self.targets), name='loss')
            self.update_op = tf.train.AdamOptimizer().minimize(self.loss)

            tf.summary.scalar('loss', self.loss)

    def _create_obs_no_xover(self):

        grads = tf.transpose(self.observations, [0, 2, 1])
        dims = tf.shape(grads)
        observations_no_xover = tf.reshape(grads, [dims[0] * dims[1], dims[2], 1], 'obs_no_xover')

        return observations_no_xover

    def _build(
        self,
        input_placeholder,
        num_outputs,
    ):

        with tf.variable_scope('hidden'):
            # forget_bias kwarg is necessary to load params
            self.lstm1 = LSTMCell(CONFIG.num_lstm_units, forget_bias=0.0)

            shape = [None, CONFIG.num_lstm_units]

            # batch_size, num_units
            c = tf.placeholder(tf.float32, shape=shape, name='c_state')
            h = tf.placeholder(tf.float32, shape=shape, name='h_state')
            self.state_placeholders = tf.contrib.rnn.LSTMStateTuple(c, h)

        h1, self.new_state = tf.nn.dynamic_rnn(self.lstm1, input_placeholder, initial_state=self.state_placeholders)

        with tf.variable_scope('output'):
            output_layer = fully_connected(h1, num_outputs)

        if CONFIG.load_params_torch:

            kernel_tf, bias_tf = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='policy_net/rnn')

            kernel_torch, bias_torch = saved_weights.get_lstm_kernel_bias_torch(CONFIG.load_params_torch)

            self.assign_ops = [
                tf.assign(kernel_tf, tf.constant(kernel_torch)),
                tf.assign(bias_tf, tf.constant(bias_torch)),
            ]

        return output_layer

    def _regularize_action_means(self, action_means):
        """
        Mix action means with original gradients to stabilize training.

        Args:
            action_means: Action means from policy.

        Returns:
            Mixture.
        """

        if not CONFIG.mix_start:
            return action_means

        self.mix_grad = tf.train.inverse_time_decay(
            learning_rate=CONFIG.mix_start,
            global_step=self.gs,
            decay_steps=CONFIG.mix_halflife,
            decay_rate=0.5,
            name='policy_mix_rate')

        self.lr = tf.subtract(1.0, self.mix_grad, name='policy_learning_rate')

        tf.summary.scalar('lr', self.lr)
        tf.summary.scalar('mix_grad_rate', self.mix_grad)

        # self.observations are gradients

        t_dict = {'grads': self.observations, 'actionus': action_means,
                  'diff_grad_actionus': self.observations - action_means}
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

        norm = tf.add(self.mix_grad * tf.norm(self.observations, axis=-1, keepdims=True),
                      self.lr * tf.norm(action_means, axis=-1, keepdims=True),
                      name='action_l2_norm')

        action_means_norm = tf.nn.l2_normalize(action_means, axis=-1) * norm

        action_means_mix = self.mix_grad * self.observations + (1 - self.mix_grad) * action_means_norm

        action_means_mix *= self.lr

        return action_means_mix


class ValueEstimator(Estimator):

    def __init__(self, obs_dim):

        super(ValueEstimator, self).__init__(obs_dim)

        with tf.variable_scope('value_net'):

            self.predicted_values = self._build(self.observations)

            self.loss = tf.nn.l2_loss(self.predicted_values - self.targets)

            self.update_op = tf.train.AdamOptimizer().minimize(self.loss)

            tf.summary.scalar('loss', self.loss)

    def _build(
        self,
        input_placeholder,
    ):

        with tf.variable_scope('output'):
            output_layer = fully_connected(input_placeholder, 1)

        output_layer = tf.squeeze(output_layer, axis=-1)

        return output_layer
