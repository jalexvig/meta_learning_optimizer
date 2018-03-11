import gym
import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import fully_connected


def get_square_data(side_len):

    def inner(batch_size):

        data = np.random.rand(batch_size, 2) * side_len
        labels = (data.sum(axis=1) > side_len).astype(int)

        return data, labels

    return inner


class BinaryClassifier(gym.Env):

    def __init__(self,
                 n_grad_steps_example=1,
                 n_episodes_weight_reset=0,
                 batch_size=64,
                 done_every=10,
                 data_gen=get_square_data(5),
                 writer: tf.summary.FileWriter=None):
        """
        Create binary classifier

        Args:
            n_grad_steps_example: Number gradient steps to take before getting new data.
            n_episodes_weight_reset: Number times to reset before reinitializing model's parameters.
            batch_size: Batch size of data to use.
            done_every: Report done every `done_every` number of steps.
            data_gen: Function that produces data (takes no arguments).
            writer: FileWriter used to write out summary statistics for tensorboard.
        """
        super(BinaryClassifier, self).__init__()

        self.batch_size = batch_size
        self.done_every = done_every
        self.n_grad_steps = n_grad_steps_example
        self.n_episodes_weight_reset = n_episodes_weight_reset
        self.writer = writer

        self.n_episodes_since_weights_reset = 0

        self._setup_net()

        self.data = data_gen

        self.num_iters = 0

        self.count = 0

    def _setup_net(self):

        with tf.variable_scope('model'):

            # TODO(jalex): Reduce size even more?

            self.inputs = tf.placeholder(tf.float32, name='inputs', shape=[None, 2])
            self.targets = tf.placeholder(tf.int32, name='targets', shape=[None])

            with tf.variable_scope('fc1'):
                self.fc1 = fully_connected(self.inputs, 4, activation_fn=tf.nn.leaky_relu)

            with tf.variable_scope('fc2'):
                self.fc2 = fully_connected(self.fc1, 2, activation_fn=tf.nn.leaky_relu)

            xent = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=self.targets,
                logits=self.fc2,
                name='xent'
            )

            self.loss = tf.reduce_mean(xent, name='loss')
            if self.writer:
                tf.summary.scalar('loss', self.loss)

            self._setup_gradients()

    def _setup_gradients(self):

        parameters = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        self.grads = tf.gradients(self.loss, parameters)

        num_W = [x.shape.num_elements() for x in parameters]
        total_num_W = sum(num_W)

        self.action_space = gym.spaces.Box(-np.inf, np.inf, shape=[total_num_W], dtype=np.float32)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=[total_num_W + 1], dtype=np.float32)

        self.processed_grads = tf.placeholder(tf.float32, name='processed_grads', shape=[total_num_W])

        apply_gradients_list = []

        start = 0
        for param, n in zip(parameters, num_W):
            param_processed_grads = self.processed_grads[start: start + n]
            op = param.assign_sub(tf.reshape(param_processed_grads, param.shape))
            apply_gradients_list.append(op)
            start += n

        self.apply_gradients_op = tf.group(*apply_gradients_list, name='apply_gradients')

    def step(self, action):
        """
        Apply the gradients passed in.

        Args:
            action: Modified gradients to apply

        Returns:
            tuple of form (observations (gradients), reward, done boolean, info dict)
        """

        self.num_iters += 1

        done = self.num_iters >= self.done_every

        self.sess.run(self.apply_gradients_op, {self.processed_grads: action[0][0]})

        if self.num_iters % self.n_grad_steps == 0:
            self.X, self.y = self.data(self.batch_size)

        obs, rew = self._get_obs_rew()

        return obs, rew, done, {}

    def set_session(self, sess):

        self.sess = sess

    def reset(self):

        self.X, self.y = self.data(self.batch_size)

        if self.n_episodes_weight_reset:

            if self.n_episodes_since_weights_reset % self.n_episodes_weight_reset == 0:
                model_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='model')
                self.sess.run(tf.variables_initializer(model_vars))
                self.n_episodes_since_weights_reset = 0

            self.n_episodes_since_weights_reset += 1

        self.num_iters = 0

        obs, _ = self._get_obs_rew()

        return obs

    def _get_obs_rew(self):

        feed = {self.inputs: self.X, self.targets: self.y}

        merged_summaries = tf.summary.merge_all(scope='model')
        rew, grads, summaries = self.sess.run([-self.loss, self.grads, merged_summaries], feed)

        if self.writer:
            self.writer.add_summary(summaries, global_step=self.count)
            self.writer.flush()

        self.count += 1

        obs = np.concatenate([[rew]] + [x.flatten() for x in grads])

        return obs, rew

    def render(self, mode='human'):
        pass
