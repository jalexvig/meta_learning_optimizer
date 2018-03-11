import gym
import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import fully_connected, max_pool2d


class MNIST(gym.Env):

    def __init__(self, n_grad_steps_example=10, n_episodes_weight_reset=1, batch_size=64, done_every=400):
        super(MNIST, self).__init__()

        self.batch_size = batch_size
        self.done_every = done_every
        self.n_grad_steps = n_grad_steps_example
        self.n_episodes_weight_reset = n_episodes_weight_reset

        self.n_episodes_since_weights_reset = 0

        self._setup_net()

        self.mnist = tf.contrib.learn.datasets.load_dataset("mnist")

        self.num_iters = 0

    def _setup_net(self):

        with tf.variable_scope('mnist'):

            # TODO(jalex): Reduce size even more?

            self.inputs = tf.placeholder(tf.float32, name='inputs', shape=[None, 784])
            self.targets = tf.placeholder(tf.int32, name='targets', shape=[None])

            inputs = tf.reshape(self.inputs, [-1, 28, 28, 1])
            pooled = max_pool2d(inputs, kernel_size=2, stride=2)
            pooled = tf.reshape(pooled, [-1, 196])

            self.fc1 = fully_connected(pooled, 196)

            self.fc2 = fully_connected(self.fc1, 10)

            xent = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=self.targets,
                logits=self.fc2,
                name='xent'
            )

            self.loss = tf.reduce_mean(xent, name='loss')

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

    def apply_gradients(self, flattened_grads):

        self.sess.run(self.apply_gradients_op, {self.processed_grads: flattened_grads[0]})

    def step(self, action):

        self.num_iters += 1

        done = self.num_iters >= self.done_every

        self.apply_gradients(action)

        if self.num_iters % self.n_grad_steps == 0:
            self.X, self.y = self.mnist.train.next_batch(self.batch_size)

        obs, rew = self._get_obs_rew()

        return obs, rew, done, {}

    def set_session(self, sess):

        self.sess = sess

    def reset(self):

        self.X, self.y = self.mnist.train.next_batch(self.batch_size)

        if self.n_episodes_since_weights_reset % self.n_episodes_weight_reset == 0:
            mnist_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='mnist')
            self.sess.run(tf.variables_initializer(mnist_vars))
            self.n_episodes_since_weights_reset = 0

        self.n_episodes_since_weights_reset += 1

        self.num_iters = 0

        obs, _ = self._get_obs_rew()

        return obs

    def _get_obs_rew(self):

        rew, grads = self.sess.run([-self.loss, self.grads], {self.inputs: self.X, self.targets: self.y})
        obs = np.concatenate([[rew]] + [x.flatten() for x in grads])

        return obs, rew

    def render(self, mode='human'):
        pass
