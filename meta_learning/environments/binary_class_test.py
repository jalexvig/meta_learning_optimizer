import tensorflow as tf

from meta_learning.environments import BinaryClassifier


LR = 0.01

env_kwargs = {
    'reset_every': 20,
    'writer': tf.summary.FileWriter('../saved/relu')
}

env = BinaryClassifier(**env_kwargs)

trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
for var in trainable_vars:
    tf.summary.histogram(var.name, var)

with tf.Session() as sess:
    env.set_session(sess)

    sess.run(tf.global_variables_initializer())

    obs = env.reset()

    for i in range(2000):
        obs, rew, done, info = env.step(LR * obs[None, None, 1:])
        if done:
            obs = env.reset()
