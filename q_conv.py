import numpy as np
import argparse
import tensorflow as tf
import time
import random

import matplotlib.pyplot as plt
from scipy import misc

# This is an improved version of Q_Basic. It has experience replay, and remembers previous transitions to train on again.
# In order to fix non-convergence problems, I manually put a reward of -200 when failing to reach 200 timesteps, and I
# run 10 supervised training updates after each episode.


# This model uses two Q networks: An old network that stays fixed for some number of episodes, while a new network is trained
# from a one-step-lookahead to the old network. The old network occasionally updates itself to the new network.
# EDIT: it turns out the two Q network setup doesn't really help, so I commented it out.

class Q_Conv():
    def __init__(self, name, sess, num_actions):
        self.num_actions = num_actions
        self.sess = sess

        self.observations_in = tf.placeholder(tf.float32, [None,84,84,4])
        h1 = tf.nn.relu(self.conv2d(self.observations_in, 4, 16, 8, 4, "h1")) # batchsize x 21 x 21 x 16
        h2 = tf.nn.relu(self.conv2d(h1, 16, 32, 4, 2, "h2")) # batchsize x 11 x 11 x 32
        flattened_h2 = tf.reshape(h2, [-1, 11 * 11 * 32])
        h3 = tf.nn.relu(self.fully_connected(flattened_h2, 11*11*32, 256,"h3"))
        self.estimated_values = self.fully_connected(h3, 256, num_actions,"output")

        self.tvars = tf.trainable_variables()

        # one-hot matrix of which action was taken
        self.action_in = tf.placeholder(tf.float32,[None,num_actions])
        # vector of size [timesteps]
        self.return_in = tf.placeholder(tf.float32,[None])
        guessed_action_value = tf.reduce_sum(self.estimated_values * self.action_in, reduction_indices=1)
        loss = tf.nn.l2_loss(guessed_action_value - self.return_in)
        self.debug = loss
        self.learning_rate = tf.placeholder(tf.float32)
        self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(loss)

    def conv2d(self, x, inputFeatures, outputFeatures, filtersize, stride, name):
        with tf.variable_scope(name):
            w = tf.get_variable("w",[filtersize,filtersize,inputFeatures, outputFeatures], initializer=tf.truncated_normal_initializer(stddev=0.02))
            b = tf.get_variable("b",[outputFeatures], initializer=tf.constant_initializer(0.0))
            conv = tf.nn.conv2d(x, w, strides=[1,stride,stride,1], padding="SAME") + b
            return conv

    def fully_connected(self, x, inputFeatures, outputFeatures, name):
        with tf.variable_scope(name):
            w = tf.get_variable("w",[inputFeatures,outputFeatures], initializer=tf.truncated_normal_initializer(stddev=0.02))
            b = tf.get_variable("b",[outputFeatures], initializer=tf.constant_initializer(0.0))
            fc = tf.matmul(x, w) + b
            return fc

    def getAction(self, observation):
        values = self.getValues(observation)
        return np.argmax(values[0], axis=0)

    def getValues(self, observation):
        observation_reshaped = np.reshape(observation,(1,84,84,4))
        return self.sess.run(self.estimated_values, feed_dict={self.observations_in: observation_reshaped})

    def getBatchValues(self,observation):
        return self.sess.run(self.estimated_values, feed_dict={self.observations_in: observation})

    def update(self, observation, action, reward, learning_rate):
        estimated, _ = self.sess.run([self.debug, self.optimizer], feed_dict={self.learning_rate: learning_rate, self.observations_in: observation, self.action_in: action, self.return_in: reward})
        return estimated
    #
    # def transferParams(self, otherModel):
    #     w1, b1, w2, b2 = self.sess.run([otherModel.w1, otherModel.b1, otherModel.w2, otherModel.b2])
    #     self.sess.run([self.w1_assign, self.b1_assign, self.w2_assign, self.b2_assign], feed_dict={self.w1_placeholder: w1, self.b1_placeholder: b1, self.w2_placeholder: w2, self.b2_placeholder: b2})

def learn(env, args):
    num_actions = int(env.action_space.n)
    observation_space = env.observation_space.shape

    sess = tf.Session()

    lr = args.learningrate

    model = Q_Conv("main", sess, num_actions)

    sess.run(tf.initialize_all_variables())

    transitions = []
    epsilon = 1
    finished_learning = False
    for episode in xrange(args.episodes):
        previous_observations = []
        partial_observation = env.reset()

        for step in xrange(args.maxframes + 1):
            env.render()
            action = None
            if step >= args.history:
                full_observation = np.moveaxis(np.asarray(previous_observations),0,-1)
                action = model.getAction(full_observation)
            else:
                action = env.action_space.sample()

            partial_observation, reward, done, info = env.step(action)

            # RBG 210 x 160 -> grayscale 84x84 as per the DQN paper
            partial_observation = np.dot(partial_observation,[0.299, 0.587, 0.114])
            partial_observation = misc.imresize(partial_observation,(84,84),"bilinear")

            previous_observations.append(partial_observation)

            # once enough steps passed to fill up history
            if step >= args.history:
                previous_observations.pop(0)

                # train
                observation_history = np.zeros((0,84,84,4))
                action_history = np.zeros((0,num_actions))
                TQ_history = np.array(())

                for _ in xrange(args.batchsize):
                    index = random.


            if done:
                break
