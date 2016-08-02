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
        delta = guessed_action_value - self.return_in
        clipped_delta = tf.clip_by_value(delta,-1,1)
        loss = tf.reduce_mean(tf.square(clipped_delta))
        self.debug = delta
        self.learning_rate = tf.placeholder(tf.float32)
        self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate, momentum=0.95, epsilon=0.01).minimize(loss)

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

    total_transitions = 0
    transitions_observations = []
    transitions_actions = []
    transitions_rewards = []
    transitions_isdone = []
    epsilon = 1
    finished_learning = False
    for episode in xrange(args.episodes):
        previous_observations = np.zeros((84,84,args.history))
        start_observation = env.reset()
        previous_observations[:,:,0] = preprocess(start_observation)
        epsilon = epsilon * args.epsilon_decay

        for step in xrange(args.maxframes + 1):
            env.render()
            action = None
            if step < args.history or random.uniform(0,1) < epsilon:
                action = env.action_space.sample()
            else:
                action = model.getAction(previous_observations)


            old_observation = previous_observations[:,:,min(step,args.history-1)]
            next_observation, reward, done, info = env.step(action)

            debug_predicted = np.amax(model.getValues(previous_observations),axis=1)
            print debug_predicted.shape
            print "e: %f took a step w/ action %d and got reward %d pred: %f" % (epsilon, action, reward, debug_predicted)

            # add current observation to the transitions
            transitions_observations.append(old_observation)
            transitions_actions.append(action)
            transitions_rewards.append(reward)
            transitions_isdone.append(done)
            total_transitions = total_transitions + 1
            # keep replay memory within a certain limit
            if total_transitions > args.memory_size:
                transitions.pop(0)

            # enough transitions stored in replay memory
            if total_transitions >= args.batchsize:
                observation_batch = np.zeros((args.batchsize,84,84,4))
                action_batch = np.zeros((args.batchsize,num_actions))
                next_observation_batch = np.zeros((args.batchsize,84,84,4))
                TQ_batch = np.zeros(args.batchsize)
                done_batch = np.zeros(args.batchsize)

                for batch_iter in xrange(args.batchsize):
                    replay_step = random.randint(3,total_transitions-2)
                    replay_observation = transitions_observations[replay_step-3:replay_step+1]
                    replay_observation_array = np.moveaxis(np.asarray(replay_observation),0,-1)
                    observation_batch[batch_iter] = replay_observation_array

                    action_onehot = np.zeros(num_actions)
                    action_onehot[transitions_actions[replay_step]] = 1.0
                    action_batch[batch_iter] = action_onehot

                    replay_next_observation = transitions_observations[replay_step-2:replay_step+2]
                    replay_next_observation_array = np.moveaxis(np.asarray(replay_next_observation),0,-1)
                    next_observation_batch[batch_iter] = replay_next_observation_array

                    TQ_batch[batch_iter] = transitions_rewards[batch_iter]
                    done_batch[batch_iter] = 1.0 - transitions_isdone[batch_iter]

                next_state_values = np.multiply(np.amax(model.getBatchValues(next_observation_batch),axis=1), done_batch)
                print "one train"
                print next_state_values
                TQ_batch = TQ_batch + args.discount*next_state_values
                print TQ_batch
                # TQ_batch = np.zeros(args.batchsize)

                model.update(observation_batch, action_batch, TQ_batch, 0.0001)


            if done:
                break

            # Preprocess next observation and add it to the previous_observations list
            # RBG 210 x 160 -> grayscale 84x84 as per the DQN paper

            next_observation = preprocess(next_observation)
            if step < args.history:
                previous_observations[:,:,step] = next_observation
            else:
                previous_observations[:,:,0:args.history-1] = previous_observations[:,:,1:args.history]
                previous_observations[:,:,args.history-1] = next_observation


def preprocess(observation):
    observation = np.dot(observation,[0.299, 0.587, 0.114])
    observation = misc.imresize(observation,(84,84),"bilinear")
    return observation
