import numpy as np
import argparse
import tensorflow as tf
import time
import random

# This is a very basic Q network, using a one-hidden-layer neural network to approximate Q(s,a).
# There are no fancy tricks emplyed, so the network sometimes scales up to infinity
# Doesn't perform very well on most environments, as it can't represent the return properly in the network.

class Q_Basic():
    def __init__(self,num_actions, num_observations, num_hidden):
        self.num_actions = num_actions
        self.num_observations = num_observations
        self.num_hidden = num_hidden

        self.observations_in = tf.placeholder(tf.float32, [None,num_observations])

        self.w1 = tf.Variable(tf.random_normal([num_observations, num_hidden], stddev=0.), name="w1")
        # self.b1 = tf.Variable(tf.random_normal([num_hidden], stddev=0.), name="b1")
        self.w2 = tf.Variable(tf.random_normal([num_hidden, num_actions], stddev=0.), name="w2")
        # self.b2 = tf.Variable(tf.random_normal([num_actions], stddev=0.), name="b2")

        self.h1 = tf.sigmoid(tf.matmul(self.observations_in, self.w1))
        self.estimated_values = tf.matmul(self.h1, self.w2)

        self.tvars = tf.trainable_variables()

        # one-hot matrix of which action was taken
        self.action_in = tf.placeholder(tf.float32,[None,num_actions])
        # vector of size [timesteps]
        self.return_in = tf.placeholder(tf.float32,[None])
        guessed_action_value = tf.reduce_sum(self.estimated_values * self.action_in, reduction_indices=1)
        loss = tf.nn.l2_loss(guessed_action_value - self.return_in)
        self.debug = loss
        self.optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

        self.sess = tf.Session()
        self.sess.run(tf.initialize_all_variables())

    def getAction(self, observation):
        values = self.getValues(observation)
        return np.argmax(values[0], axis=0)

    def getValues(self, observation):
        observation_reshaped = np.reshape(observation,(1,self.num_observations))
        return self.sess.run(self.estimated_values, feed_dict={self.observations_in: observation_reshaped})

    def getBatchValues(self,observation):
        return self.sess.run(self.estimated_values, feed_dict={self.observations_in: observation})

    def update(self, observation, action, reward):
        estimated, _ = self.sess.run([self.debug, self.optimizer], feed_dict={self.observations_in: observation, self.action_in: action, self.return_in: reward})
        return estimated

def learn(env, args):
    num_actions = int(env.action_space.n)
    num_observations, = env.observation_space.shape

    model = Q_Basic(num_actions, num_observations, args.hidden)

    for episode in xrange(args.episodes):
        observation = env.reset()
        transitions = []
        epsilon = max(0.2, ((100-episode) / 100.0))
        for frame in xrange(args.maxframes):
            if args.render:
                env.render()

            print model.getValues(observation)
            # epsilon-greedy actions
            action = model.getAction(observation)
            if random.uniform(0,1) < epsilon:
                action = env.action_space.sample()
            # action = 0

            old_observation = observation
            observation, reward, done, info = env.step(action)

            transitions.append((old_observation,action,reward,done,observation))

            # time.sleep(0.1)

            if done:
                break

        observation_history = np.zeros((0,num_observations))
        action_history = np.zeros((0,num_actions))
        TQ_history = np.array(())
        for transition in transitions:
            old_observation, action, reward, done, next_observation  = transition
            # TQ = reward + prediced reward of acting greedy on the next state
            TQ = reward + args.discount * np.amax(model.getValues(next_observation))
            # TQ = reward

            if done:
                TQ = -200

            old_observation_reshaped = np.reshape(old_observation,(1,num_observations))
            observation_history = np.append(observation_history,old_observation_reshaped,axis=0)

            action_onehot = np.zeros((1,num_actions))
            action_onehot[:,action] = 1.0
            action_history = np.append(action_history,action_onehot,axis=0)

            TQ_history = np.append(TQ_history,TQ)

        model.update(observation_history,action_history,TQ_history)
        print len(TQ_history)
        time.sleep(0.1)
