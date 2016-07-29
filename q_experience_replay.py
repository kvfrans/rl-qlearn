import numpy as np
import argparse
import tensorflow as tf
import time
import random

# This is an improved version of Q_Basic. It has experience replay, and remembers previous transitions to train on again.
# In order to fix non-convergence problems, I manually put a reward of -200 when failing to reach 200 timesteps, and I
# run 10 supervised training updates after each episode.


# This model uses two Q networks: An old network that stays fixed for some number of episodes, while a new network is trained
# from a one-step-lookahead to the old network. The old network occasionally updates itself to the new network.
# EDIT: it turns out the two Q network setup doesn't really help, so I commented it out.

class Q_Experience_Replay():
    def __init__(self, name, sess, num_actions, num_observations, num_hidden):
        self.num_actions = num_actions
        self.num_observations = num_observations
        self.num_hidden = num_hidden
        self.sess = sess

        self.observations_in = tf.placeholder(tf.float32, [None,num_observations])

        with tf.variable_scope(name):
            self.w1 = tf.Variable(self.xavier_initializer([num_observations, num_hidden]), name="w1")
            self.b1 = tf.Variable(self.xavier_initializer([num_hidden]), name="b1")
            self.w2 = tf.Variable(self.xavier_initializer([num_hidden, num_hidden]), name="w2")
            self.b2 = tf.Variable(self.xavier_initializer([num_hidden]), name="b2")
            self.w3 = tf.Variable(self.xavier_initializer([num_hidden, num_actions]), name="w3")
            self.b3 = tf.Variable(self.xavier_initializer([num_actions]), name="b3")

        self.h1 = tf.sigmoid(tf.matmul(self.observations_in, self.w1) + self.b1)
        self.h2 = tf.sigmoid(tf.matmul(self.h1, self.w2) + self.b2)
        self.estimated_values = tf.matmul(self.h2, self.w3) + self.b3

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

        self.w1_placeholder = tf.placeholder(tf.float32, [num_observations, num_hidden])
        self.b1_placeholder = tf.placeholder(tf.float32, [num_hidden])
        self.w2_placeholder = tf.placeholder(tf.float32, [num_hidden, num_hidden])
        self.b2_placeholder = tf.placeholder(tf.float32, [num_hidden])
        self.w3_placeholder = tf.placeholder(tf.float32, [num_hidden, num_actions])
        self.b3_placeholder = tf.placeholder(tf.float32, [num_actions])
        self.w1_assign = self.w1.assign(self.w1_placeholder)
        self.b1_assign = self.b1.assign(self.b1_placeholder)
        self.w2_assign = self.w2.assign(self.w2_placeholder)
        self.b2_assign = self.b2.assign(self.b2_placeholder)
        self.w3_assign = self.w3.assign(self.w3_placeholder)
        self.b3_assign = self.b3.assign(self.b3_placeholder)

    def xavier_initializer(self, shape):
        dim_sum = np.sum(shape)
        if len(shape) == 1:
            dim_sum += 1
        bound = np.sqrt(6.0 / dim_sum)
        return tf.random_uniform(shape, minval=-bound, maxval=bound)

    def getAction(self, observation):
        values = self.getValues(observation)
        return np.argmax(values[0], axis=0)

    def getValues(self, observation):
        observation_reshaped = np.reshape(observation,(1,self.num_observations))
        return self.sess.run(self.estimated_values, feed_dict={self.observations_in: observation_reshaped})

    def getBatchValues(self,observation):
        return self.sess.run(self.estimated_values, feed_dict={self.observations_in: observation})

    def update(self, observation, action, reward, learning_rate):
        estimated, _ = self.sess.run([self.debug, self.optimizer], feed_dict={self.learning_rate: learning_rate, self.observations_in: observation, self.action_in: action, self.return_in: reward})
        return estimated

    def transferParams(self, otherModel):
        w1, b1, w2, b2 = self.sess.run([otherModel.w1, otherModel.b1, otherModel.w2, otherModel.b2])
        self.sess.run([self.w1_assign, self.b1_assign, self.w2_assign, self.b2_assign], feed_dict={self.w1_placeholder: w1, self.b1_placeholder: b1, self.w2_placeholder: w2, self.b2_placeholder: b2})

def learn(env, args):
    num_actions = int(env.action_space.n)
    num_observations, = env.observation_space.shape

    sess = tf.Session()

    lr = args.learningrate

    model = Q_Experience_Replay("main", sess, num_actions, num_observations, args.hidden)
    old_model = Q_Experience_Replay("old", sess, num_actions, num_observations, args.hidden)

    sess.run(tf.initialize_all_variables())

    old_model.transferParams(model)

    np.set_printoptions(precision=3,suppress=True)

    transitions = []
    epsilon = 1
    finished_learning = False
    for episode in xrange(args.episodes):

        observation = env.reset()

        epsilon = epsilon*args.epsilon_decay
        # epsilon = 0
        # totalreward = 0
        for frame in xrange(args.maxframes + 1):
            if args.render:
                env.render()

            if not finished_learning:
                print "%s : %s" % (np.array_str(observation), np.array_str(model.getValues(observation)))
            # epsilon-greedy actions

            action = model.getAction(observation)
            if random.uniform(0,1) < epsilon:
                action = env.action_space.sample()

            old_observation = observation
            observation, reward, done, info = env.step(action)

            if done and frame < args.maxframes-1:
                reward = args.shame
                transitions.append((old_observation,action,reward,True,observation))
            else:
                transitions.append((old_observation,action,reward,False,observation))

            if frame >= args.maxframes - 1:
                print "done"
                finished_learning = True

            if len(transitions) > args.memory_size:
                transitions.pop(0)

            if done:
                print "episode%d, lasted %d eps %f lr %f" % (episode, frame, epsilon, lr*10000)
                break

        if not finished_learning:
            if len(transitions) == args.memory_size:
                print "trained %f" % epsilon
                for x in xrange(args.training_iterations):
                    observation_history = np.zeros((0,num_observations))
                    action_history = np.zeros((0,num_actions))
                    TQ_history = np.array(())
                    for transition in random.sample(transitions, args.batchsize):
                        old_observation, action, reward, done, next_observation = transition

                        # print "%s w/action %d gives reward %f to %s which was worth %s" % (np.array_str(old_observation), action, reward, np.array_str(next_observation), np.array_str(model.getValues(next_observation)))

                        TQ = reward + args.discount * np.amax(model.getValues(next_observation))

                        if done:
                            # a big negative reward when failing to reach the goal: this solves convergence problems
                            TQ = args.shame

                        old_observation_reshaped = np.reshape(old_observation,(1,num_observations))
                        observation_history = np.append(observation_history,old_observation_reshaped,axis=0)

                        action_onehot = np.zeros((1,num_actions))
                        action_onehot[:,action] = 1.0
                        action_history = np.append(action_history,action_onehot,axis=0)

                        TQ_history = np.append(TQ_history,TQ)

                    model.update(observation_history,action_history,TQ_history,lr)
                    lr *= args.learningrate_decay
        # print "params"
