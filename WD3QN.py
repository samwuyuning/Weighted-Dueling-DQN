import tensorflow as tf
import tensorlayer as tl
from collections import deque
import numpy as np
import gym
import random
import matplotlib.pyplot as plt


class Double_DQN():
    def __init__(self):
        self.env = gym.make('CartPole-v0')
        self.input_dim = self.env.observation_space.shape[0]

        self.Q_network = self.get_model()
        self.Q_network.train()
        self.target_Q_network = self.get_model()
        self.target_Q_network.eval()

        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01

        self.memory = deque(maxlen=200000)
        self.batch = 32
        self.gamma = 0.95
        self.learning_rate = 5e-4
        self.opt = tf.optimizers.Adam(self.learning_rate)
        self.is_rend = False

    '''
    def get_model(self):

        self.input = tl.layers.Input(shape=[None,self.input_dim])
        self.h1 = tl.layers.Dense(32, tf.nn.relu, W_init=tf.initializers.GlorotUniform())(self.input)
        self.h2 = tl.layers.Dense(16, tf.nn.relu, W_init=tf.initializers.GlorotUniform())(self.h1)
        self.output = tl.layers.Dense(2,act=None, W_init=tf.initializers.GlorotUniform())(self.h2)
        return tl.models.Model(inputs=self.input,outputs=self.output)
    '''
    # dueling architecture

    def get_model(self):

        input = tl.layers.Input(shape=[None, self.input_dim])
        h1 = tl.layers.Dense(16, tf.nn.relu, W_init=tf.initializers.GlorotUniform())(input)
        h2 = tl.layers.Dense(16, tf.nn.relu, W_init=tf.initializers.GlorotUniform())(h1)

        svalue = tl.layers.Dense(2, )(h2)
        avalue = tl.layers.Dense(2, )(h2)

        mean = tl.layers.Lambda(lambda x: tf.reduce_mean(x, axis=1, keepdims=True))(avalue)  # lambda layer
        advantage = tl.layers.ElementwiseLambda(lambda x, y: x - y)([avalue, mean])  # a - avg(a)

        output = tl.layers.ElementwiseLambda(lambda x, y: x + y)([svalue, advantage])
        return tl.models.Model(inputs=input, outputs=output)


    def update_epsilon(self):
        if self.epsilon >= self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_Q(self):
        for i, target in zip(self.Q_network.trainable_weights, self.target_Q_network.trainable_weights):
            target.assign(i)

    def remember(self, s, a, s_, r, done):
        data = (s, a, s_, r, done)
        self.memory.append(data)

    def process_data(self):

        data = random.sample(self.memory, self.batch)
        s = np.array([d[0] for d in data])
        a = [d[1] for d in data]
        s_ = np.array([d[2] for d in data])
        r = [d[3] for d in data]
        done = [d[4] for d in data]

        # Nature DQN
        '''
        target_Q = np.max(self.target_Q_network(np.array(s_,dtype='float32'))) 
        target = target_Q * self.gamma + r
        '''
        # Double DQN Target Q value
        y = self.Q_network(np.array(s, dtype='float32'))
        y = y.numpy()
        Q1 = self.target_Q_network(np.array(s_, dtype='float32'))
        Q2 = self.Q_network(np.array(s_, dtype='float32'))
        next_action = np.argmax(Q2, axis=1)
        next_action1 = np.argmin(Q2, axis=1)

        for i, (_, a, _, r, done) in enumerate(data):
            if done:
                target = r
            else:
                target = r + self.gamma * Q1[i][next_action[i]]
                #Weighted estimator
                c = 10
                eta = abs(Q2[i][next_action[i]]-Q2[i][next_action1[i]])/(c+abs(Q2[i][next_action[i]]-Q2[i][next_action1[i]]))
                target = r + self.gamma * (eta * Q1[i][next_action[i]]+(1-eta) * Q2[i][next_action[i]])
            target = np.array(target, dtype='float32')

            y[i][a] = target
        return s, y

    def update_Q_network(self):
        s, y = self.process_data()
        with tf.GradientTape() as tape:
            Q = self.Q_network(np.array(s, dtype='float32'))
            loss = tl.cost.mean_squared_error(Q, y)
        grads = tape.gradient(loss, self.Q_network.trainable_weights)
        self.opt.apply_gradients(zip(grads, self.Q_network.trainable_weights))
        return loss

    def get_action(self, s):
        if np.random.rand() >= self.epsilon:
            q = self.Q_network(np.array(s, dtype='float32').reshape([-1, 4]))
            a = np.argmax(q)
            return a
        else:
            a = random.randint(0, 1)
            return a

    def train(self, episode):
        step = 0
        rend = 0
        acc = []
        for ep in range(episode):

            s = self.env.reset()
            total_reward = 0
            total_loss = []
            loss = 0

            while True:
                if self.is_rend: self.env.render()

                a = self.get_action(s)
                s_, r, done, _ = self.env.step(a)
                total_reward += r
                step += 1

                self.remember(s, a, s_, r, done)
                s = s_

                if len(self.memory) > self.batch:
                    loss = self.update_Q_network()
                    total_loss.append(loss)
                    if (step + 1) % 5 == 0:
                        self.update_epsilon()
                        self.update_target_Q()

                if done:
                    acc.append(total_reward)
                    print('EP:%i,  total_rewards:%f,   epsilon:%f, loss:%f' % (ep, total_reward, self.epsilon, np.mean(loss)))
                    break

            if total_reward >= 200:
                rend += 1
                if rend == 5:
                    self.is_rend = True

        plt.plot(np.array(acc), c='k', label='WD3QN')
        plt.xlabel('Number of episodes')
        plt.ylabel('Reward')
        plt.legend()
        plt.show()


if __name__ == '__main__':
    ddqn = Double_DQN()
    ddqn.train(600)
