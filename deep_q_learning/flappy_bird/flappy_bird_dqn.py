import random
import numpy as np
import json
import flappy_bird_gym
from collections import deque
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Sequential, save_model, load_model
from tensorflow.keras.optimizers import RMSprop


def NeuralNet(input_shape, output_shape):
    model = Sequential()
    model.add(Dense(512, input_shape=input_shape, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(256, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(64, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(output_shape, activation='linear', kernel_initializer='he_uniform'))
    model.compile(loss='mse', optimizer=RMSprop(lr=0.0001, rho=0.95, epsilon=0.01), metrics=['accuracy'])
    print('[INFO] Model Summary...')
    print(model.summary())

    return model


class DQNAgent:

    def __init__(self, params_file):
        self.env = flappy_bird_gym.make("FlappyBird-v0")
        self.state_space = self.env.observation_space.shape[0]
        self.action_space = self.env.action_space.n
        self.params_file = params_file
        self.params = self.read_params()
        self.memory = deque(maxlen=2000)

        self.episodes = self.params['training']['num_episodes']
        self.gamma = self.params['training']['discount_rate']
        self.epsilon = self.params['training']['init_exploration_rate']
        self.epsilon_decay = self.params['training']['exploration_decay_rate']
        self.epsilon_min = self.params['training']['min_exploration_rate']
        self.batch_number = 64  # 16, 32, 64, 128

        self.train_start = 1000
        self.jump_prob = 0.01
        self.model = NeuralNet(input_shape=(self.state_space,), output_shape=self.action_space)

    def read_params(self):
        print(f'[INFO] Reading parameters from {self.params_file}')
        with open(self.params_file) as f:
            params = json.load(f)

        return params

    def act(self, state):
        if np.random.random() > self.epsilon:
            return np.argmax(self.model.predict(state))
        return 1 if np.random.random() < self.jump_prob else 0

    def learn(self):
        if len(self.memory) < self.train_start:
            return

        # experience replay
        minibatch = random.sample(self.memory, min(len(self.memory), self.batch_number))
        state = np.zeros((self.batch_number, self.state_space))
        next_state = np.zeros((self.batch_number, self.state_space))

        action, reward, done = [], [], []

        for i in range(self.batch_number):
            state[i] = minibatch[i][0]
            action.append(minibatch[i][1])
            reward.append(minibatch[i][2])
            next_state[i] = minibatch[i][3]
            done.append(minibatch[i][4])

        target = self.model.predict(state)
        next_target = self.model.predict(next_state)

        for i in range(self.batch_number):
            if done[i]:
                target[i][action[i]] = reward[i]
            else:
                target[i][action[i]] = reward[i] + self.gamma * np.amax(next_target[i])

        self.model.fit(state, target, batch_size=self.batch_number, verbose=0)

    def train(self, render=False):
        for i in range(self.episodes):
            state = self.env.reset()
            state = np.reshape(state, [1, self.state_space])
            done = False
            score = 0
            self.epsilon = self.epsilon * self.epsilon_decay if self.epsilon * self.epsilon_decay > self.epsilon_min else self.epsilon_min

            while not done:
                if render:
                    self.env.render()
                action = self.act(state)
                next_state, reward, done, info = self.env.step(action)

                next_state = np.reshape(next_state, [1, self.state_space])
                score += 1

                if done:
                    reward += -100

                self.memory.append((state, action, reward, next_state, done))
                state = next_state

                if done:
                    print(f'[INFO] Episode {i} got score {reward} at epsilon {round(self.epsilon, 2)}')
                    if score >= 1000:
                        self.model.save('flappy_bird_dqn.h5')
                        return

                self.learn()

    def perform(self):
        self.model = load_model('flappy_bird_dqn.h5')
        while 1:
            state = self.env.reset()
            state = np.reshape(state, [1, self.state_space])
            done = False
            score = 0

            while not done:
                self.env.render()
                action = np.argmax(self.model.predict(state))
                next_state, reward, done, info = self.env.step(action)
                state = np.reshape(next_state, [1, self.state_space])
                score += 1

                print("Current Score: {}".format(score))

                if done:
                    print('DEAD')
                    break


if __name__ == '__main__':
    agent = DQNAgent('params.json')
    agent.train()
    agent.perform()
