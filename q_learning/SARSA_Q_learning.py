import gym
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output
import random
import time
import json


class SARSA_Q_Reinforcer:

    def __init__(self, environment, params_json, q_table=None):
        self.rewards_all_episodes = []
        self.env = gym.make(environment)
        self.params_file = params_json
        self.params = self.read_params()
        self.q_table = q_table

    def read_params(self):
        print(f'[INFO] Reading parameters from {self.params_file}')
        with open(self.params_file) as f:
            params = json.load(f)

        return params

    def test_environment(self, render=False):
        episodes = 10

        for episode in range(episodes):
            state = self.env.reset()
            done = False
            score = 0

            while not done:
                if render:
                    self.env.render()
                state, reward, done, info = self.env.step(self.env.action_space.sample())
                score += reward
                clear_output(wait=True)
            print(f'[INFO] Episode {episode} got score {score}')
        print('[INFO] Environment Tested successfully')
        self.env.close()

    def epsilon_greedy(self, state, exploration_rate):

        exploration_threshold = random.uniform(0, 1)
        if exploration_threshold > exploration_rate:
            action = np.argmax(self.q_table[state, :])
        else:
            action = self.env.action_space.sample()

        return action

    def Q_learning_train(self):
        # defining parameters
        actions = self.env.action_space.n
        state = self.env.observation_space.n

        if not self.q_table:
            print('[INFO] Initializing Q table with zeroes')
            self.q_table = np.zeros((state, actions))
        else:
            if self.q_table.shape != (actions, state):
                print("[ERROR] Q Table Shape does not match environment")
                raise ValueError

        train_params = self.params["training"]
        num_episodes = train_params["num_episodes"]
        max_steps_per_episodes = train_params["max_steps_per_episodes"]

        learning_rate = train_params["learning_rate"]
        discount_rate = train_params["discount_rate"]

        exploration_rate = train_params["init_exploration_rate"]
        max_exploration_rate = train_params["max_exploration_rate"]
        min_exploration_rate = train_params["min_exploration_rate"]
        exploration_decay_rate = train_params["exploration_decay_rate"]

        self.rewards_all_episodes = []

        print('[INFO] Starting training...')
        for episode in range(num_episodes):

            state = self.env.reset()
            done = False
            rewards_current_episode = 0

            for step in range(max_steps_per_episodes):

                action = self.epsilon_greedy(state, exploration_rate)

                new_state, reward, done, info = self.env.step(action)

                self.q_table[state, action] = self.q_table[state, action] * (1 - learning_rate) + learning_rate * \
                                              (reward + discount_rate * np.max(self.q_table[new_state, :]))

                state = new_state
                rewards_current_episode += reward

                if done:
                    break

            exploration_rate = min_exploration_rate + (max_exploration_rate - min_exploration_rate) * np.exp(
                -exploration_decay_rate * episode)

            self.rewards_all_episodes.append(rewards_current_episode)

        print("[INFO] Training Complete")

    def plot_training_rewards(self):
        rewards_per_thousand_episodes = np.split(np.array(self.rewards_all_episodes),
                                                 len(self.rewards_all_episodes) / 1000)
        count = 1000

        count_list = []
        reward_list = []

        print("Avg per thousand episodes")
        for r in rewards_per_thousand_episodes:
            print(f"{count} : {sum(r / 1000)}")
            count_list.append(count)
            reward_list.append(sum(r / 1000))
            count += 1000

        plt.plot(count_list, reward_list)
        plt.xlabel("Episodes")
        plt.ylabel("Average Reward")
        plt.savefig("training_rewards.png")
        plt.show()

    def Q_table_test(self, no_of_episodes=3, max_steps_per_episodes=100):
        for episode in range(no_of_episodes):
            state = self.env.reset()
            done = False
            print(f"On episode {episode}")
            time.sleep(1)

            for step in range(max_steps_per_episodes):
                clear_output(wait=True)
                self.env.render()
                time.sleep(0.4)
                action = np.argmax(self.q_table[state, :])

                new_state, reward, done, info = self.env.step(action)

                if done:
                    clear_output(wait=True)
                    self.env.render()
                    if reward == 1:
                        print("Reached Goal")
                        time.sleep(2)
                        clear_output(wait=True)
                    else:
                        print("Failed")
                        time.sleep(2)
                        clear_output(wait=True)

                    break

                state = new_state
        self.env.close()


if __name__ == '__main__':
    q_learner = SARSA_Q_Reinforcer(environment='FrozenLake-v0', params_json='params.json')
    q_learner.test_environment()
    q_learner.Q_learning_train()
    q_learner.plot_training_rewards()
