import gym
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras.optimizers import Adam
from rl.agents import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy


def build_model(height, width, channels, actions):
    model = Sequential()
    model.add(Conv2D(32, (8, 8), strides=(4, 4), activation='relu', input_shape=(3, height, width, channels)))
    model.add(Conv2D(64, (4, 4), strides=(2, 2), activation='relu'))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(actions, activation='linear'))

    return model


def build_agent(model, actions):
    policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1, value_min=0.1, value_test=0.2,
                                  nb_steps=10000)
    memory = SequentialMemory(limit=2000, window_length=3)

    dqn = DQNAgent(model, policy, memory=memory, enable_dueling_network=True, dueling_type='avg', nb_actions=actions,
                   nb_steps_warmup=1000)
    return dqn


def test_environment(env, episodes):
    for episode in range(episodes):
        state = env.reset()
        done = False
        score = 0

        while not done:
            env.render()
            state, reward, done, info = env.step(env.action_space.sample())
            score += reward
        print(f'Episode {episode} got score {score}.')
    env.close()


def train_agent(dqn_agent, environment, train_episodes, lr):

    dqn_agent.compile(Adam(lr=lr))
    dqn_agent.fit(environment, nb_steps=train_episodes)

    return dqn_agent


def test_agent(dqn_agent, environment, test_episodes):

    scores = dqn_agent.test(environment, nb_episodes=test_episodes, visualize=True)
    print(np.mean(scores.history['episode_reward']))

    return scores


def load_agent(weights_path):
    env = gym.make('SpaceInvaders-v0')

    height, width, channels = env.observation_space.shape
    actions = env.action_space.n

    model = build_model(height, width, channels, actions)

    dqn = build_agent(model=model, actions=actions)

    dqn.load_weights(weights_path)

    return dqn


def main(train_episodes=40000, test_episodes=10, lr=0.0001):
    env = gym.make('SpaceInvaders-v0')

    height, width, channels = env.observation_space.shape
    actions = env.action_space.n

    model = build_model(height, width, channels, actions)

    dqn = build_agent(model=model, actions=actions)

    dqn = train_agent(dqn, env, train_episodes, lr)

    scores = test_agent(dqn, env, test_episodes)

    dqn.save_weights('models\\space_invaders_dqn.h5f')

    return dqn, scores


if __name__ == '__main__':
    main()

