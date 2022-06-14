import gym
import numpy as np
import random
import matplotlib.pyplot as plt

from ale_py.roms import SpaceInvaders
from ale_py import ALEInterface
from keras.optimizers import Adam

from dl_model import DLModel
from rl_agent import RLAgent

ale = ALEInterface()
ale.loadROM(SpaceInvaders)
env = gym.make('ALE/Pacman-v5', render_mode='human')
height, width, channels = env.observation_space.shape

#define actions
actions = env.action_space.n
#print(f'actions: {actions}')
env.unwrapped.get_action_meanings()
print(f'action meanings: {env.unwrapped.get_action_meanings()}')

for episode in range(0, 1):
    state = env.reset()
    done = False
    score = 0
    while not done:
        action = random.choice([0, 1, 2, 3, 4])
        n_state, reward, done, info = env.step(action)
        score += reward
    print(f'episode: {episode}\tscore: {score}')

env.close()

model = DLModel.build_model(height, width, channels, actions)
#model.summary()

dqn_agent = RLAgent.build_agent(model, actions)
dqn_agent.compile(Adam(learning_rate=1e-4))
dqn_agent.fit(env, nb_steps=20, visualize=False, verbose=2)

scores = dqn_agent.test(env, nb_episodes=2, visualize=False) #visualize=true
print(np.mean(scores.history['episode_reward']))
print(f'history: {scores.history}')

for i in scores.history.values():
    plt.plot(i['episode_reward'], i['nb_steps'])
plt.xlabel("Steps")
plt.ylabel("Rewards")
plt.title("Score-rewards")
plt.show()



