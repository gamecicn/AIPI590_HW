

import sys
import gym
import numpy as np

import sb3.stable_baselines3
from sb3.stable_baselines3.a2c.a2c import A2C

# sys.path.append(r'./sb3')
# sys.path.append(r'./sb3/stable_baselines3')

env = gym.make('CartPole-v1')
model = A2C('MlpPolicy', env, verbose=1)

model.learn(total_timesteps=1000)



