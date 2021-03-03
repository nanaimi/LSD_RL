import os
import gym
import time
import torch
import pickle
import numpy as np
import glog as log
import subprocess

import gym_unrealcv
import real_lsd

from torch.distributions import Categorical

'''---------------------------Helper functions--------------------'''
def save_obj(obj, filename):
    dir = 'data'
    # path = os.getcwd()
    PATH = '/media/scratch1/nasib'

    if dir not in os.listdir(PATH):
        PATH =  os.path.join(PATH, dir)
        os.mkdir(PATH)
    else:
        PATH =  os.path.join(PATH, dir)

    abs_file_path = PATH + '/' + filename + '.pkl'
    with open(abs_file_path, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

    return filename

'''---------------------------------------------------------------'''

# Set to INFO for debugging
log.setLevel("WARN")

# initialise environment
env = gym.make('MyUnrealLand-cpptestFloorGood-DiscreteHeightFeatures-v0')

print("Observation Space:", env.observation_space, "dimension of observation:", env.observation_space.shape[0])

num_inputs  = env.observation_space.shape[0]

print("Action Space:", env.action_space, "Number of actions:", env.action_space.n)

num_outputs = env.action_space.n

probabilities = np.zeros(num_outputs)
probabilities[:] = np.float32(1/27)
probabilities = torch.tensor(probabilities)

action_dist = Categorical(probabilities)
sample = action_dist.sample()
print("sampled action: {}".format(sample))
