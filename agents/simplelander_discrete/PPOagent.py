import math
import random

import gym
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal

# from IPython.display import clear_output
import matplotlib.pyplot as plt


class PPOAgent():
    def __init__(self,
                    num_inputs,
                    num_actions, # number of outputs of actor network
                    hidden_size,
                    lr,
                    num_steps,
                    mini_batch_size,
                    ppo_epochs,
                    threshold_reward=(-200),
                    std=0.0):

        # Class attributes
        self.num_actions = num_actions
        self.num_inputs = num_inputs

        self.model = ActorCritic(num_inputs, num_actions, hidden_size).to(device)
        self.optimizer = optim.Adam(model.parameters(), lr=lr)


def init_weights(m):
    if isinstance(m, nn.Linear):
        # Fills the the weight Tensor with values drawn from the normal distribution
        nn.init.normal_(m.weight, mean=0., std=0.1)
        # Fills the bias Tensor with the constant value
        nn.init.constant_(m.bias, 0.1)

def act(self, action_probabilities):
        """ example:
        pred = np.array([0.05, 0.85, 0.1])
        action_size = 3
        np.random.choice(a, p=pred)
        result>>> 1, because it have the highest probability to be taken
        """
        # Use the network to predict the next action to take, using the model
        action = np.random.choice(self.num_actions, p=action_probabilities)
        action_onehot = np.zeros([self.action_size])
        action_onehot[action] = 1
        return action, action_onehot

def ppo_iter(mini_batch_size, states, actions, log_probs, returns, advantage):
    batch_size = states.size(0)
    for _ in range(batch_size // mini_batch_size):
        rand_ids = np.random.randint(0, batch_size, mini_batch_size)
        yield states[rand_ids, :], actions[rand_ids, :], log_probs[rand_ids, :], returns[rand_ids, :], advantage[rand_ids, :]


def ppo_update(ppo_epochs, mini_batch_size, states, actions, log_probs, returns, advantages, clip_param=0.2):
    for _ in range(ppo_epochs):
        for state, action, old_log_probs, return_, advantage in ppo_iter(mini_batch_size, states, actions, log_probs, returns, advantages):
            action_probabilities, value = model(state)
            entropy = dist.entropy().mean()



            new_log_probs = dist.log_prob(action)

            ratio = (new_log_probs - old_log_probs).exp()
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantage

            actor_loss  = - torch.min(surr1, surr2).mean()
            critic_loss = (return_ - value).pow(2).mean()

            loss = 0.5 * critic_loss + actor_loss - 0.001 * entropy

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()





### INITIALISE ALL



if __name__ == '__main__':
