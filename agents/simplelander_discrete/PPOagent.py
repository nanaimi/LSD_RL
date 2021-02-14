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
# Import network
from model import ActorCritic

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
                    std=0.0,
                    device="cpu"):

        # Class attributes
        self.num_inputs = num_inputs
        self.num_actions = num_actions

        self.model = ActorCritic(num_inputs, num_actions, hidden_size).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def act(self, action_probabilities):
        """
        Use the network to predict the next action to take, using the model
        example:
        pred = np.array([0.05, 0.85, 0.1])
        action_size = 3
        np.random.choice(a, p=pred)
        result>>> 1, because it have the highest probability to be taken
        """
        action = np.random.choice(self.num_actions, p=action_probabilities)
        action_onehot = np.zeros([self.action_size])
        action_onehot[action] = 1
        return action, action_onehot

    def minibatch_loss(self, states, actions, old_log_probs, returns, advantages):
        # Distributions of all actions for each given state in minibatch
        print("calculating minibatch loss")
        dist, value = self.model(states)
        entropy = dist.entropy().mean()
        new_log_probs = dist.log_prob(actions)
        
        ratio = (new_log_probs - old_log_probs).exp()

        term1 = ratio * advantages
        term2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantage

        actor_loss  = - torch.min(term1, term2).mean()
        critic_loss = (returns - value).pow(2).mean()

        loss = actor_loss + 0.5 * critic_loss - 0.001 * entropy
        return loss

    def ppo_iter(self, mini_batch_size, states, actions, log_probs, returns, advantage):
        """
        Divide batch into mini_batches through generator
        mini_batch set is uniformly sampled from the batch
        """
        print("creating minibatch")
        batch_size = states.size(0)
        for _ in range(batch_size // mini_batch_size):
            rand_ids = np.random.randint(0, batch_size, mini_batch_size)
            yield states[rand_ids, :], actions[rand_ids, :], log_probs[rand_ids, :], returns[rand_ids, :], advantage[rand_ids, :]

    def ppo_update(self, ppo_epochs, mini_batch_size, states, actions, log_probs, returns, advantages, clip_param=0.2):
        print("ppo update called")
        for i in range(ppo_epochs):
            print("ppo update epoch:", i)
            for state, action, old_log_probs, return_, advantage in self.ppo_iter(mini_batch_size, states, actions, log_probs, returns, advantages):
                print("ppo update epoch:", i, "optimizing on minibatches" )
                # dist, value = model(state)
                # entropy = dist.entropy().mean()
                #
                # new_log_probs = dist.log_prob(action)
                #
                # ratio = (new_log_probs - old_log_probs).exp()
                # surr1 = ratio * advantage
                # surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantage
                #
                # actor_loss  = - torch.min(surr1, surr2).mean()
                # critic_loss = (return_ - value).pow(2).mean()
                #
                # loss = 0.5 * critic_loss + actor_loss - 0.001 * entropy
                # calculate loss for minibatch
                loss = self.minibatch_loss(state, action, old_log_probs, return_, advantage)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def compute_gae(self, next_value, rewards, masks, values, gamma=0.99, tau=0.95):
        print("#################### Computing GAE ####################")
        # print("#################### Computing GAE next_value LENGTH:", len(next_value))
        # print("#################### Computing GAE next_value TYPE:", type(next_value))
        # print("#################### Computing GAE next_value DEVICE:", next_value.device)
        # print("#################### Computing GAE next_value:", next_value)
        # print("#################### Computing GAE Rewards LENGTH:", len(rewards))
        # print("#################### Computing GAE Rewards TYPE:", type(rewards))
        # print("#################### Computing GAE Rewards DEVICE:", rewards[0].device)
        # print("#################### Computing GAE Rewards:", rewards)
        # print("#################### Computing GAE masks LENGTH:", len(masks))
        # print("#################### Computing GAE masks TYPE:", type(masks))
        # print("#################### Computing GAE masks DEVICE:", masks[0].device)
        # print("#################### Computing GAE masks:", masks)
        # print("#################### Computing GAE Values LENGTH:", len(values))
        # print("#################### Computing GAE Values TYPE:", type(values))
        # print("#################### Computing GAE Values DEVICE:", values[0].device)
        # print("#################### Computing GAE Values:", values)

        values = values + [next_value]
        gae = 0
        returns = []

        for step in reversed(range(len(rewards))):
            # print("Step: ", step, "Reward: ", rewards[step])
            # print("Step: ", step, "Mask: ",masks[step])
            # print("Step: ", step, "Value: ", values[step])
            # print("Step: ", step, "Next Value: ", values[step+1])
            delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
            gae = delta + gamma * tau * masks[step] * gae
            print("#################### Step: ", step, "Delta: ", delta, "GAE: ", gae)
            returns.insert(0, gae + values[step])


        print("#################### Computing GAE return DIM:", returns[0].size())
        print("#################### finished computing GAE ####################")
        return returns


### INITIALISE ALL

# if __name__ == '__main__':
