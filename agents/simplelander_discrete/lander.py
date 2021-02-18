import os
import gym
import time
import torch
import pickle
import numpy as np
import glog as log

from stable_baselines import PPO2
from stable_baselines.common import make_vec_env
from stable_baselines.common.policies import MlpPolicy

import gym_unrealcv
import real_lsd

from PPOagent import PPOAgent

# Helper functions
def save_obj(obj, name=''):
    dir = 'data'
    path = os.getcwd()
    filename = time.strftime("%Y%m%d_%H%M%S") + name
    if dir not in os.listdir(path):
        os.mkdir(dir)
    path = path + '/' + dir + '/'
    with open(path + filename + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

    return filename

def load_obj(filename):
    dir = 'data'
    path = os.getcwd()
    assert dir in os.listdir(path)
    path = path + '/' + dir + '/'
    with open(path + filename + '.pkl', 'rb') as f:
        return pickle.load(f)

# get activation of layers in model
activation = {}
def get_activation(name):
    def hook(model, input, output):
        # print(name)
        activation[name] = output.detach()
        return hook

# Set to INFO for debugging
log.setLevel("WARN")

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# initialise environment
env = gym.make('MyUnrealLand-cpptestFloorGood-DiscreteHeightFeatures-v0')

print("Observation Space:", env.observation_space, "dimension of observation:", env.observation_space.shape[0])
num_inputs  = env.observation_space.shape[0]
print("Action Space:", env.action_space, "Number of actions:", env.action_space.n)
num_outputs = env.action_space.n

#Hyper params:
hidden_size      = 256
lr               = 3e-4
num_steps        = 20
mini_batch_size  = 5
ppo_epochs       = 4

threshold_reward = -200 # TODO update/review


agent = PPOAgent(num_inputs,
                num_outputs,
                hidden_size,
                lr,
                num_steps,
                mini_batch_size,
                ppo_epochs,
                threshold_reward,
                std=0.0,
                device=device)

print(agent.model)
print(agent.model.actor)
print(agent.model.actor[0])
print(agent.model.actor[1])




for name, layer in agent.model.actor.named_modules():
    print(layer)
    print(type(name))
    idx = int(name)
    agent.model.actor[idx].register_forward_hook(get_activation('actor_layer_{}'.format(name)))


# output = model(x)
#
# activation['fc3']

# for name, layer in agent.model.named_modules():
#     print(name)
#     print(layer, type(layer))
#
# for name, layer in agent.model.actor.named_modules():
#     print("name",name)
#     print(layer, type(layer))


# agent.model.actor.register_forward_hook(get_activation('fc3'))
# output = model(x)
# activation['fc3']

# turn model train mode
agent.model.train()

max_frames = 15000
frame_idx  = 0
test_rewards = []
prior_action = 0

state = env.reset()

early_stop = False

while frame_idx < max_frames and not early_stop:
    log.warn("Frame: {}".format(frame_idx))
    log_probs = []
    values    = []
    states    = []
    actions   = []
    rewards   = []
    masks     = []
    # total entropy ? what is interesting about this
    entropy = 0

    for st in range(num_steps):
        action = 0
        state = torch.FloatTensor(state).to(device)
        # log.info("state:      {}".format(state))
        # log.info("state TYPE: {}".format(type(state)))
        log.info("state SIZE: {}".format(state.size()))

        log.warn("Model Input: {}".format(state))
        dist, value = agent.model(state)
        log.info("Forward Pass Dist: {}, Forward Pass value: {}".format(dist, value))

        state = state.unsqueeze(1)
        state = torch.transpose(state, 0, 1)
        # log.info("state after unsqueeze and transpose:      {}".format(state))
        # log.info("state after unsqueeze and transpose TYPE: {}".format(type(state)))
        log.info("state after unsqueeze and transpose SIZE: {}".format(state.size()))

        value = value.unsqueeze(1)
        log.info("Value: {}".format(value))
        # log.info("Value TYPE: {}".format(type(value)))

        action = dist.sample()
        log.warn("Sampled Action: {}".format(action))
        action_not_same = not (action == prior_action)
        log.warn("Sampled Action is not same as prior action: {}".format(action_not_same))
        prior_action = action

        next_state, reward, done, _ = env.step(action.cpu().numpy())
        log.info("Step REWARD: {} DONE: {}".format(reward, done))

        log_prob = dist.log_prob(action)
        log.info("Step LOG_PROB: {}".format(log_prob))

        entropy += dist.entropy().mean()

        # Append data to arrays
        np_log_prob = log_prob.detach().numpy()
        log_prob = torch.FloatTensor([np.float(np_log_prob)])
        log_prob = log_prob.unsqueeze(1)
        log_prob = log_prob.to(device)
        log_probs.append(log_prob)

        values.append(value)

        interim = torch.FloatTensor([np.float(reward)])
        interim = interim.unsqueeze(1)
        reward  = interim.to(device)
        rewards.append(reward)

        mask = float(1-done)
        mask = torch.FloatTensor([mask])
        mask = mask.unsqueeze(1)
        masks.append(mask.to(device)) # changed from 1-done

        states.append(state)

        action = action.detach().numpy()
        action = torch.FloatTensor([np.float(action)])
        action = action.unsqueeze(1)
        action = action.to(device)
        actions.append(action)

        # next state logic
        if done:
            log.warn("Resetting.")
            state = env.reset()
        else:
            state = next_state

        frame_idx += 1

        # if frame_idx % 1000 == 0:
        #     test_reward = np.mean([test_env() for _ in range(10)])
        #     test_rewards.append(test_reward)
        #     plot(frame_idx, test_rewards)
        #     if test_reward > threshold_reward: early_stop = True

    next_state = torch.FloatTensor(next_state).to(device)

    _, next_value = agent.model(next_state)

    next_state = next_state.unsqueeze(1)
    next_state = torch.transpose(next_state, 0, 1)

    returns = agent.compute_gae(next_value,
                                rewards,
                                masks,
                                values)

    returns   = torch.cat(returns).detach()
    log_probs = torch.cat(log_probs).detach()
    values    = torch.cat(values).detach()
    states    = torch.cat(states)
    actions   = torch.cat(actions)
    advantage = returns - values

    # log.info("Returns before CAT: {}".format(len(returns)))
    log.info("Returns   SIZE after CAT: {}".format(returns.size()))
    log.info("log probs SIZE after CAT: {}".format(log_probs.size()))
    log.info("Values    SIZE after CAT: {}".format(values.size()))
    log.info("States    SIZE after CAT: {}".format(states.size()))
    log.info("Actions   SIZE after CAT: {}".format(actions.size()))
    log.info("Advantage SIZE after CAT: {}".format(advantage.size()))

    log.warn("Calling PPO update.")
    agent.ppo_update(ppo_epochs, mini_batch_size, states, actions, log_probs, returns, advantage)

agent.save_model()

log.info("FINITA LA MUSICA")

# Testing the policy after training
log.warn("Time to test.")
agent.model.eval()
with torch.no_grad():
    for test in range(5):
        episode_count = 0
        successful_episodes = 0
        num_test_episodes = 20

        episodes = {}

        state = env.reset()

        while episode_count < num_test_episodes:

            done = False
            episode = {}

            states    = []
            dists     = []
            values    = []
            actions   = []
            rewards   = []
            log_probs = []

            while not done:
                action = 0
                states.append(state)

                state = torch.FloatTensor(state).to(device)
                dist, value = agent.model(state)

                action = dist.sample()
                log.info("action type: {}".format(action))
                log_prob = dist.log_prob(action)

                next_state, reward, done, info = env.step(action.cpu().numpy())

                dists.append(dist)
                values.append(value.detach().numpy())
                actions.append(action.detach().numpy())
                log_probs.append(log_prob.detach().numpy())
                rewards.append(reward)

                # next state logic
                if done:
                    if info['Success']:
                        successful_episodes += 1
                    state = env.reset()
                    episode_count += 1
                else:
                    state = next_state

            episode['states'] = states
            episode['dists'] = dists
            episode['values'] = values
            episode['actions'] = actions
            episode['rewards'] = rewards
            episode['log_probs'] = log_probs

            key = 'episode_{}'.format(episode_count)
            episodes[key] = episode

        file = save_obj(episodes, '{}'.format(test))
        log.warn("Successes out of 20: {}".format(successful_episodes))
        # print(load_obj(file))

log.warn("Done Testing.")

env.close()
