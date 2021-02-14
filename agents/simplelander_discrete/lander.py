import gym

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common import make_vec_env
from stable_baselines import PPO2
import gym_unrealcv
import real_lsd
import time

import torch
import numpy as np

from PPOagent import PPOAgent

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# initialise environment
env = gym.make('MyUnrealLand-cpptestFloorGood-DiscretePoseColor-v0')

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
threshold_reward = -200 # update/review


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

max_frames = 15000
frame_idx  = 0
test_rewards = []

state = env.reset()
print("state after reset: ",state)
early_stop = False


while frame_idx < max_frames and not early_stop:
    print("frame: ", frame_idx)
    log_probs = []
    values    = []
    states    = []
    actions   = []
    rewards   = []
    masks     = []
    # total entropy ? what is interesting about this
    entropy = 0

    for st in range(num_steps):
        state = torch.FloatTensor(state).to(device)
        dist, value = agent.model(state)

        action = dist.sample()
        print("step:", st, "sampled:", action)
        print("Device:", device)
        next_state, reward, done, _ = env.step(action.cpu().numpy())
        env.render()

        # weird
        print("#################### Distribution", dist)
        print("#################### Sampled Action", action)
        log_prob = dist.log_prob(action)
        print("#################### Log Probability", log_prob)
        print("#################### Log Probability", type(log_prob))
        entropy += dist.entropy().mean()

        # Append data to arrays
        log_probs.append(log_prob)
        values.append(value)

        print("#################### Reward HERE:", reward)
        print("#################### Reward TYPE:", type(reward))

        interim = torch.FloatTensor([np.float(reward)])
        interim = interim.unsqueeze(1)
        interim = interim.to(device)
        rewards.append(interim)

        print("#################### Rewards LENGTH:", len(rewards))

        masks.append(torch.FloatTensor(1-done).unsqueeze(1).to(device)) # changed from 1-done

        states.append(state)
        actions.append(action)

        # next state logic
        state = next_state
        frame_idx += 1

        if frame_idx % 1000 == 0:
            test_reward = np.mean([test_env() for _ in range(10)])
            test_rewards.append(test_reward)
            plot(frame_idx, test_rewards)
            if test_reward > threshold_reward: early_stop = True

    print("#################### after 20 steps Rewards LENGTH:", len(rewards))

    next_state = torch.FloatTensor(next_state).to(device)
    _, next_value = agent.model(next_state)

    print("#################### after next value Rewards LENGTH:", len(rewards))
    print("#################### after next value Rewards TYPE:", type(rewards))
    print("#################### after next value Rewards ELEMENT TYPE:", type(rewards[0]))
    print("#################### Computing GAE Rewards:", rewards)
    print("#################### after next value Rewards DEVICE:", rewards[0].device)

    # WHAT THE FUCK??? why is rewards turning into a fucking tensor
    returns = agent.compute_gae(next_value,
                                rewards,
                                masks,
                                values)
    time.sleep(5)

    returns   = torch.cat(returns).detach()
    log_probs = torch.cat(log_probs).detach()
    values    = torch.cat(values).detach()
    states    = torch.cat(states)
    actions   = torch.cat(actions)
    advantage = returns - values

    print("about to update the params of the networks")
    agent.ppo_update(ppo_epochs, mini_batch_size, states, actions, log_probs, returns, advantage)


#
# # multiprocess environment
# # example env name
# # UnrealLand-cpptestFloorGood-DiscretePoseColor-v0
# env = gym.make('MyUnrealLand-cpptestFloorGood-DiscretePoseColor-v0')
# # env = make_vec_env('UnrealSearch-RealisticRoomDoor-DiscreteColor-v0', n_envs=1)
#
# # PP02 with mlp network for both actor and critic, both with two layers and 64
# # neurons each
# model = PPO2(MlpPolicy, env, verbose=1)
# model.learn(total_timesteps=2000) # test with fewer timesteps
# model.save("testrun")
#
#
# # Enjoy trained agent
# obs = env.reset()
# while True:
#     action, _states = model.predict(obs)
#     obs, rewards, dones, info = env.step(action)
#     env.render()
