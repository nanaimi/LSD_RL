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

        print("#################### state:     ", state)
        print("#################### state TYPE:", type(state))
        print("#################### state SIZE:", state.size())

        dist, value = agent.model(state)

        state = state.unsqueeze(1)
        state = torch.transpose(state, 0, 1)

        print("#################### state after unsqueeze and transpose:     ", state)
        print("#################### state after unsqueeze and transpose TYPE:", type(state))
        print("#################### state after unsqueeze and transpose SIZE:", state.size())
        print("#################### Value                 HERE:", value)
        print("#################### Value                 TYPE:", type(value))
        value = value.unsqueeze(1)
        print("#################### Value after unsqueeze HERE:", value)
        print("#################### Value after unsqueeze TYPE:", type(value))

        action = dist.sample()

        print("############## Step:", st, "############## Sampled:", action)
        print("Device:", device)

        next_state, reward, done, _ = env.step(action.cpu().numpy())

        env.render()

        log_prob = dist.log_prob(action)

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
        reward = interim.to(device)
        rewards.append(reward)

        mask = float(1-done)
        mask = torch.FloatTensor([mask])
        mask = mask.unsqueeze(1)
        masks.append(mask.to(device)) # changed from 1-done

        # print("#################### state:", state)
        # print("#################### state SIZE:", state.size())
        # print("#################### state SIZE:", type(state))
        # print("#################### state after unsqueeze SIZE:", state.size())
        # print("#################### state after transpose SIZE:", state.size())

        states.append(state)
        # print("#################### states:    ", states)
        # print("#################### states LEN:", len(states))
        # print("#################### states TYPE:", type(states))

        action = action.detach().numpy()
        action = torch.FloatTensor([np.float(action)])
        action = action.unsqueeze(1)
        action = action.to(device)
        actions.append(action)

        # next state logic
        if done:
            print("#################### RESET SHOULD HAPPEN")
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

    print("#################### Returns          before CAT:", len(returns))
    print("#################### Returns     SIZE after  CAT:", returns.size())
    print("#################### log probs   SIZE after  CAT:", log_probs.size())
    print("#################### Values      SIZE after  CAT:", values.size())
    print("#################### States      SIZE after  CAT:", states.size())
    print("#################### Actions     SIZE after  CAT:", actions.size())
    print("#################### Advantage   SIZE after  CAT:", advantage.size())

    print("###### about to update the params of the networks ######")
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
