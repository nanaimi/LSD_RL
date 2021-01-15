import gym

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common import make_vec_env
from stable_baselines import PPO2
import gym_unrealcv
import real_lsd

use_cuda = torch.cuda.is_available()
device   = torch.device("cuda" if use_cuda else "cpu")

num_inputs  = envs.observation_space.shape[0]
num_outputs = envs.action_space.shape[0]

#Hyper params:
hidden_size      = 256
lr               = 3e-4
num_steps        = 20
mini_batch_size  = 5
ppo_epochs       = 4
threshold_reward = -200

# multiprocess environment
# example env name
# UnrealLand-cpptestFloorGood-DiscretePoseColor-v0
env = gym.make('MyUnrealLand-cpptestFloorGood-DiscretePoseColor-v0')
# env = make_vec_env('UnrealSearch-RealisticRoomDoor-DiscreteColor-v0', n_envs=1)

# PP02 with mlp network for both actor and critic, both with two layers and 64
# neurons each
model = PPO2(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=2000) # test with fewer timesteps
model.save("testrun")


# Enjoy trained agent
obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
