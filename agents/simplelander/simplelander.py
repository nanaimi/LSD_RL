import gym

from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import PPO

# multiprocess environment
# example env name UnrealSearch-RealisticRoomDoor-DiscreteColor-v0
env = make_vec_env('UnrealLand-cpptestFloorGood-DiscretePoseColor-v0', n_envs=1)

# PP02 with mlp network for both actor and critic, both with two layers and 64
# neurons each
model = PPO(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=25000)
model.save("testrun")

# Enjoy trained agent
obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
