import gym

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common import make_vec_env
from stable_baselines import PPO2

# multiprocess environment
env = make_vec_env('MYREGISTEREDENVNAME', n_envs=1)

# PP02 with mlp network for both actor and critic, both with two layers and 64
# neurons each
model = PPO2(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=25000)
model.save("")

model = PPO2.load("")

# Enjoy trained agent
obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
