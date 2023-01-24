import gym

import CausalCuriousPPO
from stable_baselines3.common.env_util import make_vec_env

# Get causal world environment # TODO
raise Exception("TODO")
env = make_vec_env("CartPole-v1", n_envs=4)


# train
model = CausalCuriousPPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=25000)


# show episode
obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()