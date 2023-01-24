import gym
from causal_world.envs import CausalWorld
from causal_world.task_generators import generate_task
from src.CausalCuriousPPO import *
from stable_baselines3.common.env_util import make_vec_env

# Get causal world environment
task = generate_task(task_generator_id='general')
env = CausalWorld(task=task, enable_visualization=False)

# train
model = CausalCuriousPPO("MlpPolicy", env,
                         episode_length=env._max_episode_length, episodes_per_update=2, verbose=1)
model.learn(total_timesteps=25000)


# show episode
obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()