import gym
from causal_world.envs import CausalWorld
from causal_world.task_generators import generate_task
from pybullet_utils.util import set_global_seeds
from stable_baselines3.common.vec_env import SubprocVecEnv

from src.custom_task import MyOwnTask
from src.CausalCuriousPPO import *
from stable_baselines3.common.env_util import make_vec_env

if __name__ == '__main__':

    # parameters ##
    seed = 22
    number_envs = 8
    ###############


    # Get causal world environment
    def _make_env(rank):

        def _init():
            task = MyOwnTask('Sphere' if rank < number_envs/2 else 'Cube')
            env = CausalWorld(task=task,
                              enable_visualization=False,
                              seed=seed + rank,
                              max_episode_length=249,
                              skip_frame=10,
                              )
            return env

        set_global_seeds(seed)
        return _init
    env = SubprocVecEnv([_make_env(i) for i in range(number_envs)])


    # train
    model = CausalCuriousPPO("MlpPolicy", env,
                             episode_length=env.get_attr("_max_episode_length", [0])[0] + 1, # this env returns the index of last step, we want total number of steps
                             episodes_per_update=8, verbose=1)
    model.learn(total_timesteps=1000000)


    print(" FINISHED LEARNING" )
    # show episode
    obs = env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render()