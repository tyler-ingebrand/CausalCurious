import gym
import matplotlib.pyplot as plt
import numpy
import torch
from causal_world.envs import CausalWorld
from causal_world.task_generators import generate_task
from gym.wrappers import TransformObservation
from gym.spaces import Box
from pybullet_utils.util import set_global_seeds
from stable_baselines3.common.vec_env import SubprocVecEnv

from src.custom_task import MyOwnTask
from src.CausalCuriousTD3 import *
from stable_baselines3.common.env_util import make_vec_env
from moviepy.editor import *
import os
import pickle


from gym import ObservationWrapper, ActionWrapper


class Transform32(ObservationWrapper, ActionWrapper):
    def __init__(self, env):
        super(Transform32, self).__init__(env)
        self.transform_state = lambda s: np.float32(s)
        self.transform_action = lambda a: np.float64(a)
        self.transform_action_reverse = lambda a: np.float32(a)
        self.observation_space = Box(np.float32(env.observation_space.low),
                                     np.float32(env.observation_space.high))
        self.action_space = Box(np.float32(env.action_space.low),
                                np.float32(env.action_space.high))
        self._max_episode_length = env._max_episode_length

    def observation(self, observation):
        return self.transform_state(observation)

    def action(self, action):
        return self.transform_action(action)

    def reverse_action(self, action):
        return self.transform_action_reverse(action)


# For Sophia, in the event I forget how to activate my virtual environment:  source /path/to/venv/bin/activate
def test(seed, number_envs, total_timesteps, change_shape,change_size,  change_mass, initial_state_randomness=0.005, multi = False):

    assert change_shape or change_size or change_mass
    print(f"Testing TD3 on {number_envs} envs for {total_timesteps} steps.\n"
          f"Varying {'shape ' if change_shape else ''}"
          f"{'mass ' if change_mass else ''}"
          f"{'size ' if change_size else ''}on seed {seed}.\n"
          f"S_0 noise = {initial_state_randomness}")
    torch.manual_seed(seed)
    numpy.random.seed(seed)
    exp_dir = "td3_multi_{}{}{}seed_{}_steps_{}_random_{}".format( "change_shape_" if change_shape else "", "change_size_" if change_size else "", "change_mass_" if change_mass else "", seed, total_timesteps, initial_state_randomness)
    os.makedirs("results/{}".format(exp_dir), exist_ok=True)

    # Get causal world environment. second half are cube, first half are sphere
    # things we can compare: weight heavy vs light, shape cube vs sphere, size big vs small? 
    def _make_multi_env(rank):
        if change_mass and change_shape and change_size: 
            raise ValueError("can only have at most 2 true" )
        def _init():
            if change_shape and change_size: 
                task = MyOwnTask(shape = "Sphere" if rank < number_envs/2  else "Cube",
                                size = "Small" if rank < number_envs/4 or rank >= number_envs/2 and rank < 3* number_envs/4  else "Big",
                                mass = "Light",
                                randomness = initial_state_randomness)

            if change_mass and change_size: 
                task = MyOwnTask(shape = "Cube", #"Sphere" if rank < number_envs/2  else "Cube",
                                size = "Small" if rank < number_envs/4 or rank >= number_envs/2 and rank < 3* number_envs/4  else "Big",
                                mass = "Light" if rank < number_envs/2  else "Heavy",
                                randomness = initial_state_randomness)

            elif change_mass and change_shape: 
                task = MyOwnTask(shape = "Sphere" if rank < number_envs/2  else "Cube",
                                size = "Small",  #if rank < number_envs/4 or rank >= number_envs/2 and rank < 3* number_envs/4  else "Big",
                                mass = "Light" if rank < number_envs/4 or rank >= number_envs/2 and rank < 3* number_envs/4  else "Heavy",
                                randomness = initial_state_randomness)
            elif change_shape and change_size: 
                task = MyOwnTask(shape = "Sphere" if rank < number_envs/2  else "Cube",
                                size = "Small" if rank < number_envs/4 or rank >= number_envs/2 and rank < 3* number_envs/4  else "Big",
                                mass = "Light",
                                randomness = initial_state_randomness)
            else: 
                raise ValueError("need at least two qualities true for a multi environement")

            env = CausalWorld(task=task,
                              enable_visualization=False,
                              seed=seed + rank,
                              max_episode_length=249,
                              skip_frame=10,
                              # action_mode='end_effector_positions'
                              )
            # random_intervention_dict = env.do_intervention(
            #       {'tool_block': {"initial_position":[0,0,0.2]}},
            #     check_bounds=False
            # )
            # print(random_intervention_dict)

            a = -initial_state_randomness
            b = initial_state_randomness
            random_offset = (b - a) * numpy.random.random_sample() + a
            env._task._current_starting_state['stage_object_state']['rigid_objects'][0][1]['initial_position'] += random_offset

            # get rid of float64
            env = Transform32(env)
            return env
        set_global_seeds(seed)
        return _init

    

    env = SubprocVecEnv([_make_multi_env(i) for i in range(number_envs)])
 


    # train
    model = CausalTD3("MlpPolicy", env,
                             train_freq=(1, "episode"),
                             episode_length=env.get_attr("_max_episode_length", [0])[0] + 1, # this env returns the index of last step, we want total number of steps
                             verbose=0,
                             debug_dir = "results/{}".format(exp_dir), multi = multi )

 
    model.learn(total_timesteps=total_timesteps, progress_bar=True)




    # plot result of learning
    l1, = plt.plot(model.timesteps, model.mean_distance_my_cluster, color='r')
    l2, = plt.plot(model.timesteps, model.mean_distance_other_cluster, color='orange')
    plt.xlabel("Env Interactions")
    plt.ylabel("Euclidean Distance")

    # plot result of learning
    plt.twinx()
    l3, = plt.plot(model.timesteps, model.success_rates, color='b')
    plt.xlabel("Env Interactions")
    plt.ylabel("Correct Identification Percentage")
    plt.title("Cluster properties during training")
    plt.legend([l1, l2, l3], ["Cluster Size", "Cluster Separation", "Classification Success rate"])
    plt.savefig("results/{}/properties.png".format(exp_dir), dpi=300, bbox_inches='tight')
    plt.clf()

    # make a pickel
    pickle.dump({"timesteps": model.timesteps,
                 "mean_distance_my_cluster": model.mean_distance_my_cluster,
                 "mean_distance_other_cluster": model.mean_distance_other_cluster,
                 "success_rates": model.success_rates
                 },
                open(f"results/{exp_dir}/data.pkl", 'wb'))

    # show episode, save
    seconds_per_frame = 0.1
    base_file_name = "experiment"
    for episode in range(10):
        images = []
        done = False
        obs = env.reset()
        for t in range(100):
            action, _states = model.predict(obs)
            obs, rewards, dones, info = env.step(action)
            frame = env.render(mode='rgb_array')
            images.append(ImageClip(frame).set_duration(seconds_per_frame))
            done = dones.any()
        video = concatenate(images, method="compose")
        video.write_videofile("results/{}/{}_episode_{}.mp4".format( exp_dir, base_file_name, episode), fps=24)
        del images
        del video

if __name__ == '__main__':

    # parameters ##
    seed = 1
    number_envs = 32
    episodes_per_update = 1
    total_timesteps =  300_000
    change_shape = True
    change_size = False
    change_mass = True
    random = 0.005
    test(seed, number_envs, total_timesteps, change_shape, change_size,  change_mass, initial_state_randomness=random, multi= True)
