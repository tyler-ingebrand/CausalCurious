import numpy
from causal_world.envs import CausalWorld
from pybullet_utils.util import set_global_seeds
from stable_baselines3.common.vec_env import SubprocVecEnv
import torch
from src.CausalCuriousPPO import CausalCuriousPPO
from src.custom_task import MyOwnTask

if __name__ == "__main__":
    number_envs=1
    change_size=False
    change_shape=True
    change_mass=False
    seed=0
    torch.manual_seed(seed)
    numpy.random.seed(seed) 
    # Get causal world environment. second half are cube, first half are sphere
    # things we can compare: weight heavy vs light, shape cube vs sphere, size big vs small?
    def _make_env(rank):
        def _init():
            task = MyOwnTask(shape="Sphere" if rank < number_envs/2 and change_shape else "Cube",
                             size="Small" if rank < number_envs/2  and change_size else "Big",
                             mass="Light" if rank < number_envs/2 and change_mass else "Heavy")
            env = CausalWorld(task=task,
                              enable_visualization=False,
                              seed=seed,
                              max_episode_length=249,
                              skip_frame=10,
                              # action_mode='end_effector_positions'
                              )
            # random_intervention_dict = env.do_intervention(
            #       {'tool_block': {"initial_position":[0,0,0.2]}},
            #     check_bounds=False
            # )
            # print(random_intervention_dict)

            a = -0.05
            b = 0.05
            random_offset = (b - a) * numpy.random.random_sample() + a
            env._task._current_starting_state['stage_object_state']['rigid_objects'][0][1]['initial_position'] += random_offset


            #print(env._task._current_starting_state['stage_object_state']['rigid_objects'][0][1]['initial_position'])
            #print(env._task._current_starting_state['stage_object_state']['rigid_objects'][0][1]['initial_orientation'])
            return env

        set_global_seeds(seed)
        return _init

    env = SubprocVecEnv([_make_env(i) for i in range(number_envs)])


    # train
    model = CausalCuriousPPO("MlpPolicy", env,
                             episode_length=env.get_attr("_max_episode_length", [0])[0] + 1, # this env returns the index of last step, we want total number of steps
                             episodes_per_update=1, verbose=1,
                             )






    # show episode, save
    for episode in range(2):
        obs = env.reset()
        print(obs[: , 27:])
        for t in range(100):
            action, _states = model.predict(obs)
            obs, rewards, dones, info = env.step(action)
            # frame = env.render(mode='human')
            done = dones.any()
            # input()

