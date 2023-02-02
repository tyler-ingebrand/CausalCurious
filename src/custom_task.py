import os

from causal_world.envs.causalworld import CausalWorld
from causal_world.task_generators.base_task import BaseTask
import numpy as np

class MyOwnTask(BaseTask):

    def __init__(self, shape="Cube", **kwargs):
        assert shape == 'Cube' or shape == 'Sphere', "Shape must be one of 'Cube' or 'Sphere', got {}".format(shape)
        self.shape = shape
        super().__init__(task_name="new_task",
                         variables_space='space_a_b',
                         fractional_reward_weight=1,
                         dense_reward_weights=np.array([]))
        self._task_robot_observation_keys = [
            "joint_positions", "joint_velocities", "end_effector_positions"
        ]


    #This is not even needed, it will just be an empty stage
    def _set_up_stage_arena(self):
        # Create relative import by getting current file location and appending relative file location
        dir_path = os.path.dirname(os.path.realpath(__file__))
        if self.shape == 'Cube':
            filename = os.path.join(dir_path, 'objs/cube.obj')
            scale = [0.09, 0.09, 0.09] # scaling is different because obj models are different sizes
        elif self.shape == 'Sphere':
            filename = os.path.join(dir_path, 'objs/sphere.obj')
            scale = [0.05, 0.05, 0.05]# scaling is different because obj models are different sizes
        else:
            raise Exception("Shape must be one of 'Cube' or 'Sphere', got {}".format(self.shape))


        #NOTE: you need to add rigid objects before silhouettes for determinism (pybullet limitation)
        # create main obj
        creation_dict = {
            'name': "tool_block",
            'filename': filename,
            'initial_position': [0, 0, 0.1],
            'scale': scale
        }
        self._stage.add_rigid_mesh_object(**creation_dict)

        # define state space
        self._task_stage_observation_keys = [
            "tool_block_type", "tool_block_size",
            "tool_block_cartesian_position", "tool_block_orientation",
            "tool_block_linear_velocity", "tool_block_angular_velocity",
        ]
        return