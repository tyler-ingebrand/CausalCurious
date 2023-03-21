import os

from causal_world.envs.causalworld import CausalWorld
from causal_world.task_generators.base_task import BaseTask
import numpy as np

class MyOwnTask(BaseTask):

    def __init__(self, shape="Cube", mass="Heavy", size="Big", **kwargs):
        assert shape == 'Cube' or shape == 'Sphere', "Shape must be one of 'Cube' or 'Sphere', got {}".format(shape)
        assert mass == 'Heavy' or mass == 'Light', "Mass must be one of 'Heavy' or 'Light', got {}".format(mass)
        assert size == 'Big' or size == 'Small', "Size must be one of 'Big' or 'Small', got {}".format(size)
        self.shape = shape
        self.size = size
        self.mass = mass
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

        # change mass
        if self.mass == 'Heavy':
            mass = 0.12
        elif self.mass == 'Light':
            mass = 0.04
        else:
            raise Exception("Mass must be one of 'Heavy' or 'Light', got {}".format(self.mass))

        # Change size
        if self.size == "Big":
            scale = 1.0
        elif self.size == "Small":
            scale = 0.5
        else:
            raise Exception("Size must be one of 'Big' or 'Small', got {}".format(self.size))

        # change object model
        if self.shape == 'Cube':
            filename = os.path.join(dir_path, 'objs/cube.obj')
            width = 0.09 * scale # cube width
        elif self.shape == 'Sphere':
            filename = os.path.join(dir_path, 'objs/sphere.obj')
            width = 0.05 * scale # radius I think
        else:
            raise Exception("Shape must be one of 'Cube' or 'Sphere', got {}".format(self.shape))
        scale = [width, width, width]

        # create main obj
        creation_dict = {
            'name': "tool_block",
            'filename': filename,
            'initial_position': [0, 0, 0.1],
            'scale': scale,
            'mass': mass,
        }
        self._stage.add_rigid_mesh_object(**creation_dict)

        # define state space
        self._task_stage_observation_keys = [
            "tool_block_cartesian_position", "tool_block_orientation",
            "tool_block_linear_velocity", "tool_block_angular_velocity",
        ]
        return
    

    def reset_task(self, interventions_dict=None, check_bounds=True):
        """
        :param interventions_dict: (dict) intervention dict to be specified
                                          if an intervention to be latched
                                          as the new starting state of the
                                          environment.
        :param check_bounds: (bool) specified when not in train mode and a
                                    check for the intervention if its allowed
                                    or not is needed.

        :return: (tuple): success_signal specifying if the intervention is
                          successful or not,
                          interventions_info specifying
                          number of interventions and other info,
                          reset_observation_space_signal a bool specifying if
                          the observation space needs to be changed or not.
        """
        self.reset_default_state()
        #print(self._current_starting_state['stage_object_state']['rigid_objects'][0][1]['initial_position'])
        a, b = -0.005, 0.005 # this one works but is maybe too easy
        # a, b = -0.02, 0.02 # this one works but is maybe too easy
        random_offset = (b - a) * np.random.random_sample((3,)) + a
        self._current_starting_state['stage_object_state']['rigid_objects'][0][1]['initial_position'] += random_offset
        #print(self._current_starting_state['stage_object_state']['rigid_objects'][0][1]['initial_position'])

        self._robot.clear()
        reset_observation_space_signal = \
            self.restore_state(self._current_starting_state)

        self._task_solved = False
        success_signal = None
        interventions_info = None
        if interventions_dict is not None:
            success_signal, interventions_info, reset_observation_space_signal = \
                self.apply_interventions(interventions_dict,
                                         check_bounds=check_bounds)
            if success_signal:
                self._current_starting_state = self.save_state()
        self._set_task_state()
        return success_signal, interventions_info, reset_observation_space_signal