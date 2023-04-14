from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union

import numpy as np
import torch
import torch as th
from gym import spaces
from torch.nn import functional as F

from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import get_parameters_by_name, polyak_update
from .CustomTD3Policies import *
from .clustering_functions import *

SelfCausalTD3 = TypeVar("SelfCausalTD3", bound="CausalTD3")


class CausalTD3(OffPolicyAlgorithm):
    """
    Twin Delayed DDPG (CausalTD3)
    Addressing Function Approximation Error in Actor-Critic Methods.
    Original implementation: https://github.com/sfujim/CausalTD3
    Paper: https://arxiv.org/abs/1802.09477
    Introduction to CausalTD3: https://spinningup.openai.com/en/latest/algorithms/CausalTD3.html
    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: learning rate for adam optimizer,
        the same learning rate will be used for all networks (Q-Values, Actor and Value function)
        it can be a function of the current progress remaining (from 1 to 0)
    :param buffer_size: size of the replay buffer
    :param learning_starts: how many steps of the model to collect transitions for before learning starts
    :param batch_size: Minibatch size for each gradient update
    :param tau: the soft update coefficient ("Polyak update", between 0 and 1)
    :param gamma: the discount factor
    :param train_freq: Update the model every ``train_freq`` steps. Alternatively pass a tuple of frequency and unit
        like ``(5, "step")`` or ``(2, "episode")``.
    :param gradient_steps: How many gradient steps to do after each rollout (see ``train_freq``)
        Set to ``-1`` means to do as many gradient steps as steps done in the environment
        during the rollout.
    :param action_noise: the action noise type (None by default), this can help
        for hard exploration problem. Cf common.noise for the different action noise type.
    :param replay_buffer_class: Replay buffer class to use (for instance ``HerReplayBuffer``).
        If ``None``, it will be automatically selected.
    :param replay_buffer_kwargs: Keyword arguments to pass to the replay buffer on creation.
    :param optimize_memory_usage: Enable a memory efficient variant of the replay buffer
        at a cost of more complexity.
        See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
    :param policy_delay: Policy and target networks will only be updated once every policy_delay steps
        per training steps. The Q values will be updated policy_delay more often (update every training step).
    :param target_policy_noise: Standard deviation of Gaussian noise added to target policy
        (smoothing noise)
    :param target_noise_clip: Limit for absolute value of target policy smoothing noise.
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: Verbosity level: 0 for no output, 1 for info messages (such as device or wrappers used), 2 for
        debug messages
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    """

    policy_aliases: Dict[str, Type[BasePolicy]] = {
        "MlpPolicy": MlpPolicy,
        "CnnPolicy": CnnPolicy,
        "MultiInputPolicy": MultiInputPolicy,
    }

    def __init__(
        self,
        policy: Union[str, Type[CausalTD3Policy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 1e-3,
        episode_length:int = 200,
        learning_starts: int = 100,
        batch_size: int = 100,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: Union[int, Tuple[int, str]] = (1, "episode"),
        gradient_steps: int = -1,
        action_noise: Optional[ActionNoise] = None,
        replay_buffer_class: Optional[Type[ReplayBuffer]] = None,
        replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
        optimize_memory_usage: bool = False,
        policy_delay: int = 2,
        target_policy_noise: float = 0.2,
        target_noise_clip: float = 0.5,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        debug_dir=None, multi= False,
    ):
        number_envs = 32
        train_freq = (episode_length ,"step")
        # buffer_size = episode_length * number_envs
        # buffer_size = 100_000
        buffer_size = number_envs * episode_length * 15
        self.recent_steps = episode_length
        self.multi= multi

        super().__init__(
            policy,
            env,
            learning_rate,
            buffer_size,
            learning_starts,
            batch_size,
            tau,
            gamma,
            train_freq,
            gradient_steps,
            action_noise=action_noise,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            policy_kwargs=policy_kwargs,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            seed=seed,
            sde_support=False,
            optimize_memory_usage=optimize_memory_usage,
            supported_action_spaces=(spaces.Box),
            support_multi_env=True,
        )

        self.policy_delay = policy_delay
        self.target_noise_clip = target_noise_clip
        self.target_policy_noise = target_policy_noise

        if _init_setup_model:
            self._setup_model()

        self.mean_distance_my_cluster = []
        self.mean_distance_other_cluster = []
        self.timesteps = []
        self.debug_dir = debug_dir
        self.episode_starts = []
        self.success_rates = [] #TODO: Here

    def _setup_model(self) -> None:
        super()._setup_model()
        self._create_aliases()
        # Running mean and running var
        self.actor_batch_norm_stats = get_parameters_by_name(self.actor, ["running_"])
        self.critic_batch_norm_stats = get_parameters_by_name(self.critic, ["running_"])
        self.actor_batch_norm_stats_target = get_parameters_by_name(self.actor_target, ["running_"])
        self.critic_batch_norm_stats_target = get_parameters_by_name(self.critic_target, ["running_"])

    def _create_aliases(self) -> None:
        self.actor = self.policy.actor
        self.actor_target = self.policy.actor_target
        self.critic = self.policy.critic
        self.critic_target = self.policy.critic_target

    def get_episode_starts(self):
        buffer = self.replay_buffer

        # for i in range(len(buffer.dones)):
        #     if (buffer.dones[i] != 0).any():
        #         if not (buffer.episode_starts[i] != 0).all():
        #             raise Exception("Episodes do not have the same length, TODO Can we handle this?")
        #         self.episode_starts.append(i)
        self.episode_starts.append(0)  # append the length of the buffer so we have a marker at the start and end of each episode
        self.episode_starts.append(len(buffer.dones))  # append the length of the buffer so we have a marker at the start and end of each episode


    def record_success_rates_multi(self, labels):
        num_items = len(labels)
        group_averages = []

        group_1_counter = [0,0]
        num_group_1 = 0
        group_2_counter = [0, 0]
        num_group_2 = 0
        group_3_counter = [0,0]
        num_group_3 = 0
        group_4_counter = [0,0]
        num_group_4 = 0

        for i in range(len(labels)):
            if i < num_items / 4:
                group_1_counter[labels[i]] += 1
                num_group_1 += 1
            elif i >= num_items/4 and i < num_items/2:
                group_2_counter[labels[i]] += 1
                num_group_2 += 1
            elif i>= num_items/2  and i < 3*num_items/4:
                group_3_counter[labels[i]] += 1
                num_group_3 += 1
            else:
                group_4_counter[labels[i]] += 1
                num_group_4 += 1

        group_1_avg = max(group_1_counter)/num_group_1
        group_2_avg = max(group_2_counter)/num_group_2
        group_3_avg = max(group_3_counter)/num_group_3
        group_4_avg = max(group_4_counter)/num_group_4
        print(max(range(len(group_1_counter)), key=group_1_counter.__getitem__),
              max(range(len(group_2_counter)), key=group_2_counter.__getitem__),
              max(range(len(group_3_counter)), key=group_3_counter.__getitem__),
              max(range(len(group_4_counter)), key=group_4_counter.__getitem__),)
        self.success_rates.append( (group_1_avg + group_2_avg + group_3_avg + group_4_avg)/ 4)
        # return (group_1_avg + group_2_avg + group_3_avg + group_4_avg)/ 4

    def record_success_rates(self, labels):
        # print(labels)
        guess_0, guess_1 = np.zeros(len(labels)), np.zeros(len(labels))
        guess_0[:len(labels)//2] = 1
        guess_1[len(labels)//2:] = 1
        error_0 = sum(labels == guess_0)/len(labels)
        error_1 = sum(labels == guess_1)/len(labels)
        # print("Success rate: ", max(error_0, error_1))
        self.success_rates.append(max(error_0, error_1))

    def generate_synthetic_reward(self):
        buffer = self.replay_buffer

        # find episode starts if we have not already, IE during the first update
        if len(self.episode_starts) == 0:
            self.get_episode_starts()

        # get obs data. Remove robot states since we dont care about those for clyustering
        # Robot obs are the first 27 dims in statespace
        end = buffer.pos if buffer.pos != 0 else buffer.observations.shape[0]
        start = end - self.recent_steps
        obs = buffer.observations[start : end, :, 27:]
        # print(f"start = {start}, end = {end}")
        # print(obs.shape, "buffer pos = ", buffer.pos, ", recent steps = ", self.recent_steps, "querying from ", buffer.pos - self.recent_steps, " to ", buffer.pos)

        # Reorder state information, need it to be n_trajectories x n_timesteps x dimensions
        # reformat obs for clustering alg. Is n_trajs X n_timesteps X dimension of state space
        data = format_obs(obs, self.episode_starts, normalize=False)

        # do tslearn stuff
        kmeans = cluster(data,
                        n_clusters=2,
                        distance_metric="softdtw",
                        multi_process = True,
                        plot = True,
                        verbose = False,
                        timestep=self.num_timesteps,
                        debug_dir=self.debug_dir)
        if self.multi == False: 
            self.record_success_rates(kmeans.labels_)
        else:
            self.record_success_rates_multi(kmeans.labels_)


        # compute distances between current cluster and other cluster
        distance_to_my_cluster, distance_to_other_cluster = get_distances_between_trajectories_and_clusters(kmeans.labels_,
                                                                                                            kmeans.cluster_centers_,
                                                                                                            data,
                                                                                                            verbose=False,
                                                                                                            plot=False)

        mean_distance_to_my_cluster = np.mean(distance_to_my_cluster)
        mean_distance_to_other_cluster = np.mean(distance_to_other_cluster)

        # change_in_distance_to_my_cluster, change_in_distance_to_other_cluster = get_change_in_distance(distance_to_my_cluster, distance_to_other_cluster)


        # normalize distances
        #distance_to_my_cluster = normalize_distances(distance_to_my_cluster)
        #distance_to_other_cluster = normalize_distances(distance_to_other_cluster)


        # create reward
        # reward = 2 * distance_to_other_cluster - distance_to_my_cluster
        ## TEST: run this with only the distance to the other cluster
        # reward = 3 * distance_to_other_cluster - distance_to_my_cluster
        reward = distance_to_other_cluster  - distance_to_my_cluster

        # assign reward to respective timesteps
        n_envs = len(buffer.rewards[1])
        n_episodes = int(reward.shape[0]/n_envs) 
        restacked_reward = [reward[n_envs * i: n_envs * (i+1)] for i in range(n_episodes)]
        reshaped_reward = np.transpose(np.concatenate(restacked_reward, axis = 1))
        self.replay_buffer.rewards[start:end, :] = reshaped_reward

        return mean_distance_to_my_cluster, mean_distance_to_other_cluster

         #return kmeans.inertia_ , compute_distance_between_trajectory_and_cluster(kmeans.cluster_centers_[0], kmeans.cluster_centers_[1])


    def train(self, gradient_steps: int, batch_size: int = 100) -> None:

        # for dim in range(self.replay_buffer.observations.shape[2]):
        #     print(self.replay_buffer.observations[0, :, dim])
        # print(self.critic_target.qf0[0].weight.dtype)

        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)

        # Update learning rate according to lr schedule
        self._update_learning_rate([self.actor.optimizer, self.critic.optimizer])

        # custom function to get our custom reward
        # modifies the replay buffer
        # after, we just continue with normal PPO
        # cluster_inertia, cluster_dist = self.generate_synthetic_reward()
        mean_distance_to_my_cluster, mean_distance_to_other_cluster = self.generate_synthetic_reward()
        self.logger.record("train/mean_distance_my_cluster", mean_distance_to_my_cluster)
        self.logger.record("train/mean_distance_other_cluster", mean_distance_to_other_cluster)
        self.mean_distance_my_cluster.append(mean_distance_to_my_cluster)
        self.mean_distance_other_cluster.append(mean_distance_to_other_cluster)
        self.timesteps.append(self.num_timesteps)



        actor_losses, critic_losses = [], []
        for _ in range(gradient_steps):
            self._n_updates += 1
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

            with th.no_grad():
                # Select action according to policy and add clipped noise
                noise = replay_data.actions.clone().data.normal_(0, self.target_policy_noise)
                noise = noise.clamp(-self.target_noise_clip, self.target_noise_clip)
                next_actions = (self.actor_target(replay_data.next_observations) + noise).clamp(-1, 1)

                # Compute the next Q-values: min over all critics targets
                next_q_values = th.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                target_q_values = replay_data.rewards.to(torch.float32) + (1 - replay_data.dones) * self.gamma * next_q_values

            # Get current Q-values estimates for each critic network
            current_q_values = self.critic(replay_data.observations, replay_data.actions)

            # Compute critic loss
            critic_loss = sum(F.mse_loss(current_q, target_q_values) for current_q in current_q_values)
            critic_losses.append(critic_loss.item())

            # Optimize the critics
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            # Delayed policy updates
            if self._n_updates % self.policy_delay == 0:
                # Compute actor loss
                actor_loss = -self.critic.q1_forward(replay_data.observations, self.actor(replay_data.observations)).mean()
                actor_losses.append(actor_loss.item())

                # Optimize the actor
                self.actor.optimizer.zero_grad()
                actor_loss.backward()
                self.actor.optimizer.step()

                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                polyak_update(self.actor.parameters(), self.actor_target.parameters(), self.tau)
                # Copy running stats, see GH issue #996
                polyak_update(self.critic_batch_norm_stats, self.critic_batch_norm_stats_target, 1.0)
                polyak_update(self.actor_batch_norm_stats, self.actor_batch_norm_stats_target, 1.0)

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        if len(actor_losses) > 0:
            self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        # self.replay_buffer.reset()

    def learn(
        self: SelfCausalTD3,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        tb_log_name: str = "CausalTD3",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfCausalTD3:
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )

    def _excluded_save_params(self) -> List[str]:
        return super()._excluded_save_params() + ["actor", "critic", "actor_target", "critic_target"]  # noqa: RUF005

    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        state_dicts = ["policy", "actor.optimizer", "critic.optimizer"]
        return state_dicts, []