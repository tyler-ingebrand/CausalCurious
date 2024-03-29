import warnings
from typing import Any, Dict, Optional, Type, TypeVar, Union

import numpy as np
import torch
import torch as th
from gym import spaces
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnv
from torch.nn import functional as F

from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import explained_variance, get_schedule_fn, obs_as_tensor

from .clustering_functions import cluster, format_obs, compute_distance_between_trajectory_and_cluster, \
    get_distances_between_trajectories_and_clusters, normalize_distances, get_change_in_distance

import time
from .CustomPPOPolicies import *


SelfCausalCuriousPPO = TypeVar("SelfCausalCuriousPPO", bound="CausalCuriousPPO")


class CausalCuriousPPO(OnPolicyAlgorithm):
    """
    :param policy: The policy model to use (MlCausalCuriousPPOlicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: The learning rate, it can be a function
        of the current progress remaining (from 1 to 0)
    :param episode_length: The length of a full episode in the env. Episodes should not be truncated.
    :param episodes_per_update: How many episodes are needed for each update.
        Should be a largish number so clustering can work
    :param n_epochs: Number of epoch when optimizing the surrogate loss
    :param gamma: Discount factor
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
    :param clip_range: Clipping parameter, it can be a function of the current progress
        remaining (from 1 to 0).
    :param clip_range_vf: Clipping parameter for the value function,
        it can be a function of the current progress remaining (from 1 to 0).
        This is a parameter specific to the OpenAI implementation. If None is passed (default),
        no clipping will be done on the value function.
        IMPORTANT: this clipping depends on the reward scaling.
    :param normalize_advantage: Whether to normalize or not the advantage
    :param ent_coef: Entropy coefficient for the loss calculation
    :param vf_coef: Value function coefficient for the loss calculation
    :param max_grad_norm: The maximum value for the gradient clipping
    :param use_sde: Whether to use generalized State Dependent Exploration (gSDE)
        instead of action noise exploration (default: False)
    :param sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
        Default: -1 (only sample at the beginning of the rollout)
    :param target_kl: Limit the KL divergence between updates,
        because the clipping is not enough to prevent large update
        see issue #213 (cf https://github.com/hill-a/stable-baselines/issues/213)
        By default, there is no limit on the kl div.
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: Verbosity level: 0 for no output, 1 for info messages (such as device or wrappers used), 2 for
        debug messages
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    """

    policy_aliases: Dict[str, Type[BasePolicy]] = {
        "MlpPolicy": ActorCriticPolicy,
        "CnnPolicy": ActorCriticCnnPolicy,
        "MultiInputPolicy": MultiInputActorCriticPolicy,
    }

    def __init__(
        self,
        policy: Union[str, Type[ActorCriticPolicy]],
        env: Union[GymEnv, str],
        episode_length:int,
        episodes_per_update:int,
        learning_rate: Union[float, Schedule] = 3e-4,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: Union[float, Schedule] = 0.2,
        clip_range_vf: Union[None, float, Schedule] = None,
        normalize_advantage: bool = True,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        target_kl: Optional[float] = None,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        debug_dir=None,
    ):
        n_envs = env.num_envs if isinstance(env, VecEnv) else 1
        n_steps = episode_length * episodes_per_update
        batch_size = int(n_envs * n_steps / 100)
        print("Found {} envs with episode length {} and {} episodes per update step for a total of {} steps per update per env.".format(n_envs, episode_length, episodes_per_update, n_steps))
        print("Batch size set to {}".format(batch_size))
        super().__init__(
            policy,
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            gamma=gamma,
            gae_lambda=gae_lambda,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            seed=seed,
            _init_setup_model=False,
            supported_action_spaces=(
                spaces.Box,
                spaces.Discrete,
                spaces.MultiDiscrete,
                spaces.MultiBinary,
            ),
        )

        # Sanity check, otherwise it will lead to noisy gradient and NaN
        # because of the advantage normalization
        if normalize_advantage:
            assert (
                batch_size > 1
            ), "`batch_size` must be greater than 1. See https://github.com/DLR-RM/stable-baselines3/issues/440"

        if self.env is not None:
            # Check that `n_steps * n_envs > 1` to avoid NaN
            # when doing advantage normalization
            buffer_size = self.env.num_envs * self.n_steps
            assert buffer_size > 1 or (
                not normalize_advantage
            ), f"`n_steps * n_envs` must be greater than 1. Currently n_steps={self.n_steps} and n_envs={self.env.num_envs}"
            # Check that the rollout buffer size is a multiple of the mini-batch size
            untruncated_batches = buffer_size // batch_size
            if buffer_size % batch_size > 0:
                warnings.warn(
                    f"You have specified a mini-batch size of {batch_size},"
                    f" but because the `RolloutBuffer` is of size `n_steps * n_envs = {buffer_size}`,"
                    f" after every {untruncated_batches} untruncated mini-batches,"
                    f" there will be a truncated mini-batch of size {buffer_size % batch_size}\n"
                    f"We recommend using a `batch_size` that is a factor of `n_steps * n_envs`.\n"
                    f"Info: (n_steps={self.n_steps} and n_envs={self.env.num_envs})"
                )
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.clip_range = clip_range
        self.clip_range_vf = clip_range_vf
        self.normalize_advantage = normalize_advantage
        self.target_kl = target_kl
        self.episode_starts = []
        if _init_setup_model:
            self._setup_model()

        self.mean_distance_my_cluster = []
        self.mean_distance_other_cluster = []
        self.timesteps = []
        self.debug_dir = debug_dir
    def _setup_model(self) -> None:
        super()._setup_model()

        # Initialize schedules for policy/value clipping
        self.clip_range = get_schedule_fn(self.clip_range)
        if self.clip_range_vf is not None:
            if isinstance(self.clip_range_vf, (float, int)):
                assert self.clip_range_vf > 0, "`clip_range_vf` must be positive, " "pass `None` to deactivate vf clipping"

            self.clip_range_vf = get_schedule_fn(self.clip_range_vf)

    def get_episode_starts(self):
        buffer = self.rollout_buffer

        for i in range(len(buffer.episode_starts)):
            if (buffer.episode_starts[i] != 0).any():
                if not (buffer.episode_starts[i] != 0).all():
                    raise Exception("Episodes do not have the same length, TODO Can we handle this?")
                self.episode_starts.append(i)
        self.episode_starts.append(len(buffer.episode_starts))  # append the length of the buffer so we have a marker at the start and end of each episode

    def generate_synthetic_reward(self):
        buffer = self.rollout_buffer

        # find episode starts if we have not already, IE during the first update
        if len(self.episode_starts) == 0:
            self.get_episode_starts()

        # get obs data. Remove robot states since we dont care about those for clyustering
        # Robot obs are the first 27 dims in statespace
        obs = buffer.observations[:, :, 27:]

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
        print(kmeans.labels_)

        # compute distances between current cluster and other cluster
        distance_to_my_cluster, distance_to_other_cluster = get_distances_between_trajectories_and_clusters(kmeans.labels_,
                                                                                                            kmeans.cluster_centers_,
                                                                                                            data,
                                                                                                            verbose=False,
                                                                                                            plot=False)

        if np.isnan(distance_to_my_cluster).any():
            print("Distance my cluster contains nan")
            print(distance_to_my_cluster)
        if np.isnan(distance_to_other_cluster).any():
            print("Distance other cluster contains nan")
            print(distance_to_other_cluster)

        mean_distance_to_my_cluster = np.mean(distance_to_my_cluster)
        mean_distance_to_other_cluster = np.mean(distance_to_other_cluster)

        # change_in_distance_to_my_cluster, change_in_distance_to_other_cluster = get_change_in_distance(distance_to_my_cluster, distance_to_other_cluster)


        # normalize distances
        distance_to_my_cluster = normalize_distances(distance_to_my_cluster)
        distance_to_other_cluster = normalize_distances(distance_to_other_cluster)
        if np.isnan(distance_to_my_cluster).any():
            print("normed Distance my cluster contains nan")
            print(distance_to_my_cluster)
        if np.isnan(distance_to_other_cluster).any():
            print("normed Distance other cluster contains nan")
            print(distance_to_other_cluster)
        # create reward
        # reward = 2 * distance_to_other_cluster - distance_to_my_cluster
        ## TEST: run this with only the distance to the other cluster
        reward = 3 * distance_to_other_cluster - distance_to_my_cluster
        if np.isnan(reward).any():
            print("reward contains nan")
            print(reward)
        # assign reward to respective timesteps
        n_envs = len(buffer.rewards[1])
        n_episodes = int(reward.shape[0]/n_envs)
        restacked_reward = [reward[n_envs * i: n_envs * (i+1)] for i in range(n_episodes)]
        reshaped_reward = np.transpose(np.concatenate(restacked_reward, axis = 1))
        self.rollout_buffer.rewards = reshaped_reward

        return mean_distance_to_my_cluster, mean_distance_to_other_cluster

         #return kmeans.inertia_ , compute_distance_between_trajectory_and_cluster(kmeans.cluster_centers_[0], kmeans.cluster_centers_[1])

    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        """

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

        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []

        continue_training = True

        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long().flatten()

                # Re-sample the noise matrix because the log_std has changed
                if self.use_sde:
                    self.policy.reset_noise(self.batch_size)

                values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
                if torch.isnan(log_prob).any():
                    print("Log prob nan")
                    print(log_prob)
                if torch.isnan(values).any():
                    print("values nan")
                    print(values)
                values = values.flatten()
                # Normalize advantage
                advantages = rollout_data.advantages
                # Normalization does not make sense if mini batchsize == 1, see GH issue #325
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                if torch.isnan(advantages).any():
                    print("advantages nan")
                    print(advantages)

                # ratio between old and new policy, should be one at the first iteration
                ratio = th.exp(log_prob - rollout_data.old_log_prob)
                if torch.isnan(ratio).any():
                    print("ratio nan")
                    print(ratio)
                # clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()
                if torch.isnan(policy_loss).any():
                    print("policy_loss nan")
                    print(policy_loss)
                # Logging
                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                if torch.isnan(values).any():
                    print("values nan")
                    print(values)
                if self.clip_range_vf is None:
                    # No clipping
                    values_pred = values
                else:
                    # Clip the difference between old and new value
                    # NOTE: this depends on the reward scaling
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                if torch.isnan(rollout_data.old_values).any():
                        print("rollout_data.old_values nan")
                        print(rollout_data.old_values)
                # Value loss using the TD(gae_lambda) target
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                if torch.isnan(rollout_data.returns).any():
                        print("(rollout_data.returns nan")
                        print(rollout_data.returns)
                if torch.isnan(value_loss).any():
                        print("value_loss nan")
                        print(value_loss)
                value_losses.append(value_loss.item())

                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)

                entropy_losses.append(entropy_loss.item())

                if torch.isnan(entropy_loss).any():
                        print("entropy_loss nan")
                        print(entropy_loss)
                #loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss
                loss = policy_loss +  self.vf_coef * value_loss

                # Calculate approximate form of reverse KL Divergence for early stopping
                # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                # and Schulman blog: http://joschu.net/blog/kl-approx.html
                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                # Optimization step
                self.policy.optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

            if not continue_training:
                break

        self._n_updates += self.n_epochs
        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        # Logs
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        # self.logger.record("train/average_cluster_distance", np.average(cluster_dist) )
        # self.logger.record("train/cluster_inertia", cluster_inertia)
        
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)

    def learn(
        self: SelfCausalCuriousPPO,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        tb_log_name: str = "CausalCuriousPPO",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfCausalCuriousPPO:

        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )
