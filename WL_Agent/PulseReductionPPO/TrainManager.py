import os
import gin
import torch
import concurrent
import numpy as np
from copy import deepcopy

from buffers import PPOMemory, EpisodeData


@gin.configurable
class PPOTrainManager:

    def __init__(self,
                 experiment_dir,
                 wl_idx,
                 agent,
                 chips,
                 device,
                 min_l_2,
                 ckpt_freq=50,
                 num_episodes=100,
                 entropy_alpha=0.1,
                 reward_symmetry_factor=0.1,
                 policy_clip=0.2,
                 batch_size=32,
                 n_update_epochs=1,
                 max_episodes=64,
                 lr=8e-5,
                 debug_mode=False,
                 init_agent_from_ckpt=None,
                 n_batches_per_epoch=5,
                 decay_lr=3000,
                 decay_asymmetry=1000,
                 decay_entropy=1000
                 ):

        self.memory = PPOMemory(max_episodes=max_episodes)
        self.batch_size = batch_size
        self.debug_mode = debug_mode
        self.start_episode = 1
        self.initial_entropy_alpha = entropy_alpha
        self.entropy_alpha = entropy_alpha
        self.initial_reward_symmetry_factor = reward_symmetry_factor
        self.reward_symmetry_factor = reward_symmetry_factor
        self.initial_lr = lr

        self.experiment_dir = experiment_dir
        self.agent = agent
        self.n_targets = agent.n_targets
        self.device = device
        self.policy_clip = policy_clip
        self.num_episodes = num_episodes
        self.optimizers = {}
        self.episode_log = {}
        self.optimizer = torch.optim.Adam(self.agent.parameters(), lr, eps=1e-5)
        self.chips = chips
        self.wl_idx = wl_idx
        self.ckpt_freq = ckpt_freq
        self.n_update_epochs = n_update_epochs
        self.n_batches_per_epoch = n_batches_per_epoch

        self.train_n_steps = self.agent.max_n_pulses

        self.decay_lr = decay_lr
        self.decay_asymmetry = decay_asymmetry
        self.decay_entropy = decay_entropy

        self.min_l2 = min_l_2

        if init_agent_from_ckpt is not None:
            checkpoint = torch.load(init_agent_from_ckpt, map_location=self.device)
            self.agent.load_state_dict(checkpoint)

    def train_agent(self):
        for episode in range(self.start_episode, self.num_episodes):

            self.episode_log = {}
            self.single_episode()

            self.agent.train()
            self.update_model()
            self.agent.eval()
            self.update_learning_params(episode=episode)

            if episode % self.ckpt_freq == 0:
                self._eval_agent(episode=episode)
                if not self.debug_mode:
                    self.save_checkpoint(episode_num=episode)

                print("----------------------------------------------")
                print(f'Episode {episode} evaluation:')
                print("----------------------------------------------")
                for k, v in self.episode_log.items():
                    print(f'{k} : {v}')

                print("----------------------------------------------")

    def update_learning_params(self, episode):

        self.entropy_alpha = self.initial_entropy_alpha / np.sqrt(1 + episode / self.decay_entropy)
        self.reward_symmetry_factor = self.initial_reward_symmetry_factor / np.sqrt(1 + episode / self.decay_entropy)

        self.episode_log['entropy_alpha'] = self.entropy_alpha
        self.episode_log['reward_symmetry_factor'] = self.reward_symmetry_factor
        for g in self.optimizer.param_groups:
            if g['lr'] >= 5e-5:
                g['lr'] = self.initial_lr / np.sqrt(1 + episode / self.decay_lr)

            self.episode_log['lr'] = g['lr']

    def single_chip_episode(self, chip, agent):
        targets_mv, sub_level_indicator_vec, block_idx, wl_idx = self.init_episode(chip=chip)

        self.episode_log['block idx'] = block_idx
        self.episode_log['wl idx'] = wl_idx
        observations_common = []
        observations_per_level = []
        actions = []
        log_probs = []
        values = np.array([])
        skip_pulses = []

        self.episode_log['skip param'] = 0

        dist_mean = np.zeros(self.agent.action_ranges.shape[0])
        dist_std = np.zeros(self.agent.action_ranges.shape[0])

        agent.reset(chip=chip, block_idx=block_idx, wl_idx=wl_idx, sub_levels_indicator_vec=sub_level_indicator_vec)

        pulse_index = 0

        for step_index in range(self.agent.max_n_pulses):
            step_output = agent.step(chip=chip, p_ind=pulse_index)

            self.episode_log['skip param'] += step_output['skip']['raw']

            observations_common.append(step_output['obs_common'])
            observations_per_level.append(step_output['obs_per_level'])
            actions.append(step_output['raw'])
            log_probs.append(step_output['log_prob'])
            values = np.append(values, step_output['critic_value'])
            skip_pulses.append([step_output['skip']['raw'], step_output['skip']['value']])

            pulse_index += 1 - step_output['skip']['value']
            dist_mean += step_output['mean_dist'].cpu().detach().numpy()
            dist_std += step_output['std_dist'].cpu().detach().numpy()

        self.n_pulses = pulse_index
        self.episode_log['skip param'] /= self.agent.max_n_pulses

        read_for_return = chip.read_voltages(block_idx=block_idx, wl_idx=wl_idx)
        return_val, mean_error, std, mean_l2 = self._calc_return(targets_mv=targets_mv,
                                                                 sub_level_indicator_vec=sub_level_indicator_vec,
                                                                 read_cells_mv=read_for_return)

        returns = np.full(shape=self.agent.max_n_pulses, fill_value=return_val)
        print(f'Episode return = {return_val}')

        advantages = self._calc_advantages(return_val, values)

        episode_data = EpisodeData(observations_common=np.array(observations_common),
                                   observations_per_level=np.array(observations_per_level),
                                   actions=np.array(actions),
                                   log_probs=np.array(log_probs),
                                   returns=np.array(returns),
                                   advantages=advantages,
                                   skip_pulses=np.array(skip_pulses))

        return episode_data, return_val, mean_error, std, dist_mean, dist_std, mean_l2

    def single_episode(self):
        if len(self.chips) == 1:
            agents = [self.agent]
        else:
            agents = [deepcopy(self.agent) for _ in range(len(self.chips))]

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(self.chips)) as executor:
            results = list(executor.map(self.single_chip_episode, self.chips, agents))

        for i, r in enumerate(results):
            self.memory.store_memory(r[0])
            return_val, mean_error, std, dist_mean, dist_std, mean_l2 = r[1:]

            if i == 0:
                for j in range(len(dist_std)):
                    self.episode_log[f'Distribution - std {self.agent.action_names[j]}'] = dist_std[
                                                                                               j] / self.agent.max_n_pulses

            self.episode_log[f'Chip {i} TRAIN Mean mean error'] = mean_error
            self.episode_log[f'Chip {i} TRAIN return'] = return_val
            self.episode_log[f'Chip {i} TRAIN mean STD'] = std
            self.episode_log[f'Chip {i} TRAIN mean L2'] = mean_l2

    def init_episode(self, chip):

        targets_mv = self.agent.targets
        sub_level_indicator_vec = np.random.randint(-1, targets_mv.size, chip.n_cells_per_wl)

        block_idx = 1  # whitened

        return targets_mv, sub_level_indicator_vec, block_idx, self.wl_idx

    def update_model(self):

        observations_common_arr = torch.tensor(self.memory.observations_common,
                                               dtype=torch.float).to(self.device)
        observations_per_level_arr = torch.tensor(self.memory.observations_per_level,
                                                  dtype=torch.float).to(self.device)
        returns_arr = torch.tensor(self.memory.returns, dtype=torch.float).to(self.device)
        actions_arr = torch.tensor(self.memory.actions, dtype=torch.float).to(self.device)

        advantage_mean = np.mean(self.memory.advantages)
        advantage_std = np.std(self.memory.advantages) + 1e-6

        advantages_arr = torch.tensor((self.memory.advantages - advantage_mean) /
                                      advantage_std, dtype=torch.float).to(self.device)

        old_probs_arr = torch.tensor(self.memory.log_probs, dtype=torch.float).to(self.device).sum(axis=-1)

        skipped_pulses_arr = torch.tensor(self.memory.skip_pulses[..., 1], dtype=torch.float).to(self.device)
        skip_old_probs_arr = torch.tensor(self.memory.skip_pulses[..., 1], dtype=torch.float).to(self.device) \
                             * torch.log(
            torch.tensor(self.memory.skip_pulses[..., 0], dtype=torch.float).to(self.device)) \
                             + torch.tensor(1 - self.memory.skip_pulses[..., 1], dtype=torch.float).to(self.device) \
                             * torch.log(
            torch.tensor(1 - self.memory.skip_pulses[..., 0], dtype=torch.float).to(self.device))

        for _ in range(self.n_update_epochs):
            for _ in range(self.n_batches_per_epoch):
                episodes = np.random.randint(self.memory.n_episodes, size=self.batch_size)
                episode_starts = np.random.randint(self.agent.max_n_pulses - self.train_n_steps + 1,
                                                   size=self.batch_size)
                indices = episode_starts[:, np.newaxis] + np.arange(self.train_n_steps)
                row_indices = episodes[:, np.newaxis]

                observations_common = observations_common_arr[row_indices, indices, :]
                observations_per_level = observations_per_level_arr[row_indices, indices, :]
                old_probs = old_probs_arr[row_indices, indices]
                actions = actions_arr[row_indices, indices]
                skipped_pulses = skipped_pulses_arr[row_indices, indices]
                skip_old_probs = skip_old_probs_arr[row_indices, indices]

                self.agent.model.reset()
                new_policy, skip_policy, critic_value = self.agent.get_policy(observations_common,
                                                                              observations_per_level)

                critic_value = torch.squeeze(critic_value)
                ret = returns_arr[row_indices, indices]
                advantage = advantages_arr[row_indices, indices]

                new_probs = new_policy.log_prob(actions).sum(axis=-1)
                entropy = new_policy.entropy()[..., :1 + self.agent.n_targets]
                entropy = (entropy[..., 0] + entropy[..., 1:].mean(axis=-1)) / 2

                skip_new_probs = skip_policy.log_prob(skipped_pulses)
                skip_entropy = skip_policy.entropy()

                mask = skipped_pulses == 1
                new_probs = new_probs.masked_fill(mask, 0)
                old_probs = old_probs.masked_fill(mask, 0)
                entropy = entropy.masked_fill(mask, 0)

                entropy = torch.exp(skip_new_probs) * entropy + skip_entropy

                prob_ratio = torch.exp(new_probs + skip_new_probs - old_probs - skip_old_probs)

                weighted_probs = advantage * prob_ratio
                weighted_clipped_probs = torch.clamp(prob_ratio,
                                                     1 - self.policy_clip,
                                                     1 + self.policy_clip) * advantage
                actor_loss = -torch.min(weighted_probs, weighted_clipped_probs).mean()

                critic_loss = ((critic_value - torch.tensor(ret, device=self.device)) ** 2).mean()
                total_loss = actor_loss - self.entropy_alpha * entropy.mean() + critic_loss

                self.episode_log['actor loss'] = actor_loss.detach()
                self.episode_log['critic loss'] = critic_loss.detach()
                self.episode_log['advantage'] = advantage.detach().mean()

                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm(self.agent.parameters(), 0.5)
                self.optimizer.step()

    def _calc_return(self, targets_mv, sub_level_indicator_vec, read_cells_mv):
        std = 0
        mean_error = 0
        final_rewards = np.zeros(shape=targets_mv.size, dtype=np.float64)
        mean_l2 = 0
        for i, t in enumerate(targets_mv):
            target_cells = read_cells_mv[sub_level_indicator_vec == i]
            cur_voltage_diff_mv = (target_cells - t)

            normalized_cur_voltage_diff_mv = cur_voltage_diff_mv
            final_rewards[i] = np.sqrt(np.square(normalized_cur_voltage_diff_mv).mean())
            mean_l2 += np.sqrt(np.square(cur_voltage_diff_mv).mean())
            std += np.std(target_cells)
            mean_error += np.mean(cur_voltage_diff_mv)

        reward_final = - np.maximum(final_rewards.mean(), self.min_l2) - 0.0005 * self.n_pulses

        if self.reward_symmetry_factor != 0:
            reward_final -= self.reward_symmetry_factor * np.std(final_rewards)

        std /= targets_mv.size
        mean_error /= targets_mv.size
        mean_l2 /= targets_mv.size

        return reward_final, mean_error, std, mean_l2

    @staticmethod
    def _calc_advantages(return_val, values):
        returns = np.zeros_like(values)
        returns[-1] = return_val
        values = np.append(values, 0)
        advantages = np.flip(np.flip(returns + values[1:] - values[:-1], 0).cumsum(), 0)
        return advantages

    def save_checkpoint(self, episode_num):
        """
        This function saves the model for inference.
        :param episode_num:
        :return:
        """
        # ---------------- SAVE MODEL ----------------------
        # first build the path to the model file
        path_to_model = os.path.abspath(os.path.join(self.experiment_dir, "models"))

        # Check if the path exists
        os.makedirs(path_to_model, exist_ok=True)

        # Prepare the model file name
        # timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        model_file_name = f'model_at_epoch_{episode_num}.mdl'
        model_full_path = os.path.join(path_to_model, model_file_name)

        # Save the model
        torch.save(self.agent.state_dict(), model_full_path)
        # logger(to_db={'models_path': model_full_path}, append=True)

        # ---------------- SAVE CHECKPOINT -----------------
        # First delete the previous checkpoint
        for f_name in os.listdir(self.experiment_dir):
            candidate_f = os.path.join(self.experiment_dir, f_name)
            if os.path.isfile(candidate_f) and f_name.endswith('.ckpt'):
                os.remove(candidate_f)

        # Checkpoint file name
        checkpoint_file_name = f'checkpoint_at_epoch_{episode_num}.ckpt'
        checkpoint_full_path = os.path.join(self.experiment_dir, checkpoint_file_name)

        # Save the checkpoint
        checkpoint_dict = {'episode': episode_num,
                           'model_state_dict': self.agent.state_dict(),
                           'optimizer_state_dicts': self.optimizer.state_dict(), }

        torch.save(checkpoint_dict, checkpoint_full_path)

    def load_checkpoint(self, path):
        print(f'resuming from checkpoint. path : {path}')

        # Load the checkpoint
        ckpt_path = os.path.join(self.experiment_dir, path)
        checkpoint = torch.load(ckpt_path, map_location=self.device)

        # Update the epoch index
        self.start_episode = checkpoint['episode'] + 1

        self.optimizer.load_state_dict(checkpoint['optimizer_state_dicts'])
        self.agent.load_state_dict(checkpoint['model_state_dict'])
        self.agent.model.train()

    def _eval_agent(self, episode):
        print(f"--------- EVAL at episode {episode} ---------")
        chip = np.random.choice(self.chips)
        targets_mv, sub_level_indicator_vec, block_idx, wl_idx = self.init_episode(chip)
        self.agent.reset(chip=chip, block_idx=block_idx, wl_idx=wl_idx,
                         sub_levels_indicator_vec=sub_level_indicator_vec)
        pulse_index = 0
        pulse_seq = np.array([])
        for step_index in range(self.agent.max_n_pulses):
            with torch.no_grad():
                p = self.agent.act(chip=chip, p_ind=pulse_index, verbose=True)
                if p:
                    pulse_index += 1
                pulse_seq = np.append(pulse_seq, p)

        read = chip.read_voltages(block_idx=block_idx, wl_idx=wl_idx)
        mean_errs = np.zeros_like(targets_mv)
        stds = np.zeros_like(targets_mv)
        mean_l2 = 0
        for i, t in enumerate(targets_mv):
            t_read = read[sub_level_indicator_vec == i]
            mean_l2 += np.sqrt(np.square(t_read - t).mean())
            mean_errs[i] = np.mean(t_read) - t
            stds[i] = np.std(t_read)

        print(f'block {block_idx}, WL {wl_idx} mean errs {mean_errs}, STDs {stds}')

        pulse_gradient = pulse_seq[1:] - pulse_seq[:-1]


        self.episode_log['EVAL Mean err'] = np.mean(np.abs(mean_errs))
        self.episode_log['EVAL STD'] = np.mean(stds)
        self.episode_log['EVAL L2'] = mean_l2 / targets_mv.size
