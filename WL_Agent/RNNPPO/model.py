import os
import gin
import numpy as np
import torch
from torch.distributions import Beta
from actor_nns import AFPPNN, PulseOnlyNN


@gin.configurable
class AFPPAgent(torch.nn.Module):
    def __init__(self,
                 n_pulses: int,
                 targets,
                 model_cls,
                 env_handler,
                 delta_verify=0,
                 learn_verify_level=True,
                 learn_reading_point=False,
                 pulse_min=0,
                 pulse_max=0.5):

        super(AFPPAgent, self).__init__()

        self.n_pulses = n_pulses
        self.targets = np.array(targets)
        self.n_targets = self.targets.size
        action_names = ['pulse']
        self.env_handler = env_handler()
        self.delta_verify = delta_verify
        self.learn_verify_level = learn_verify_level
        self.n_common_actions = len(action_names)

        self.action_ranges = np.array([[pulse_min, pulse_max]])
        self.n_actions_per_level = 0
        if self.learn_verify_level:
            self.n_actions_per_level += 1
            for i in range(self.n_targets):
                action_names.append(f'verify_{i}')
                self.action_ranges = np.vstack([self.action_ranges, [0, targets[i] + 0.06]])

        self.learn_reading_point = learn_reading_point
        if learn_reading_point:
            self.n_actions_per_level += 1
            for i in range(self.n_targets):
                action_names.append(f'read_{i}')
                self.action_ranges = np.vstack([self.action_ranges, [0, targets[i] + 0.06]])

        self.action_names = action_names
        self.last_action = None
        input_common_size, input_per_level_size = self.env_handler.state_size

        input_common_size += self.n_common_actions
        input_per_level_size += self.n_actions_per_level

        targets_range = targets[-1] - targets[0]
        self.model = model_cls(input_common_size=input_common_size,
                               input_per_level_size=input_per_level_size,
                               n_targets=self.n_targets,
                               targets_range=targets_range,
                               output_common_size=self.n_common_actions * 2,
                               output_per_level_size=self.n_actions_per_level * 2
                               )

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        self.last_action_common = np.zeros(self.n_common_actions)
        self.last_action_per_level = np.zeros((self.n_targets, self.n_actions_per_level))

    def get_policy(self, obs_common, obs_per_level, targets):
        out, values = self.model(obs_common, obs_per_level, targets=targets, critic=True)
        a = 1 + torch.nn.Softplus()(out[..., ::2])
        b = 1 + torch.nn.Softplus()(out[..., 1::2])
        return Beta(a, b), values

    def act(self, chip, p_ind, targets, verbose=False):
        obs_common, obs_per_level = self.env_handler.get_obs(p_ind)

        obs_common = np.hstack((obs_common, self.last_action_common))
        obs_per_level = np.hstack((obs_per_level, self.last_action_per_level))

        obs_common = torch.tensor(obs_common).to(self.device)
        obs_per_level = torch.tensor(obs_per_level).to(self.device)

        output_dict = {}
        raw_output, _ = self.model.single_step(obs_common, obs_per_level, targets=targets)
        raw_output = raw_output.view(-1)
        a = 1 + torch.nn.Softplus()(raw_output[::2]).cpu().detach().numpy()
        b = 1 + torch.nn.Softplus()(raw_output[1::2]).cpu().detach().numpy()
        output_dict['raw'] = a / (a + b)

        self.rescale_actions(actions_dict=output_dict)
        pulse = output_dict['value'][0]
        self._update_last_action(output_dict=output_dict)

        v_verify_arr = targets - self.delta_verify
        reading_point_arr = targets - self.delta_verify

        if self.learn_verify_level:
            v_verify_arr = np.array(output_dict['value'][1:self.n_targets+1])

        if self.learn_reading_point:
            reading_point_arr = np.array(output_dict['value'][self.n_targets + 1:])

        if verbose:
            with np.printoptions(precision=3, suppress=True):
                print(f'Obs: {obs_per_level[:, 0].cpu().detach().numpy()},'
                              f' Read voltages: {reading_point_arr},'
                              f' Verify: {v_verify_arr}, Pulse: {pulse:.3f}')

        self.env_handler.act(chip=chip,
                             pulse=pulse,
                             v_verify_arr=v_verify_arr,
                             read_levels=reading_point_arr)

        return pulse

    def _update_last_action(self, output_dict):
        self.last_action_common = output_dict['raw'][:1]
        if self.learn_reading_point or self.learn_verify_level:
            self.last_action_per_level = output_dict['raw'][1:].reshape(
                -1, self.n_actions_per_level, order='F')

    def rescale_actions(self, actions_dict: dict):
        actions_dict['value'] = actions_dict['raw'] * (self.action_ranges[:, 1] - self.action_ranges[:, 0])
        actions_dict['value'] += self.action_ranges[:, 0]

    def step(self, chip, p_ind, targets):
        obs_common, obs_per_level = self.env_handler.get_obs(p_ind)

        obs_common = np.hstack((obs_common, self.last_action_common))
        obs_per_level = np.hstack((obs_per_level, self.last_action_per_level))

        obs_common = torch.tensor(obs_common).to(self.device)
        obs_per_level = torch.tensor(obs_per_level).to(self.device)

        raw_output, critic_value = self.model.single_step(obs_common, obs_per_level, targets=targets)
        output_dict = self._process_nn_output(raw_output=raw_output.view(-1))
        output_dict['critic_value'] = critic_value.cpu().detach().numpy().flatten()
        output_dict['obs_common'] = obs_common.cpu().detach().numpy().flatten()
        output_dict['obs_per_level'] = obs_per_level.cpu().detach().numpy().reshape(-1, obs_per_level.shape[-1])
        pulse = output_dict['value'][0]

        self._update_last_action(output_dict=output_dict)
        v_verify_arr = targets - self.delta_verify
        reading_point_arr = targets - self.delta_verify

        if self.learn_verify_level:
            v_verify_arr = np.array(output_dict['value'][1:self.n_targets+1])

        if self.learn_reading_point:
            reading_point_arr = np.array(output_dict['value'][self.n_targets + 1:])

        self.env_handler.act(chip=chip, pulse=pulse, v_verify_arr=v_verify_arr, read_levels=reading_point_arr)

        return output_dict

    def reset(self, chip, block_idx, wl_idx, sub_levels_indicator_vec, read_levels):
        self.model.reset()
        self.last_action_common[:] = 0
        self.last_action_per_level[:] = 0

        self.env_handler.reset(chip=chip,
                               block_idx=block_idx,
                               wl_idx=wl_idx,
                               sub_levels_indicator_vec=sub_levels_indicator_vec,
                               read_levels=read_levels)

    def _process_nn_output(self, raw_output):

        output = dict()
        a = 1 + torch.nn.Softplus()(raw_output[::2])
        b = 1 + torch.nn.Softplus()(raw_output[1::2])
        policy = Beta(a, b)
        s = policy.sample()
        output['mean_dist'] = policy.mean
        output['std_dist'] = policy.stddev
        output['log_prob'] = policy.log_prob(s).cpu().detach().numpy()
        s = s.cpu().numpy().copy()
        output['raw'] = s
        self.rescale_actions(actions_dict=output)
        return output

    @classmethod
    def load(cls, folder_path, episode, conf_name='config.gin.yaml'):
        config_path = os.path.join(folder_path, conf_name)
        assert os.path.exists(config_path)
        gin.parse_config_file(config_path, skip_unknown=True)
        agent = cls()
        ckpt_path = os.path.join(folder_path, 'models', f'model_at_epoch_{episode}.mdl')
        assert os.path.exists(ckpt_path)
        checkpoint = torch.load(ckpt_path, map_location=agent.device)
        agent.load_state_dict(checkpoint)
        return agent
