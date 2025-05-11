import os
import gin
import numpy as np
import torch
from torch.distributions import Beta, Bernoulli
from actor_nns import AFPPRNN


@gin.configurable
class AFPPAgent(torch.nn.Module):
    def __init__(self,
                 n_pulses: int,
                 targets: dict,
                 model_cls,
                 env_handler,
                 delta_verify=0,
                 pulse_min=0,
                 pulse_max=0.5):

        super(AFPPAgent, self).__init__()

        self.max_n_pulses = n_pulses
        self.targets = np.array(targets)
        self.n_targets = self.targets.size
        action_names = ['pulse']
        self.env_handler = env_handler()

        self.delta_verify = delta_verify
        self.n_common_actions = len(action_names)

        self.n_actions_per_level = 1

        self.action_ranges = np.array([pulse_min, pulse_max])

        for i in range(self.n_targets):
            action_names.append(f'verify_{i}')
            self.action_ranges = np.vstack([self.action_ranges, [0, targets[i] + 0.06]])

        self.action_names = action_names
        self.last_action = None
        input_common_size, input_per_level_size = self.env_handler.state_size

        input_common_size += self.n_common_actions
        input_per_level_size += self.n_actions_per_level

        self.model = model_cls(input_common_size=input_common_size,
                               input_per_level_size=input_per_level_size,
                               targets=np.array(targets),
                               output_common_size=self.n_common_actions * 2,
                               output_per_level_size=self.n_actions_per_level * 2
                               )

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        self.last_action_common = np.zeros(self.n_common_actions)
        self.last_action_per_level = np.zeros((self.n_targets, self.n_actions_per_level))

    def get_policy(self, obs_common, obs_per_level):
        out, values = self.model(obs_common, obs_per_level, critic=True)
        a = 1 + torch.nn.Softplus()(out[..., :-1:2])
        b = 1 + torch.nn.Softplus()(out[..., 1:-1:2])

        p = torch.nn.Sigmoid()(out[..., -1])
        return Beta(a, b), Bernoulli(p), values

    def act(self, chip, p_ind, verbose=False):
        obs_common, obs_per_level = self.env_handler.get_obs(p_ind)

        obs_common = np.hstack((obs_common, self.last_action_common))
        obs_per_level = np.hstack((obs_per_level, self.last_action_per_level))

        obs_common = torch.tensor(obs_common).to(self.device)
        obs_per_level = torch.tensor(obs_per_level).to(self.device)

        output_dict = {}
        raw_output, _ = self.model.single_step(obs_common, obs_per_level)
        raw_output = raw_output.view(-1)

        a = 1 + torch.nn.Softplus()(raw_output[:-1:2]).cpu().detach().numpy()
        b = 1 + torch.nn.Softplus()(raw_output[1:-1:2]).cpu().detach().numpy()
        output_dict['raw'] = a / (a + b)

        output_dict['skip'] = {}
        p = torch.nn.Sigmoid()(raw_output[-1]).cpu().detach().numpy()
        if p >= 0.5:
            output_dict['skip']['value'] = 1
            output_dict['raw'] = np.zeros(len(output_dict['raw']))
            self.last_action_common = np.zeros(self.n_common_actions)
            self.last_action_per_level = np.zeros((self.n_targets, self.n_actions_per_level))
            if verbose:
                with np.printoptions(precision=1, suppress=True):
                    print(f'Skip prob: {np.round(p)}')
            return []

        output_dict['skip']['value'] = 0

        self.last_action_common = output_dict['raw'][:1]
        self.last_action_per_level = output_dict['raw'][1:].reshape(-1, self.n_actions_per_level, order='F')

        self.rescale_actions(actions_dict=output_dict)

        pulse = output_dict['value'][0]
        v_verify_arr = np.array([output_dict['value'][1 + i] for i in range(self.n_targets)])
        reading_point_arr = self.targets - self.delta_verify


        if verbose:
            with np.printoptions(precision=4, suppress=True):
                print(
                    f'Targets: {self.targets}, Verify: {v_verify_arr}, Skip prob: {p:.2f}, Pulse: {pulse:.3f}')

        self.env_handler.act(chip=chip, pulse=pulse, v_verify_arr=v_verify_arr, read_levels=reading_point_arr)

        return pulse

    def rescale_actions(self, actions_dict: dict):
        actions_dict['value'] = actions_dict['raw'] * (self.action_ranges[:, 1] - self.action_ranges[:, 0])
        actions_dict['value'] += self.action_ranges[:, 0]

    def step(self, chip, p_ind):
        obs_common, obs_per_level = self.env_handler.get_obs(p_ind)

        obs_common = np.hstack((obs_common, self.last_action_common))
        obs_per_level = np.hstack((obs_per_level, self.last_action_per_level))

        obs_common = torch.tensor(obs_common).to(self.device)
        obs_per_level = torch.tensor(obs_per_level).to(self.device)

        raw_output, critic_value = self.model.single_step(obs_common, obs_per_level)
        output_dict = self._process_nn_output(raw_output=raw_output.view(-1))
        output_dict['critic_value'] = critic_value.cpu().detach().numpy().flatten()
        output_dict['obs_common'] = obs_common.cpu().detach().numpy().flatten()
        output_dict['obs_per_level'] = obs_per_level.cpu().detach().numpy().reshape(-1, obs_per_level.shape[-1])
        pulse = output_dict['value'][0]
        v_verify_arr = np.array([output_dict['value'][1 + i] for i in range(self.n_targets)])
        reading_point_arr = self.targets - self.delta_verify


        if output_dict['skip']['value'] == 1:
            self.last_action_common = np.zeros(self.n_common_actions)

            self.last_action_per_level = np.zeros((self.n_targets, self.n_actions_per_level))
            return output_dict
        self.env_handler.act(chip=chip, pulse=pulse, v_verify_arr=v_verify_arr, read_levels=reading_point_arr)

        self.last_action_common = output_dict['raw'][:1]
        self.last_action_per_level = output_dict['raw'][1:].reshape(-1, self.n_actions_per_level, order='F')

        return output_dict

    def reset(self, chip, block_idx, wl_idx, sub_levels_indicator_vec):
        self.model.reset()
        self.last_action_common[:] = 0
        self.last_action_per_level[:] = 0

        self.env_handler.reset(chip=chip, block_idx=block_idx, wl_idx=wl_idx,
                               sub_levels_indicator_vec=sub_levels_indicator_vec,
                               read_levels=self.targets - self.delta_verify)

    def _process_nn_output(self, raw_output):

        output = dict()

        output['skip'] = {}
        p = torch.nn.Sigmoid()(raw_output[-1])
        skip_policy = Bernoulli(p)
        skip_s = skip_policy.sample()
        output['skip']['log_prob'] = skip_policy.log_prob(skip_s).cpu().detach().numpy().item()
        output['skip']['raw'] = p.cpu().detach().numpy().copy().item()
        output['skip']['value'] = skip_s.cpu().detach().numpy().copy().item()

        a = 1 + torch.nn.Softplus()(raw_output[:-1:2])
        b = 1 + torch.nn.Softplus()(raw_output[1:-1:2])
        policy = Beta(a, b)
        s = policy.sample()
        output['mean_dist'] = policy.mean
        output['std_dist'] = policy.stddev
        output['raw'] = s.cpu().numpy().copy()
        output['log_prob'] = policy.log_prob(s).cpu().detach().numpy()

        self.rescale_actions(actions_dict=output)
        return output

    @classmethod
    def load(cls, folder_path, episode):
        config_path = os.path.join(folder_path, 'config.gin.yaml')
        assert os.path.exists(config_path)
        gin.parse_config_file(config_path, skip_unknown=True)
        agent = cls()
        ckpt_path = os.path.join(folder_path, 'models', f'model_at_epoch_{episode}.mdl')
        assert os.path.exists(ckpt_path)
        checkpoint = torch.load(ckpt_path, map_location=agent.device)
        agent.load_state_dict(checkpoint)
        return agent
