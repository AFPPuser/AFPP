import numpy as np
import torch
import gin


@gin.configurable()
class AFPPSNN(torch.nn.Module):
    def __init__(self,
                 hidden_size: int,
                 num_layers: int,
                 pulse_layers_size,
                 verify_layers_size,
                 critic_layer_size,
                 input_common_size: int,
                 input_per_level_size: int,
                 targets: np.ndarray,
                 output_common_size: int,
                 output_per_level_size: int,
                 rnn_type: torch.nn.RNN,
                 non_linearity_module=torch.nn.ReLU):

        super(AFPPSNN, self).__init__()

        self.n_targets = len(targets)
        self.targets = (targets - (targets[-1] + targets[0]) / 2) / (targets[-1] - targets[0])
        self.common_obser = input_common_size
        self.obser_per_level = input_per_level_size + 1
        self.hidden_size = hidden_size

        self.hidden_state = None

        self.rnn = rnn_type(input_size=self.common_obser + self.obser_per_level,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True)

        layers_size = [self.hidden_size * self.n_targets] + pulse_layers_size

        # Build the networks
        layers = []
        for i in range(1, len(layers_size)):
            layers.append(torch.nn.Linear(layers_size[i - 1], layers_size[i]))
            layers.append(non_linearity_module())

        layers.append(torch.nn.Linear(layers_size[-1], output_common_size + 1))

        self.pulse_layers = torch.nn.Sequential(*layers)

        layers_size = [self.hidden_size] + verify_layers_size
        layers = []
        for i in range(1, len(layers_size)):
            layers.append(torch.nn.Linear(layers_size[i - 1], layers_size[i]))
            layers.append(non_linearity_module())

        layers.append(torch.nn.Linear(layers_size[-1], output_per_level_size))

        self.verify_layers = torch.nn.Sequential(*layers)

        layers_size = [self.hidden_size * self.n_targets] + critic_layer_size

        # Build the network
        layers = []
        for i in range(1, len(layers_size)):
            layers.append(torch.nn.Linear(layers_size[i - 1], layers_size[i]))
            layers.append(non_linearity_module())

        layers.append(torch.nn.Linear(layers_size[-1], 1))
        self.critic = torch.nn.Sequential(*layers)

    def reset(self):
        self.hidden_state = None

    def single_step(self, in_data_common, in_data_per_level):
        with torch.no_grad():
            targets = torch.tensor(self.targets).float().to(in_data_common.device).view(-1, 1)

            output, self.hidden_state = self.rnn(
                torch.cat((in_data_common * torch.ones_like(targets), targets, in_data_per_level), dim=-1).float().view(
                    targets.shape[0], 1, -1),
                self.hidden_state)

            policy = self.verify_layers(output)
            reading_policy = policy[..., 2:]
            pulse_policy = self.pulse_layers(output.flatten())
            skip_policy = pulse_policy[-1]
            pulse_policy = torch.cat((pulse_policy[:-1], policy[..., :2].flatten()))
            critic = self.critic(output.flatten())

        return torch.cat((pulse_policy, reading_policy.flatten(), skip_policy.view(1)), dim=-1), critic

    def forward(self, in_data_common, in_data_per_level, critic=True):

        unit_vec = torch.ones((*in_data_common.shape[:-1], 1)).float().to(in_data_common.device)
        batch_size, n_steps = in_data_common.shape[:-1]
        input_size = in_data_common.shape[-1] + in_data_per_level.shape[-1] + 1

        input_rnn = torch.empty((self.n_targets * batch_size, n_steps, input_size)).to(in_data_common.device)
        input_agents = torch.empty((batch_size, n_steps, self.hidden_size * self.n_targets)).to(in_data_common.device)
        pulse_policy = torch.empty((batch_size, n_steps, 2 * (self.n_targets + 1))).to(in_data_common.device)

        for i in range(len(self.targets)):
            input_rnn[i * batch_size:(i + 1) * batch_size] = torch.cat(
                (in_data_common, self.targets[i] * unit_vec, in_data_per_level[:, :, i]), dim=-1).float()

        output, _ = self.rnn(input_rnn, None)
        policy = self.verify_layers(output).view(self.n_targets, batch_size, n_steps, -1)
        output = output.view(self.n_targets, batch_size, n_steps, -1)

        for i in range(len(self.targets)):
            pulse_policy[..., 2 * (i + 1):2 * (i + 2)] = policy[i, :, :, :2]
            input_agents[..., self.hidden_size * i:self.hidden_size * (i + 1)] = output[i]

        reading_policy = torch.tensor([]).to(in_data_common.device)
        if policy.shape[-1] > 2:
            reading_policy = torch.empty((*batch_size, 2 * self.n_targets)).to(in_data_common.device)
            for i in range(len(self.targets)):
                reading_policy[..., 2 * (i + 1):2 * (i + 2)] = policy[i, :, :, 2:]

        pulse_policy_tmp = self.pulse_layers(input_agents)
        pulse_policy[..., :2] = pulse_policy_tmp[..., :-1]
        skip_policy = pulse_policy_tmp[..., -1]

        critic_out = None
        if critic:
            critic_out = self.critic(input_agents)

        return torch.cat((pulse_policy, reading_policy, skip_policy.view(*skip_policy.shape, -1)), dim=-1), critic_out
