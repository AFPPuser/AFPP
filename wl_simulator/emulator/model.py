import torch
from torch import nn
import utils
from .mlp import MLP
import gin
import os


@gin.configurable
class Decoders(nn.Module):
    def __init__(self, config: dict):
        super(Decoders, self).__init__()

        num_units_feat_sp = 1
        dim_z_aux = config['dim_z_aux']
        dim_y_sp = config['dim_y_sp']
        x_lnvar = config['x_lnvar']
        hidlayers_aux_dec = config['hidlayers_aux_dec']

        self.register_buffer('param_x_lnvar', torch.ones(1) * x_lnvar)

        self.func_aux_expand = MLP(
            [dim_y_sp + dim_z_aux, ] + hidlayers_aux_dec + [2 * num_units_feat_sp, ], 'leakyrelu')

        self.out = MLP([num_units_feat_sp, ] + [num_units_feat_sp, ], 'softplus', actfun_output=True)


@gin.configurable
class Encoders(nn.Module):
    def __init__(self, config: dict):
        super(Encoders, self).__init__()

        dim_z_aux = config['dim_z_aux']

        num_units_feat_sp = 1
        dim_y_sp = config['dim_y_sp']
        measure_per_cell = config['measure_per_cell']
        hidlayers_aux_enc = config['hidlayers_aux_enc']

        self.func_z_aux = MLP([measure_per_cell * (num_units_feat_sp + dim_y_sp), ] + hidlayers_aux_enc + [
            2 * dim_z_aux, ], 'leakyrelu')

        self.out_z_aux = MLP([dim_z_aux, ] + [dim_z_aux, ], 'softplus', actfun_output=True)


class VAE(nn.Module):
    def __init__(self, normalization_constants, config: dict):
        super(VAE, self).__init__()

        self.dim_z_aux = config['dim_z_aux']

        self.dim_y_sp = config['dim_y_sp']

        self.activation = config['activation']

        self.num_units_feat_sp = 1

        self.x_lnvar = config['x_lnvar']
        self.measure_per_cell = config['measure_per_cell']
        self.hidlayers_aux_dec = config['hidlayers_aux_dec']
        self.hidlayers_aux_enc = config['hidlayers_aux_enc']

        # Decoding part
        self.dec = Decoders(config)

        # Encoding part
        self.enc = Encoders(config)

    def priors(self, n: int, device):

        prior_z_aux_stat = {'mean': torch.zeros(n, self.dim_z_aux, device=device),
                            'lnvar': torch.ones(n, self.dim_z_aux, device=device)}
        return prior_z_aux_stat

    def decode(self, z_aux: torch.Tensor, y_sp: torch.Tensor):

        inp = torch.cat((z_aux, y_sp), -1)
        x = self.dec.func_aux_expand(inp)

        return x[:, :x.shape[1] // 2], self.dec.out(x[:, x.shape[1] // 2:])

    def encode(self, x_sp: torch.Tensor, y_sp: torch.Tensor):

        inp = torch.cat((x_sp, y_sp), -1)
        # infer z_aux
        z_stat_tmp = self.enc.func_z_aux(inp)

        return torch.hstack((z_stat_tmp[:, :self.dim_z_aux],
                             self.enc.out_z_aux(z_stat_tmp[:, self.dim_z_aux:])))

    def draw(self, z_aux_stat, hard_z: bool = False):

        if not hard_z:
            z_aux = utils.draw_normal(z_aux_stat[:, :z_aux_stat.shape[1] // 2],
                                      z_aux_stat[:, z_aux_stat.shape[1] // 2:])
        else:
            z_aux = z_aux_stat[:, :z_phy_stat.shape[1] // 2].clone()

        return z_aux.to(z_aux_stat.device)

    def normalize(self, data):
        data = (data - self.normalization_constants['means']) / self.normalization_constants['stds']
        return data

    def denormalize(self, data):
        data = data * self.normalization_constants['stds'] + self.normalization_constants['means']
        return data

    @staticmethod
    def load_model(device):

        model_full_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'model_at_epoch_500.mdl')
        mdl_dict = torch.load(model_full_path, map_location=device)
        normalization_constants = {'means': 0, 'stds': 0}
        mdl_dict['config']['dim_y_sp'] = 3
        mdl_dict['config']['measure_per_cell'] = 16
        model = VAE(normalization_constants, mdl_dict['config']).to(device)

        model.load_state_dict(mdl_dict['model_state_dict'])

        model.eval()
        return model

    def forward(self, x_sp: torch.Tensor, y_sp: torch.Tensor, reconstruct: bool = True, hard_z: bool = False):
        z_aux_stat = self.encode(x_sp, y_sp)

        if not reconstruct:
            return z_aux_stat

        # draw & reconstruction
        z_aux = self.draw(z_aux_stat, hard_z=hard_z).repeat(1, self.measure_per_cell).view(-1, self.dim_z_aux)
        x, x_lnvar = self.decode(z_aux, y_sp.reshape(y_sp.shape[0] * self.measure_per_cell, -1))

        return z_aux_stat, x, x_lnvar
