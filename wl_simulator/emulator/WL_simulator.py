import torch
import torch.utils.data
import gin

#from emulator.model import VAE
from wl_simulator.emulator import utils
import numpy as np


@gin.configurable
class WL_simulator:

    def __init__(self,
                 model,
                 n_cells_per_wl,
                 n_latent_params,
                 device,
                 mu,
                 sigma
                 ):

        self.model = model
        self.threshold_voltages = torch.zeros(n_cells_per_wl).to(device)
        self.n_cells_per_wl = n_cells_per_wl
        self.z_aux = utils.draw_normal(mu * torch.ones((n_cells_per_wl, n_latent_params)).to(device),
                                       sigma * torch.ones((n_cells_per_wl, n_latent_params)).to(device))
        self.device = device
        self.model.eval()

    def program(self, v_pgm: torch.Tensor, wl_idx: int, inhibit_vector: np.array):
        """
        Normalized programming function
        """
        n_cells_to_program = inhibit_vector.size - np.sum(inhibit_vector)
        params = torch.empty(size=(n_cells_to_program, 3)).to(self.device)
        params[:, 0] = (wl_idx * 0.5) + 0.2
        params[:, 1] = v_pgm
        params[:, 2] = self.threshold_voltages[~inhibit_vector]

        with torch.no_grad():
            mu_delta_v, sigma_delta_v = self.model.decode(self.z_aux[~inhibit_vector], params)

        delta_v = utils.draw_normal(mu_delta_v, sigma_delta_v)
        delta_v[mu_delta_v < 0] = 0
        delta_v = delta_v.flatten()

        self.threshold_voltages[~inhibit_vector] += delta_v

    def read(self):
        return self.threshold_voltages


