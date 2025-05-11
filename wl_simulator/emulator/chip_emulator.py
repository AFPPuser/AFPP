import sys
import pickle
import numpy as np
from scipy import interpolate
from .model import VAE
from .WL_simulator import WL_simulator
import torch


class Chip:
    """
    A chip simulator. Everything is normalized, including WL index, blocks, pulse magnitudes and voltage values.
    """

    def __init__(self, n_cells_per_wl=None):
        self.n_cells_per_wl = n_cells_per_wl
        self.chip_id = -1
        self.board_uid = -1

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.model = VAE.load_model(device=self.device)
        self.wl = WL_simulator(self.model,
                               n_cells_per_wl=self.n_cells_per_wl,
                               n_latent_params=1,
                               device=self.device,
                               mu=0, sigma=0.8)

    def erase_block(self, block_idx):
        self.wl = WL_simulator(self.model, n_cells_per_wl=self.n_cells_per_wl, n_latent_params=1,
                               device=self.device, sigma=0.8, mu=0)

    def read_voltages_verify_levels(self, block_idx, wl_idx, verify_levels):
        # find to which bin each cell belongs
        cells_bin_indexes = np.searchsorted(verify_levels, self.wl.read().cpu().detach().numpy(), 'right')
        bin_representative_values = np.concatenate(
            (verify_levels[0] - 0.001, verify_levels[:-1] / 2. + verify_levels[1:] / 2.,
             verify_levels[-1] + 0.001),
            axis=None)
        cells_voltage_mv = bin_representative_values[cells_bin_indexes]
        return cells_voltage_mv

    def read_voltages(self, block_idx=0, wl_idx=0):
        return self.wl.read().detach().cpu().numpy()

    def verify(self, v_verify, block_idx, wl_idx):
        return (self.wl.read() >= v_verify).cpu().numpy()

    def pulse(self,
              v_verify_per_sub_level_mv,
              vec_v_p_mv,
              sub_levels_indicator_vec,
              wl_idx,
              block_idx=0):
        vec_v_p_mv = np.array([vec_v_p_mv]).reshape(-1)
        v_verify_per_sub_level_mv = np.array([v_verify_per_sub_level_mv]).reshape(-1)
        inhibit_vec = np.zeros_like(sub_levels_indicator_vec, dtype='bool')
        inhibit_vec[sub_levels_indicator_vec == -1] = True
        for p in vec_v_p_mv:
            for i, v in enumerate(v_verify_per_sub_level_mv):
                verify_output = self.verify(v_verify=v, block_idx=block_idx, wl_idx=wl_idx)
                sub_level_cells_verified_cells = (sub_levels_indicator_vec == i) & (verify_output == 1)
                inhibit_vec[sub_level_cells_verified_cells] = True

            self.wl.program(v_pgm=p, wl_idx=wl_idx, inhibit_vector=inhibit_vec)
            self.sub_levels_indicator_new = sub_levels_indicator_vec.copy()
            self.sub_levels_indicator_new[inhibit_vec] = -1
        return self.sub_levels_indicator_new

    def pulse_and_read(self,
                       v_verify_per_sub_level_mv,
                       vec_v_p_mv,
                       sub_levels_indicator_vec,
                       block_idx=0,
                       wl_idx=0,
                       verify_levels=None,
                       return_indicator_flag=True):

        sub_levels_indicator_new = self.pulse(v_verify_per_sub_level_mv=v_verify_per_sub_level_mv,
                                              vec_v_p_mv=vec_v_p_mv,
                                              sub_levels_indicator_vec=sub_levels_indicator_vec,
                                              block_idx=block_idx,
                                              wl_idx=wl_idx)

        read = self.read_voltages_verify_levels(block_idx=block_idx,
                                                wl_idx=wl_idx,
                                                verify_levels=verify_levels)
        if return_indicator_flag:
            return read, sub_levels_indicator_new

        return read
