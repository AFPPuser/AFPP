import gin
import numpy as np


class EnvHandler:

    def __init__(self):
        self.sub_levels_indicator_vec = None
        self.block_idx = -1
        self.wl_idx = -1
        self.p_above_targets = None
        self.last_read = None

    @property
    def state_size(self):
        raise NotImplementedError

    def act(self, chip, pulse, v_verify_arr, read_levels=None):
        raise NotImplementedError

    def reset(self, chip, block_idx, wl_idx, sub_levels_indicator_vec, read_levels=None):
        chip.erase_block(block_idx=block_idx)
        self.last_read = np.zeros(chip.n_cells_per_wl)
        self.sub_levels_indicator_vec = sub_levels_indicator_vec
        self.current_sub_levels = -1 * np.ones(chip.n_cells_per_wl)
        self.block_idx = block_idx
        self.wl_idx = wl_idx
        self.read_levels = read_levels


@gin.configurable()
class CellsAboveTargetState(EnvHandler):
    def __init__(self):
        super().__init__()
        self.last_read = None
        self.current_sub_levels = None
        self.read_levels = None

    @property
    def state_size(self):
        return 0, 1

    def act(self, chip, pulse, v_verify_arr, read_levels=None):
        self.current_sub_levels = self.sub_levels_indicator_vec.copy()
        self.current_sub_levels[v_verify_arr[self.sub_levels_indicator_vec] <= self.last_read] = -1
        self.read_levels = read_levels
        read, new_sub_levels = chip.pulse_and_read(block_idx=self.block_idx,
                                                   wl_idx=self.wl_idx,
                                                   v_verify_per_sub_level_mv=v_verify_arr,
                                                   vec_v_p_mv=np.array([pulse]),
                                                   sub_levels_indicator_vec=self.current_sub_levels,
                                                   verify_levels=np.sort(read_levels),
                                                   return_indicator_flag=True)

        self.last_read = np.max([self.last_read,
                                 -4000 * (read < read_levels[self.sub_levels_indicator_vec]) +
                                 read_levels[self.sub_levels_indicator_vec] * (
                                         read >= read_levels[self.sub_levels_indicator_vec]),

                                 -4000 * (self.current_sub_levels <= new_sub_levels) +
                                 v_verify_arr[self.sub_levels_indicator_vec] * (
                                         self.current_sub_levels > new_sub_levels)],
                                axis=0)

    def get_obs(self, p_ind):
        p_above_targets = np.array([[np.mean(self.last_read[self.sub_levels_indicator_vec == i] >= self.read_levels[i])]
                                    for i in range(self.read_levels.size)])

        common_state = np.array([])

        return common_state, p_above_targets