import gin
import numpy as np


class EnvHandler:

    def __init__(self, targets, n_pulses):
        self.targets = np.array(targets)
        self.n_targets = self.targets.size
        self.n_pulses = n_pulses
        self.sub_levels_indicator_vec = None
        self.block_idx = -1
        self.wl_idx = -1
        self.p_above_targets = None

    @property
    def state_size(self):
        raise NotImplementedError

    def act(self, chip, pulse, v_verify_arr, read_levels=None):
        raise NotImplementedError

    def reset(self, chip, block_idx, wl_idx, sub_levels_indicator_vec):
        chip.erase_block(block_idx=block_idx)
        self.sub_levels_indicator_vec = sub_levels_indicator_vec
        self.block_idx = block_idx
        self.wl_idx = wl_idx


@gin.configurable()
class CellsAboveTargetState(EnvHandler):
    def __init__(self, targets, n_pulses):
        super().__init__(targets=targets, n_pulses=n_pulses)
        self.last_read = None
        self.read_levels = None
        self.current_sub_levels = None

    @property
    def state_size(self):
        return 1, 1

    def act(self, chip, pulse, v_verify_arr, read_levels=None):
        self.current_sub_levels = self.sub_levels_indicator_vec.copy()
        self.current_sub_levels[v_verify_arr[self.sub_levels_indicator_vec] <= self.last_read] = -1

        read, new_sub_levels = chip.pulse_and_read(block_idx=self.block_idx,
                                                   wl_idx=self.wl_idx,
                                                   v_verify_per_sub_level_mv=v_verify_arr,
                                                   vec_v_p_mv=np.array([pulse]),
                                                   sub_levels_indicator_vec=self.current_sub_levels,
                                                   verify_levels=read_levels,
                                                   return_indicator_flag=True)
        self.read_levels = read_levels
        self.last_read = np.max([self.last_read,
                                 read_levels[self.sub_levels_indicator_vec] * (
                                         read >= read_levels[self.sub_levels_indicator_vec]),
                                 v_verify_arr[self.sub_levels_indicator_vec] * (
                                         self.current_sub_levels > new_sub_levels)],
                                axis=0)

    def get_obs(self, p_ind):
        p_above_targets = np.array([[np.mean(self.last_read[self.sub_levels_indicator_vec == i] >= self.read_levels[i])]
                                    for i in range(self.n_targets)])

        common_state = np.array([p_ind / self.n_pulses])

        return common_state, p_above_targets

    def reset(self, chip, block_idx, wl_idx, sub_levels_indicator_vec):
        super().reset(chip=chip, block_idx=block_idx, wl_idx=wl_idx, sub_levels_indicator_vec=sub_levels_indicator_vec)
        self.last_read = np.zeros(chip.n_cells_per_wl)
        self.current_sub_levels = -1 * np.ones(chip.n_cells_per_wl)
        self.read_levels = self.targets


@gin.configurable()
class AboveTargetLearningReadPoint(EnvHandler):
    def __init__(self, targets, n_pulses):
        super().__init__(targets=targets, n_pulses=n_pulses)
        self.last_read = None
        self.read_targets = None
        self.current_sub_levels = None

    @property
    def state_size(self):
        return 0, 2

    def act(self, chip, pulse, v_verify_arr, read_levels=None):
        self.current_sub_levels = self.sub_levels_indicator_vec.copy()
        self.current_sub_levels[v_verify_arr[self.sub_levels_indicator_vec] <= self.last_read] = -1

        read, new_sub_levels = chip.pulse_and_read(block_idx=self.block_idx,
                                                   wl_idx=self.wl_idx,
                                                   v_verify_per_sub_level_mv=v_verify_arr,
                                                   vec_v_p_mv=np.array([pulse]),
                                                   sub_levels_indicator_vec=self.current_sub_levels,
                                                   verify_levels=np.sort(read_levels),
                                                   return_indicator_flag=True)

        self.last_read = np.max([self.last_read,
                                 read_levels[self.sub_levels_indicator_vec] * (
                                         read >= read_levels[self.sub_levels_indicator_vec]),
                                 v_verify_arr[self.sub_levels_indicator_vec] * (
                                         self.current_sub_levels > new_sub_levels)],
                                axis=0)
        self.read_targets = read_levels

    def get_obs(self, p_ind):
        p_above_targets = np.array(
            [[np.mean(self.last_read[self.sub_levels_indicator_vec == i] >= self.read_targets[i]),
              self.read_targets[i]] for i in range(self.n_targets)])

        common_state = np.array([])

        return common_state, p_above_targets

    def reset(self, chip, block_idx, wl_idx, sub_levels_indicator_vec):
        super().reset(chip=chip, block_idx=block_idx, wl_idx=wl_idx, sub_levels_indicator_vec=sub_levels_indicator_vec)
        self.last_read = np.zeros(chip.n_cells_per_wl)
        self.current_sub_levels = -1 * np.ones(chip.n_cells_per_wl)
        self.read_targets = self.targets


@gin.configurable()
class NoPulseNumber(EnvHandler):
    def __init__(self, targets, n_pulses):
        super().__init__(targets=targets, n_pulses=n_pulses)
        self.last_read = None
        self.current_sub_levels = None

    @property
    def state_size(self):
        return 0, 1

    def act(self, chip, pulse, v_verify_arr, read_levels=None):
        self.current_sub_levels = self.sub_levels_indicator_vec.copy()
        self.current_sub_levels[v_verify_arr[self.sub_levels_indicator_vec] <= self.last_read] = -1

        read, new_sub_levels = chip.ispp_mv_read(block_idx=self.block_idx,
                                                 wl_idx=self.wl_idx,
                                                 v_verify_per_sub_level_mv=v_verify_arr,
                                                 vec_v_p_mv=np.array([pulse]),
                                                 sub_levels_indicator_vec=self.current_sub_levels,
                                                 verify_levels=read_levels,
                                                 return_indicator_flag=True)

        self.last_read = np.max([self.last_read,
                                 read_levels[self.sub_levels_indicator_vec] * (
                                             read >= read_levels[self.sub_levels_indicator_vec]),
                                 v_verify_arr[self.sub_levels_indicator_vec] * (
                                         self.current_sub_levels > new_sub_levels)], axis=0)

    def get_obs(self, p_ind):
        p_above_targets = np.array([[np.mean(self.last_read[self.sub_levels_indicator_vec == i] >= self.targets[i])]
                                    for i in range(self.n_targets)])

        common_state = np.array([])

        return common_state, p_above_targets

    def reset(self, chip, block_idx, wl_idx, sub_levels_indicator_vec):
        super().reset(chip=chip, block_idx=block_idx, wl_idx=wl_idx, sub_levels_indicator_vec=sub_levels_indicator_vec)
        self.last_read = np.zeros(chip.n_cells_per_wl)
        self.current_sub_levels = -1 * np.ones(chip.n_cells_per_wl)
