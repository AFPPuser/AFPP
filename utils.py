import os
import gin
import torch
import inspect
import argparse
from wl_simulator.emulator.chip_emulator import Chip


@gin.configurable
def init_chip(n_cells_per_wl=9_000):
    """
    This function init a chip object or a simulator
    In this version only the simulator is supported.
    """
    chips = [Chip(n_cells_per_wl=n_cells_per_wl)]
    return chips


def register_torch_modules_with_gin():
    nn_modules_list = inspect.getmembers(torch.nn)
    for nn_module in nn_modules_list:
        if inspect.isclass(nn_module[1]):
            gin.external_configurable(nn_module[1])


@gin.configurable
def load_config(run_dir, default_config=None):
    # Register all torch.nn classes with gin-config:
    register_torch_modules_with_gin()

    parser = argparse.ArgumentParser(description='ARLPS arguments parser.')
    parser.add_argument('--name', type=str, required=True, help='Run name')
    sys_args = parser.parse_args()
    results_dir_path = os.path.join(run_dir, 'output')

    # Set the results folder
    experiment_dir = os.path.join(results_dir_path, sys_args.name)

    # Load the config file
    os.makedirs(experiment_dir, exist_ok=True)
    config_path = os.path.join(run_dir, default_config)
    gin.parse_config_file(config_path)

    return experiment_dir
