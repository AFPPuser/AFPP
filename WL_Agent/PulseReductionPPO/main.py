import traceback
import os

import torch
import sys

from os.path import dirname, realpath

add_path = dirname(dirname(dirname(realpath(__file__))))
if add_path not in sys.path:
    sys.path.insert(0, add_path)

import utils
from model import AFPPAgent
from TrainManager import PPOTrainManager
from WL_Agent.env_handler import *

# ---------------------------------------- SET GLOBALS --------------------------------------------
CONFIG_FILE_NAME = 'config.gin.yaml'

# Set the torch device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# ******************************************************************************************************


@gin.configurable
def main(experiment_dir=None,
         ):

    try:
        # ----------- TRAIN MODEL -------------
        # Initialize the model
        rlpp_agent = AFPPAgent()

        # Initialize the train manager
        chips = utils.init_chip()
        train_manager = PPOTrainManager(experiment_dir=experiment_dir,
                                        agent=rlpp_agent,
                                        chips=chips,
                                        device=device)
        # Train the model
        train_manager.train_agent()

    except Exception as e:
        tb = traceback.format_exc()
        print('error:  {:s}'.format(str(e)))
        print('error traceback:  {:s}'.format(str(tb)))


if __name__ == '__main__':

    # Load the config file
    dirname = os.path.dirname(__file__)
    output_dir = utils.load_config(run_dir=dirname, default_config=CONFIG_FILE_NAME)

    # Run the experiment
    main(experiment_dir=output_dir)
