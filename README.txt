############
Introduction
############

A python implementation for the paper "Reinforcement Learning for Adaptive Programming Strategies in Flash
Memory Systems".


The library supports the 2 main experiments described in the paper:

1. Accuracy optimization:
    The library contains the three types of agents described in the paper:
        a. The "Pulse-only" agent.
        b. The "Pulse & Verify agent".
        c. The "full" agent.

2. Latency optimization:
    The library contains an agent dynamically learning to skip pulses.


A NAND-Flash chip simulator is added for training.
We emphasize that all chip parameter are normalized for security reasons.

############
Instructions
############

---------------------
Accuracy optimization:
---------------------
To train the agent execute WL_Agent/RNNPPO/main.py --name <run_name>
The config file for the run is located at:  WL_Agent/RNNPPO/config.gin.yaml

---------------------
Latency optimization:
---------------------
To train the agent execute WL_Agent/PulseReductionPPO/main.py --name <run_name>
The config file for the run is located at:  WL_Agent/PulseReductionPPO/config.gin.yaml


############
Requirements
############

Python 3.x
numpy>=1.8
torch>=1.8
gin-config>=0.4.0


###########################
Config files and parameters
###########################

-----------------------------------------------------------------------------------------------------------------------
| Parameter                                   | Description                                                           |
-----------------------------------------------------------------------------------------------------------------------
| init_chip.n_cells_per_wl                    | The number of cells in WL for the simulator                           |
|                                                                                                                     |
|                                                                                                                     |
| AFPPAgent.n_pulses                          | The number of pulses per episode                                      |
| AFPPAgent.targets                           | The Programming targets. Floats in range (0, 1)                       |
| AFPPAgent.model_cls                         | The NN type.                                                          |
|                                                To train the "pulse only" agent choose @PulseOnlyNN                  |
|                                                To train the "pulse & verify" and "full" agents choose @AFPPNN       |
| AFPPAgent.env_handler                       | Specifies the observation and action types.                           |
|                                                To train the "pulse only" and "pulse & verify"                       |
|                                                agents choose @CellsAboveTargetState                                 |
|                                                To train the "full" agent choose @AboveTargetLearningReadPoint       |
| AFPPAgent.learn_verify_level                | indicates whether to learn the verify levels                          |
|                                                (for the "verify & pulse" and "full" version of the agent).          |
| AFPPAgent.learn_reading_point               | indicates whether to learn the read levels                            |
|                                                (for the "full" version of the agent).                               |
|                                                                                                                     |
|                                                                                                                     |
| AFPPNN.pulse_layers_size                    | The hidden layer sizes of the actor pulse NN                          |
| AFPPNN.verify_layers_size                   | The hidden layer sizes of the actor verify & read NN                  |
| AFPPNN.critic_layer_size                    | The hidden layer sizes of critic                                      |
| AFPPNN.non_linearity_module                 | The actor & critic activation function                                |
| AFPPNN.hidden_size                          | The RNN hidden size.                                                  |
| AFPPNN.num_layers                           | The number of recurrent layers.                                       |
| AFPPNN.rnn_type                             | The RNN type of RNN (LSTM / Vanilla RNN)                              |
|                                                                                                                     |
|                                                                                                                     |
| PPOTrainManager.train_sub_steps             | The maximal level to program at each training step                    |
| PPOTrainManager.train_n_pulses              | The number of programming pulses for each training step               |
| PPOTrainManager.next_step_conditions        | The L2 condition required to move to the next training step           |
| PPOTrainManager.wl_idx                      | The NORMALIZED WL index. Float in range (0, 1)                        |
| PPOTrainManager.num_episodes                | The number of train episodes                                          |
| PPOTrainManager.batch_size                  | The batch size                                                        |
| PPOTrainManager.max_episodes                | The replay buffer size (in episodes)                                  |
| PPOTrainManager.lr                          | The actor and critic learning rate                                    |
| PPOTrainManager.ckpt_freq                   | The checkpoint saving frequency                                       |
| PPOTrainManager.init_agent_from_ckpt        | Path to a saved model to start training from                          |
| PPOTrainManager.reward_symmetry_factor      | The reward inequality coefficient                                     |
| PPOTrainManager.entropy_alpha               | The entropy term coefficient                                          |
| PPOTrainManager.policy_clip                 | The PPO policy clip                                                   |
| PPOTrainManager.n_batches_per_epoch         | Number of train epochs after each episode                             |
| PPOTrainManager.decay_lr                    | The lr term decay coefficient                                         |
| PPOTrainManager.decay_asymmetry             | The reward inequality term decay coefficient                          |
| PPOTrainManager.decay_entropy               | The entropy term decay coefficient                                    |
-----------------------------------------------------------------------------------------------------------------------


###########################
Licence
###########################
CC BY-NC-SA 4.0
