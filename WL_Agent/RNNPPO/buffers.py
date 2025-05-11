import numpy as np
import dataclasses
import json


@dataclasses.dataclass
class EpisodeData:
    observations_common: np.ndarray
    observations_per_level: np.ndarray
    actions: np.ndarray
    log_probs: np.ndarray
    returns: np.ndarray
    advantages: np.ndarray
    multiple_targets: np.ndarray

    def save(self, filename):
        d = dataclasses.asdict(self)
        for k in d:
            if type(d[k]) == np.ndarray:
                d[k] = d[k].tolist()
        episode_string = json.dumps(d)
        with open(filename, 'a') as f:
            f.write(episode_string + '\n')


class PPOMemory:
    def __init__(self, max_episodes=32):
        self.max_episodes = max_episodes
        self.observations_common = None
        self.observations_per_level = None
        self.actions = None
        self.log_probs = None
        self.returns = None
        self.advantages = None
        self.multiple_targets = None

    def store_memory(self, episode_data: EpisodeData):

        returns = episode_data.returns.reshape(1, *episode_data.returns.shape)

        if self.returns is None:
            self.returns = returns
        else:
            self.returns =  np.append(self.returns, returns, axis=0)

        observations = episode_data.observations_common.reshape(1, *episode_data.observations_common.shape)
        if self.observations_common is None:
            self.observations_common = observations
        else:
            self.observations_common =  np.append(self.observations_common, observations, axis=0)

        observations = episode_data.observations_per_level.reshape(1, *episode_data.observations_per_level.shape)
        if self.observations_per_level is None:
            self.observations_per_level = observations
        else:
            self.observations_per_level =  np.append(self.observations_per_level, observations, axis=0)

        log_probs = episode_data.log_probs.reshape(1, *episode_data.log_probs.shape)
        if self.log_probs is None:
            self.log_probs = log_probs
        else:
            self.log_probs =  np.append(self.log_probs, log_probs, axis=0)

        actions = episode_data.actions.reshape(1, *episode_data.actions.shape)
        if self.actions is None:
            self.actions = actions
        else:
            self.actions =  np.append(self.actions, actions, axis=0)


        advantages = episode_data.advantages.reshape(1, *episode_data.advantages.shape)
        if self.advantages is None:
            self.advantages = advantages
        else:
            self.advantages =  np.append(self.advantages, advantages, axis=0)

        multiple_targets = episode_data.multiple_targets.reshape(1, *episode_data.multiple_targets.shape)
        if self.multiple_targets is None:
            self.multiple_targets = multiple_targets
        else:
            self.multiple_targets =  np.append(self.multiple_targets, multiple_targets, axis=0)


        if self.n_episodes > self.max_episodes:
            self.observations_common = self.observations_common[-self.max_episodes:]
            self.observations_per_level = self.observations_per_level[-self.max_episodes:]
            self.actions = self.actions[-self.max_episodes:]
            self.log_probs = self.log_probs[-self.max_episodes:]
            self.returns = self.returns[-self.max_episodes:]
            self.advantages = self.advantages[-self.max_episodes:]
            self.multiple_targets = self.multiple_targets[-self.max_episodes:]

    @property
    def n_episodes(self):
        return self.observations_common.shape[0]
