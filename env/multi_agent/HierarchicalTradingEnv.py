from ray.rllib.env import MultiAgentEnv
from env.env_ST_advanced_raylib import StockTradingEnvAdvanced
from env.multi_agent.manager import Manager
from env.multi_agent.worker import Worker
import gymnasium as gym
from gymnasium import spaces
import numpy as np


class HierarchicalTradingEnv(MultiAgentEnv):

    def __init__(self, env_config):

        self.manager_env = Manager(env_config=env_config["manager_config"])
        self.worker_env = Worker(env_config=env_config["worker_config"])

        self.action_space = spaces.Box(
            low=-1, high=1, shape=(5,), dtype=np.float32)
        self.observation_space = spaces.Dict({
            "worker": self.worker_env.observation_space,

        })
        # self.observation_space = spaces.Dict({
        #     "manager": self.manager_env.observation_space,
        #     "worker": self.worker_env.observation_space
        # })

        self.state = self._initiate_state
        self.manager_env.set_value(66)

    def reset(self, *, seed=None, options=None, **kwargs):

        manager_obs, _ = self.manager_env.reset()
        worker_obs, _ = self.worker_env.reset()
        print("HierarchicalTradingEnv reset:", {
              "manager": manager_obs, "worker": worker_obs})

        return {"worker": worker_obs}, {}
        # return {"manager": manager_obs, "worker": worker_obs}, {}

    def step(self, action):
        manager_obs, _, manager_done, _ = self.manager_env.step(
            None)  # Pass None for action
        worker_obs, _, worker_done, _ = self.worker_env.step(
            None)  # Pass None for action

        reward = {
            "manager": 1.0,  # Or whatever constant you'd like
            "worker": 1.0   # Or whatever constant you'd like
        }

        done = {
            "manager": manager_done,
            "worker": worker_done,
            "__all__": manager_done and worker_done
        }
        print("HierarchicalTradingEnv step:", {
              "worker": worker_obs})

        return {"worker": worker_obs}, reward, done, {}
        # return {"manager": manager_obs, "worker": worker_obs}, reward, done, {}

    def _initiate_state(self, info):
        manager_obs, _ = self.manager_env.reset()
        worker_obs, _ = self.worker_env.reset()
        state = {"worker": worker_obs}
        print("HierarchicalTradingEnv initiate:", {
              "worker": worker_obs})
        # state = {"manager": manager_obs, "worker": worker_obs}
        # print("HierarchicalTradingEnv initiate:", {
        #       "manager": manager_obs, "worker": worker_obs})

        return state
