from ray.rllib.env import MultiAgentEnv
from env.env_ST_advanced_raylib import StockTradingEnvAdvanced
from env.multi_agent.manager import Manager
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from ray.rllib.evaluation import episode_v2

#! TODO
#? 1) gather accumulated rewards from worker --> solely on trading revenue


class HRL(MultiAgentEnv):
 
    def __init__(self, env_config, print_verbosity=1, initial_capital=2e6):
        super(HRL, self).__init__()
        self.initial_capital = initial_capital
        self.manager = Manager(env_config=env_config["manager_config"], initial_capital=initial_capital)
        self._agent_ids = set(["manager"] + list(self.manager.workers.keys()))
        self.workers = self.manager.workers
        self.print_verbosity = print_verbosity
        self.episode = self.day = 0
    
        # Rewards


        self.observation_space = gym.spaces.Dict({
            **{"manager": self.manager.observation_space},
            **{tic: worker.observation_space for tic, worker in self.workers.items()}
        })



        self.action_space = gym.spaces.Dict({
            **{"manager": self.manager.action_space},
            **{tic: worker.action_space for tic, worker in self.workers.items()}
        })

    def reset(self, *, seed=None, options=None):
        manager_obs, manager_info = self.manager.reset()  # This will now only contain manager-specific states
        self.episode += 1
        self._reset_to_initial_values()

        obs = {"manager": manager_obs}
        info = {"manager": manager_info}
        
        # Loop over all workers and gather their observations
        for worker_id, worker in self.workers.items():
            worker_obs, worker_info = worker.reset()
            obs[worker_id] = worker_obs  # This should be a dictionary, as per your worker reset method
            info[worker_id] = worker_info
            
        return obs, info
    
    def _reset_to_initial_values(self):
        self.day = 0
        self.accumulated_worker_rewards_dict = {}
        self.manager_rewards_array = []


    def _calculate_reward(self,worker_rewards, manager_reward):
        
        return 1

    def step(self, action_dict):
        fully_done = False
        # print("action dict in hrl step: ",action_dict)
        # manager_action = action_dict["manager"]
        
        obs = {"manager": None}  # We will update this later
        reward = {"manager": None}  # We will update this later
        terminateds = {"__all__": False}
        truncateds = {"__all__": False}
        info = {"manager": None}  # We will update this later

        worker_observations = []
        worker_rewards = []
        worker_dones = []
        worker_truncateds = []
    
        # Loop over all workers, apply actions, and gather results
        for worker_id, worker in self.workers.items():
            worker_action = action_dict[worker_id]
            worker_obs, worker_reward, worker_done, worker_truncated, worker_info = worker.step(worker_action)
            
            worker_observations.append(worker_obs)
            worker_rewards.append(worker_reward)
            worker_dones.append(worker_done)
            worker_truncateds.append(worker_truncated)
            
            obs[worker_id] = worker_obs
            reward[worker_id] = worker_reward
            terminateds[worker_id] = worker_done
            truncateds[worker_id] = worker_truncated
            info[worker_id] = worker_info

        # Now call the manager's step method with the collected worker data
        #! for tune run da wir keine managher action haben
        manager_obs, manager_reward, manager_done, manager_truncated, manager_info = self.manager.step(None, worker_observations, worker_rewards, worker_dones, worker_truncateds)
        #! das hatten wir vorher
        # manager_obs, manager_reward, manager_done, manager_truncated, manager_info = self.manager.step(manager_action, worker_observations, worker_rewards, worker_dones, worker_truncateds)
        # Get the manager's decisions



        
     
        terminateds["__all__"] = terminateds["__all__"] or worker_done

        fully_done = terminateds["__all__"]

        if fully_done:
            self._handle_done()
        obs["manager"] = manager_obs
        reward["manager"] = manager_reward
        terminateds["manager"] = manager_done
        truncateds["manager"] = manager_truncated
        info["manager"] = manager_info


        self.day +=1
  
   
        return obs, reward, terminateds, truncateds, info
        
    
    
    def _handle_done(self):
        if self.episode % self.print_verbosity == 0:
            print("=========HRL is done=============")
            print("HRL is done")
            print(f"day: {self.day}, episode: {self.episode}")
            print(f"Total Cash Transfers: {self.manager.total_cash_transfers}")           
            print(f"total_portfolio_trades: {self.manager._calculate_total_portfolio_trades()[0]}")           
            print(f"Beginn_Portfolio_Value: {round(self.initial_capital)}")           
            print(f"End_Portfolio_Value: {self.manager._get_state().get('total_portfolio_value')[0]}")
            print(f"Annual Return: {self.manager._calculate_annualized_return()*100:0.2f} %")
            for worker_id, worker in self.workers.items():
                print(f"Worker ID: {worker_id} Current Stock Exposure: {round(worker._get_state().get('current_stock_exposure')[0])}")
            print(f"Free Cash: {self.manager._get_state().get('total_free_cash')[0]}")
            print(f"Total Costs: {self.manager._get_state().get('total_free_cash')[0]}")
            print("=================================")


            # print(f"day: {self.day}, episode: {self.episode}")
            # print(f"total_pnl: {self.total_pnl_history[-1]:0.2f}")
            # print(f"self.cash_spent: {self.cash_spent:0.2f}")
            # print(f"self.cash_from_sales: {self.cash_from_sales:0.2f}")
            # print(f"total_cost: {self.total_costs:0.2f}")
            # print(f"total_trades: {self.total_trades}")
            # print(f"Begin_portfolio_value: {self.cash_initial }")
            # print(f"End stock holdings: {self.stock_holding_memory[-1]}")
            # print(f"Last stock price: {self.current_price}")
            # print(f"End_portfolio_value: {self._calculate_assets()}")
            # print(f"Invalid Actions {self.invalid_action_count}")
            # print("=================================")