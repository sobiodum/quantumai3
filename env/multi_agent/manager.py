import gymnasium as gym
from gymnasium import spaces
import numpy as np
from env.multi_agent.worker_components.manger_state_space import ManagerStateClass
from env.multi_agent.worker_hrl import Worker
import logging
logger = logging.getLogger(__name__)


class Spec(object):
    def __init__(self, id, max_episode_steps):
        self.id = id
        self.max_episode_steps = max_episode_steps


class Manager(gym.Env):
    def __init__(self, env_config, initial_capital):
        super(Manager, self).__init__()
        df = env_config["df"]
        unique_tickers = df['tic'].unique()
        self.total_capital = initial_capital
        self.workers = {}
        self.day = 0
        capital_per_worker = self.total_capital / len(unique_tickers)
        for ticker in unique_tickers:
            ticker_df = df[df['tic'] == ticker]
            worker_env_config = env_config.copy()
            worker_env_config["df"] = ticker_df  # Overwrite with individual stock data
            self.workers[ticker] = Worker(worker_env_config, initial_capital=capital_per_worker)
        self.step_count = 0
        self.decisions = {}
        self.total_cash_transfers = 0

         # Rewards
        self.worker_rewards_dict = {}
 
        self.manager_rewards_array = []
        # Trades
        self.worker_total_trades_dict = {}
        #Exposure
        self.worker_expsoure_dict = {}
        #PnL
        self.worker_accumulated_pnl_dict = {}
        # Sharpe Ratio
        self.worker_sharpe_ratio_dict = {}
        # Rewards
        self.return_worker_return_memory_dict = {}
        # Return per Volatility
        self.worker_return_per_volatility = {}
        # Free Cash
        self.worker_free_cash = {}



        self.action_space = spaces.Dict({
            ticker: self.worker_directive_space() for ticker in unique_tickers
        })
        
        self.observation_space = spaces.Dict({
            'capital_allocation': spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
            'risk_limit': spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
            'position_limit': spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
            'total_free_cash': spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
            'portfolio_sharpe_ratio': spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
            'total_portfolio_trades': spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
            'total_portfolio_value': spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),

        })

  

        # self.state = self._initiate_state()
    
    def step(self, action, worker_observations, worker_rewards, worker_dones, worker_truncateds):
        self.step_count += 1
        self.day += 0
        # Step 1: Create a dummy manager action dictionary
        # dummy_manager_action = {
        #     'capital_allocation': np.array([0.0], dtype=np.float32), 
        #     'risk_limit': np.array([0.0], dtype=np.float32), 
        #     'position_limit': np.array([0.0], dtype=np.float32), 
        #     'total_free_cash': np.array([0.0], dtype=np.float32), 
        #     'portfolio_sharpe_ratio': np.array([0.0], dtype=np.float32), 
        #     'total_portfolio_trades': np.array([0.0], dtype=np.float32), 
        #     'total_portfolio_value': np.array([0.0], dtype=np.float32)
        # }
        
        # # Step 2: Check if the 'manager' key is in the action dictionary, and if not, add it
        # if 'manager' not in action:
        #     action['manager'] = dummy_manager_action
        # self.state = self._update_state(worker_observations)
        self.worker_total_trades_dict = {}
        self.worker_rewards_dict = {}
        self.worker_expsoure_dict = {}
        self.worker_accumulated_pnl_dict = {}
        self.worker_sharpe_ratio_dict = {}
        self.return_worker_return_memory_dict = {}
        self.worker_return_per_volatility = {}
        for worker_id, worker in self.workers.items():
            worker_state = worker._get_state()
            worker_sharpe_ratios = worker._get_sharpe_ratio()
            worker_return_memory = worker._get_return_memory()
            worker_current_cash = worker.current_cash
            self.worker_total_trades_dict[worker_id] = worker_state["total_trades"]           
            self.worker_rewards_dict[worker_id] = worker_state["current_stock_exposure"]
            self.worker_expsoure_dict[worker_id] = worker_state["current_stock_exposure"]
            self.worker_accumulated_pnl_dict[worker_id] = worker_state["pnl"]
            self.worker_sharpe_ratio_dict[worker_id] = worker_sharpe_ratios
            self.return_worker_return_memory_dict[worker_id] = worker_return_memory
            self.worker_return_per_volatility[worker_id] = worker_state["return_per_volatility"]
            self.worker_free_cash[worker_id] = worker_current_cash

 

        # Generate high-level orders for each worker
        if self.step_count % 50 == 0:
            self.reallocate_capital()

        

        all_workers_done = all(worker_dones)
        all_workers_truncated = any(worker_truncateds)
    
        # for ticker, worker in self.workers.items():
        #     worker.set_directives(action[ticker])
        
        # Update the manager's state based on worker observations or other criteria
        self.day += 1
        reward = self._calculate_reward()


        obs = self._get_state()
        info = {}
        

        return obs, reward, all_workers_done, all_workers_truncated, info
    
    def make_decision(self, worker_id, observation):
        # ... (logic to make decisions based on the current observations)
        self.decisions[worker_id] = None  # store the decision
        pass

    def reset(self, *, seed=None, options=None):
        # self.state = self._initiate_state()
        self._reset_to_initial_values()
        obs= self._get_state()
        manager_obs = obs
        manager_info = {}
    
        return manager_obs, manager_info

    def _reset_to_initial_values(self):
        self.step_count = self.day = 0
        self.decisions = {}
        self.worker_total_trades_dict = {}
        self.worker_rewards_dict = {}
        self.worker_expsoure_dict = {}
        self.worker_accumulated_pnl_dict = {}
        self.worker_sharpe_ratio_dict = {}
        self.return_worker_return_memory_dict = {}
        self.worker_return_per_volatility = {}
        self.worker_free_cash = {}
        self.total_cash_transfers = 0

    
    def _get_state(self):
        state = {}
        state['capital_allocation'] = self.calculate_cash_allocation()
        state['risk_limit'] = self._calculate_risk_limit()
        state['position_limit'] = self._calculate_position_limit()
        state['total_free_cash'] = self._calculate_free_cash()
        state['portfolio_sharpe_ratio'] = self._calculate_portfolio_sharpe_ratio()
        state['total_portfolio_trades'] = self._calculate_total_portfolio_trades()
        state['total_portfolio_value'] = self._calculate_total_portfolio_value()
        return state
    
    def _calculate_total_portfolio_value(self):
        portfolio_value = 0
        for worker in self.workers.values():
            portfolio_value += worker._calculate_assets()
        return np.array([portfolio_value], dtype=np.float32)
    
    def _calculate_free_cash(self):
        if not self.worker_free_cash:
            return np.array([0.0], dtype=np.float32)
        free_cash = sum(self.worker_free_cash.values())
        return np.array([free_cash], dtype=np.float32)

    def _calculate_total_portfolio_trades(self):
        if not self.worker_total_trades_dict:
            return np.array([0.0], dtype=np.float32)
        total_trades = sum(self.worker_total_trades_dict.values())
        return np.array(total_trades, dtype=np.float32)

            
    
    def _calculate_sharpe_ratio(self):
        """ Calculate Sharpe Ratio for all workers combined """
        # Calculate Sharpe Ratio for all workers combined
        total_returns = sum(self.worker_rewards_dict.values())
        total_volatility = np.std(list(self.worker_rewards_dict.values()))      
        sharpe_ratio = total_returns / total_volatility if total_volatility != 0 else 0
        if np.isnan(sharpe_ratio):
            sharpe_ratio = 0
        return np.array([sharpe_ratio], dtype=np.float32)
    


    
    def calculate_cash_allocation(self):
        return np.array([0.0], dtype=np.float32)
    def _calculate_risk_limit(self):
        return np.array([0.0], dtype=np.float32)
    def _calculate_position_limit(self):
        return np.array([0.0], dtype=np.float32)

    def worker_directive_space(self):
        return spaces.Dict({
            "capital_allocation": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            "risk_limit": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            "position_limit": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        })



    def pre_approve_trade(self, worker_id, proposed_trade):
        # Here, proposed_trade could be a dictionary with details of the proposed trade
        # such as {'action': 'buy', 'amount': 1000, 'price': 50}
        
        # Apply risk management rules to decide whether to approve the trade
        # For simplicity, let's just use a dummy rule here. You can replace this with real rules
        if proposed_trade['amount'] * proposed_trade['price'] <= self.capital_allocation[worker_id]:
            return True  # Approve the trade
        else:
            return False  # Deny the trade
    def manage_transaction_costs(self, worker_id, proposed_trade):
        # Here, you can implement logic to estimate the transaction costs of the proposed trade
        # and advise the worker to delay the trade if the costs are too high
        # For simplicity, let's just use a dummy rule here. You can replace this with real rules
        estimated_cost = proposed_trade['amount'] * proposed_trade['price'] * 0.001  # Just a dummy estimation
        if estimated_cost <= 10:  # Dummy threshold
            return True  # Proceed with the trade
        else:
            return False  # Advise to delay the trade
        
    
    def _calculate_reward(self):
        #  reward = self._calculate_portfolio_sharpe_ratio()
        #  reward = reward[0]
         annual_return = self._calculate_annualized_return()
         return np.float32(annual_return)

    def _calculate_annualized_return(self):
        if self.day == 0:
            return 0.0
        end_value = self._get_state().get("total_portfolio_value")[0]
        start_value = self.total_capital
        return_value = end_value / start_value
        annualized_return = (return_value ** (365.0 / self.day)) - 1
        return annualized_return


    def reallocate_capital(self):
        """
        Handles the Cash allocation among workers bases on return by volatility
        """
        # Step 1: Identify the worker with the highest and lowest return per volatility
        best_worker_id = max(self.worker_return_per_volatility, key=self.worker_return_per_volatility.get)
        worst_worker_id = min(self.worker_return_per_volatility, key=self.worker_return_per_volatility.get)

        # Step 2: Calculate the amount to be transferred (here we set it as 5% of total free cash as an example)
        transfer_amount = 0.05 * self._get_state().get('total_free_cash')[0]


        max_capital_per_worker = 0.20 * self.total_capital
        # Define the minimum threshold for stock exposure
        min_stock_exposure_threshold = 70000

        # Get the current cash levels of the best and worst workers
        best_worker_cash = self.workers[best_worker_id].current_cash
        worst_worker_cash = self.workers[worst_worker_id].current_cash
        
        transfer_amount = min(transfer_amount, worst_worker_cash - min_stock_exposure_threshold, max_capital_per_worker - best_worker_cash)
        # Step 3: Check if the transfer respects the maximum capital limit for each worker
        if transfer_amount > 0:
            self.total_cash_transfers += 1
            # Step 4: Adjust the cash of both workers
            self.workers[best_worker_id].adjust_cash(transfer_amount)
            self.workers[worst_worker_id].adjust_cash(-transfer_amount)

    # def send_rebalancing_signals(self, max_exposure=100000):
    #     rebalancing_signals = {}

    #     for worker_id, worker in self.workers.items():
    #         current_state = worker._get_state()
    #         current_exposure = current_state['current_cash'] - (current_state['shares_held'] * current_state['current_price'])

    #         if current_exposure > max_exposure:
    #             # Calculate the number of shares to sell to bring exposure to max_exposure
    #             shares_to_sell = (current_exposure - max_exposure) / current_state['current_price']
    #             rebalancing_signals[worker_id] = {'action': 'sell', 'shares': shares_to_sell}
    #         elif current_exposure < max_exposure:
    #             # Calculate the number of shares to buy to bring exposure to max_exposure
    #             shares_to_buy = (max_exposure - current_exposure) / current_state['current_price']
    #             rebalancing_signals[worker_id] = {'action': 'buy', 'shares': shares_to_buy}
    #         else:
    #             rebalancing_signals[worker_id] = {'action': 'hold'}

    #     return rebalancing_signals
    
   
    def _calculate_portfolio_sharpe_ratio(self):
        """
        Calculate the overall portfolio Sharpe ratio.

        The method calculates the annualized Sharpe ratio based on the daily returns of the entire portfolio,
        which is aggregated from the daily returns of individual workers. If the standard deviation of the 
        returns is zero or if any non-finite values are encountered, the Sharpe ratio is returned as zero.

        Returns:
            float: The calculated Sharpe ratio.
        """
        if not self.return_worker_return_memory_dict:
            return np.array([0.0], dtype=np.float32)

        portfolio_returns = np.concatenate([returns for returns in self.return_worker_return_memory_dict.values() if returns.size > 0])

        # If concatenation results in an empty array, return a default Sharpe ratio value of 0
        if portfolio_returns.size == 0:
            return np.array([0.0], dtype=np.float32)

        # Step 2: Calculate Mean and Standard Deviation
        # Ensure that there are no non-finite values in the returns array
        if not np.all(np.isfinite(portfolio_returns)):
            return np.array([0.0], dtype=np.float32)

        # Calculate the mean and standard deviation of the portfolio returns
        mean_returns = np.mean(portfolio_returns)
        std_returns = np.std(portfolio_returns)

        # Step 3: Calculate Sharpe Ratio
        # Calculate the annualized Sharpe ratio
        sharpe_ratio = (mean_returns / std_returns) * np.sqrt(252) if std_returns != 0 else 0.0

        # Check for non-finite values in the calculated Sharpe ratio
        if not np.isfinite(sharpe_ratio):
            sharpe_ratio = np.array([0.0], dtype=np.float32)

        return np.array([sharpe_ratio], dtype=np.float32)