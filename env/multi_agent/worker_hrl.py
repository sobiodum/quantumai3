import gymnasium as gym
from gymnasium import spaces
import numpy as np
from collections import deque
import warnings
import traceback
from env.multi_agent.worker_components.hrl_worker_performance import WorkerPerformanceStats

#! CATCH EROR WITH the following
# with warnings.catch_warnings():
#     warnings.filterwarnings('error')
#     try:
#         # Your suspected code here, e.g., np.mean(empty_array)
#     except RuntimeWarning:
#         traceback.print_exc()


class Spec(object):
    def __init__(self, id, max_episode_steps):
        self.id = id
        self.max_episode_steps = max_episode_steps


class Worker(gym.Env):
    def __init__(self, env_config,trading_cost=0.001, initial_capital=1e6, initial_shares_held = 0,
                 invalid_action_penalty=-0, print_verbosity=1, tic=None):
        super(Worker, self).__init__()
    
        self.df = env_config["df"]
        self.tic = tic
        self.day = 0
        self.data = self.df.loc[self.day, :]
        self.trading_cost = trading_cost
        self.initial_shares_held = initial_shares_held
        self.episode = 0
        self.print_verbosity = print_verbosity

        #State Info
        self.cash_initial = np.float32(initial_capital)
        self.current_cash = np.float32(initial_capital)
        # self.tech_indicator_list = env_config.get("tech_indicator_list")
        self.tech_indicator_list = [ 'avgvol_50',
       'sma_10', 'sma_50', 'sma_100', 'sma_200', 'wma_50', 'rsi_14',
       'volatility_30', 'volatility_100', 'stddev_30', 'dmi_14', 'adx_14',
       'macd', 'atr_14']

        self.current_price = np.float32(self.df.iloc[self.day]['close'])
        

        #Cash Info
 
        self.cash_spent = 0
        self.cash_from_sales = 0

        #Stock Info
        self.shares_held = 0
        self.average_stock_cost = 0.0

        #Manager/HRL related information
        self.peformance_stats = WorkerPerformanceStats(self)

        #Memory & Reward related Info

        self.reward = 0
        self.total_pnl_history = [0]
      
        self.actions_memory = [0]
        self.action_type = []
        self.trading_memory = [0]
        self.return_memory = [0]

        self.invalid_action_penalty = invalid_action_penalty
        self.cash_adjustment = 0  # To track any cash adjustments made by the manager
        self.previous_portfolio_value = self._calculate_assets() 

        #Track position change
     
        self.stock_holding_memory = np.array([initial_shares_held], dtype=np.float32)  # To store the net position after each trade
        self.position_memory = np.array([0], dtype=np.float32)  # To store the net position after each trade
        
        self.position_change_times = []  # To store the times of significant position changes
        self.position_change_sizes = []  # To store the sizes of significant position changes
        self.previous_action_sign = 0  # To store the sign of the previous action

        #Manager Directives
        self.capital_allocation = None
        self.risk_limit = None
        self.position_limit = None


        #Debugging info
        self.invalid_action_count = 0
        self.penalty = 0

        #Trading related
        self.trading_pentalty = 0
        self.total_costs = 0
        self.total_trades= 0
        self.trading_cost_cumulated = 0
        self.current_step_cost = 0

        self.spec = Spec(id="worker-single-stock",
                         max_episode_steps=len(self.df.index.unique()) - 1)

        self.observation_space = spaces.Dict({
            'current_cash': spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
            'shares_held': spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
            'current_price': spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
            'total_costs': spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
            'day': spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
            'tech_indicators': spaces.Box(low=-np.inf, high=np.inf, shape=(14,), dtype=np.float32),
            'pnl': spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
            'return_per_volatility': spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
            'total_trades': spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
            'current_stock_exposure': spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
        
        })

      



        self.action_space = spaces.Dict({
            'type': spaces.Discrete(3),  # 0: hold, 1: buy, 2: sell
            'amount': spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)  # Percentage of cash/shares to use
        })


    
    def set_directives(self, directives):
        self.capital_allocation = directives["capital_allocation"]
        self.risk_limit = directives["risk_limit"]
        self.position_limit = directives["position_limit"]

    def decide_trade_action(self, manager_directives):
        # This method decides the trading action (buy, sell, hold) based on the worker's current state
        # and possibly other factors like market conditions, technical indicators, etc.

        # ... your code to analyze the current state and other factors

        # Here, we need to decide 'type' and 'amount' based on the analysis
        # For simplicity, let's assume we have some methods `decide_trade_type` and `decide_trade_amount`
        # that analyze the current state (and possibly other factors) to decide the trade type and amount

        trade_type = self.decide_trade_type()
        trade_amount = self.decide_trade_amount()

        # Construct the action dictionary
        low_level_action = {
            'type': trade_type,  # 0: hold, 1: buy, 2: sell
            'amount': trade_amount  # a float value between 0 and 1 representing the percentage of cash/shares to use
        }

        return low_level_action

    def decide_trade_type(self):
        # This method decides the trade type (hold, buy, or sell) based on the worker's current state
        # and possibly other factors.

        # ... your code to decide the trade type based on the analysis of the current state and other factors

        # For simplicity, here we randomly decide the trade type
        # In a real implementation, you would use a more sophisticated method to decide the trade type
        return np.random.choice([0, 1, 2])

    def decide_trade_amount(self):
        # This method decides the trade amount (a float value between 0 and 1) based on the worker's current state
        # and possibly other factors.

        # ... your code to decide the trade amount based on the analysis of the current state and other factors

        # For simplicity, here we randomly decide the trade amount
        # In a real implementation, you would use a more sophisticated method to decide the trade amount
        return np.random.uniform(0, 1)
    
    def modify_action_based_on_decision(self, action, manager_decision):
        # ... (logic to modify the action based on the manager's decision)
        # For example, if the manager_decision indicates to reduce the position size, modify the action accordingly
        return 0

    def step(self, action, manager_decision=None):
        done = self.day >= len(self.df.index.unique()) - 1
        if done:
            self._handle_done()
        else:
            self.current_step_cost = 0
            if manager_decision:
                # Modify the action based on the manager's decision
                action = self.modify_action_based_on_decision(action, manager_decision)
                # low_level_action = self.decide_trade_action(manager_directives)
                # action_type = low_level_action['type']
                # action_amount = low_level_action['amount']

            begin_adj_portfolio_value = self._calculate_assets()
            action_type = action['type']
            action_amount = action['amount'][0]
            
    
            if action_type == 0:  # hold
                pass
            elif action_type in [1, 2]:  # buy or sell
                trade = self._handle_trading(action_type,action_amount)
                self.actions_memory.append(action_amount)
                self.trading_memory.append(trade)
            self.action_type.append(action_type)
            self.day += 1
            self.data = self.df.loc[self.day, :]
            self.state = self._get_state()
  
            end_adj_portfolio_value = self._calculate_assets()
            portfolio_return = (end_adj_portfolio_value / begin_adj_portfolio_value) - 1
            if isinstance(portfolio_return, np.ndarray) and portfolio_return.size == 1:
                portfolio_return = portfolio_return.item()
            self.return_memory.append(portfolio_return)
            total_pnl = self._calculate_pnl()
            self.total_pnl_history.append(total_pnl)
            # Calculate the rolling 30 day portfolio volatility
          
            
            
            self.reward = self._calculate_reward(begin_adj_portfolio_value, end_adj_portfolio_value)
        self.cash_adjustment = 0
        self.trading_pentalty = 0
        self.current_step_cost = 0
        truncated = False
        info = {}
        obs = self._get_state()
 
        return obs, self.reward, done, truncated, info
    
    def _calculate_reward(self,begin_adj_portfolio_value, end_adj_portfolio_value):
        reward = 0
        pnl = end_adj_portfolio_value - begin_adj_portfolio_value - self.current_step_cost 
        if self.invalid_action_count > 0:
            reward += self.invalid_action_penalty
        reward = 0.01 * pnl - self.trading_pentalty 
        return np.float32(reward)
        
    
    def _handle_trading(self, action_type, action_amount):
        if self.current_price == 0:
            return 0

        shares_to_sell = shares_to_buy = 0
        
        if action_type == 2: #Selling
            if self.shares_held <= 0.0001:
                self.trading_pentalty += 1
                self.invalid_action_count += 1
                return 0
            action_nominal_value = self.current_price * self.shares_held * action_amount
            shares_to_sell = min(self.shares_held, int(action_nominal_value / self.current_price))
            sell_amount = self.current_price * shares_to_sell * (1 - self.trading_cost)
            
            self.cash_from_sales += sell_amount
            self.current_cash += np.float32(sell_amount)
            self.total_trades += 1
           
        elif action_type == 1:
            if self.current_cash <= 0.9:
                self.trading_pentalty += 1
                self.invalid_action_count += 1
                return 0
            max_affordable_shares = self.current_cash / (self.current_price * (1 + self.trading_cost))
            shares_to_buy = int(min(max_affordable_shares, action_amount * self.current_cash / self.current_price))    
            buy_amount = self.current_price * shares_to_buy * (1 + self.trading_cost)
            self.cash_spent += buy_amount 
            self.current_cash -= np.float32(buy_amount)
            self.total_trades +=1


      
        # Calculate the net position after the trade
        net_position = np.float32(self.shares_held * self.current_price)
        self.position_memory.append(net_position)
        self.shares_held += shares_to_buy - shares_to_sell
        self.current_step_cost += self.current_price * (abs(shares_to_sell) + abs(shares_to_buy)) * self.trading_cost
        self.total_costs += np.float32(self.current_step_cost)
        self.stock_holding_memory.append(self.shares_held)

        return np.float32(shares_to_buy - shares_to_sell)


    def _calculate_assets(self):
        return np.float32(self.current_cash + self.shares_held * self.current_price - self.cash_adjustment)
    

    # def caclulate_metrics(self, begin_adj_portfolio_value, end_adj_portfolio_value):
    #     if begin_adj_portfolio_value != 0:
    #         portfolio_return = (end_adj_portfolio_value[0] / begin_adj_portfolio_value[0]) - 1
    #     else:
    #         portfolio_return = 0
    #     self.return_memory.append(np.float32(portfolio_return))
    

    def _calculate_pnl(self):
        """Calculated accumulated pnl and appends to self.total_pnl_history """
        pnl = (self.cash_from_sales - self.cash_spent)  + self.current_price * self.shares_held
        return np.float32(pnl)
     
    def _get_sharpe_ratio(self):
        """
        Calculate the Sharpe Ratio to be accessed by the manager.
        
        The method calculates the annualized Sharpe ratio based on the daily returns stored in self.return_memory.
        If the standard deviation of the returns is zero or if any non-finite values are encountered,
        the Sharpe ratio is returned as zero.
        
        Returns:
            np.ndarray: A numpy array containing the single Sharpe ratio value.
        """
        returns = np.array(self.return_memory)

        # Ensure that there are no non-finite values in the returns array
        if not np.all(np.isfinite(returns)):
            return np.array([0.0], dtype=np.float32)

        # Calculate the Sharpe ratio
        sharpe_ratio = ((np.mean(returns))  / (np.std(returns))) * np.sqrt(252) if np.std(returns) != 0 else 0.0

        # Check for non-finite values in the calculated Sharpe ratio
        if not np.isfinite(sharpe_ratio):
            sharpe_ratio = 0.0

        return np.array([sharpe_ratio], dtype=np.float32)
    
    def _get_return_memory(self):
        """
        Access the return memory for the manager
        
        Returns:
            np.ndarray: A numpy array containing all return
        """
        return np.array([self.return_memory], dtype=np.float32)
        




    def render(self, mode='human'):
        pass

    def close(self):
        pass




    def reset(self, *, seed=None, options=None):
        
        self._reset_to_initial_values()
        self.episode += 1
        obs = self._get_state()
        info = {}
        return obs, info
    
    def _get_state(self): 
        self.current_price = np.float32(self.data["close"])

        # State representation
        # Current cash, shares held, current price, technical indicators, PnL, and new components
        stock_data = self.data
        # for tech in self.tech_indicator_list:
        #     print(f"Value and shape of {tech}: {stock_data[tech]}, {np.shape(stock_data[tech])}")
        tech_indicators = [np.float32(stock_data[tech]) for tech in self.tech_indicator_list]

        state = {
            'current_cash': np.array([self.current_cash], dtype=np.float32),
            'shares_held': np.array([self.shares_held], dtype=np.float32),
            'current_price': np.array([self.current_price], dtype=np.float32),
            'total_costs': np.array([self.total_costs], dtype=np.float32),  # Total trading costs
            'day': np.array([self.day], dtype=np.float32),  # Current step in the episode
            'tech_indicators': np.array(tech_indicators, dtype=np.float32),
            'pnl': np.array([self._calculate_pnl()], dtype=np.float32),
            'return_per_volatility': np.array([self.peformance_stats.return_per_volatility()], dtype=np.float32),
            'total_trades': np.array([self.total_trades], dtype=np.float32),
            'current_stock_exposure': np.array([self.current_price * self.shares_held], dtype=np.float32)
        }
  
        return state

    def _reset_to_initial_values(self):
        self.shares_held = self.trading_cost_cumulated = self.reward = 0
        self.cash_spent = self.cash_from_sales = 0
        self.current_price = np.float32(self.df.iloc[self.day]['close'])
        self.invalid_action_count = 0
        self.day = 0
        self.current_step_cost = 0
        self.penalty = 0
        self.total_costs = np.float32(0)
        self.total_trades = 0

        # Sum all the values from an array to one int
        
 
        self.actions_memory = [0]
        self.trading_memory = [0]
        #For position tracking
        self.stock_holding_memory = [self.initial_shares_held]  # To store the net position after each trade
        self.position_memory = [0]  # To store the net position after each trade
        self.position_change_times = []
        self.position_change_sizes = []
        self.previous_action_sign = 0
        self.action_type = []
        self.total_pnl_history
        self.current_cash = self.cash_initial
        self.capital_allocation = None
        self.risk_limit = None
        self.position_limit = None
        self.return_memory = [0]

    #! Passing information to Manager

    def _manager_get_current_cash(self):
        return np.float32(self.current_cash)
    
 
    def adjust_cash(self, cash_amount):
        """Adjust the current cash balance by a specified amount."""
        if isinstance(cash_amount, np.ndarray):
            cash_amount = cash_amount.item()
        
        if isinstance(self.current_cash, np.ndarray):
            self.current_cash = self.current_cash.item()
        
        self.current_cash += np.float32(cash_amount)
        self.cash_adjustment += cash_amount
    # def adjust_cash(self, cash_amount):
    #     """Adjust the current cash balance by a specified amount."""
    #     self.current_cash += np.float32(cash_amount)
    #     self.cash_adjustment += cash_amount  # Record the adjustment


    def _handle_done(self):
        return
        if self.episode % self.print_verbosity == 0:

            print(f"day: {self.day}, episode: {self.episode}")
            print(f"total_pnl: {self.total_pnl_history[-1]:0.2f}")
            print(f"self.cash_spent: {self.cash_spent:0.2f}")
            print(f"self.cash_from_sales: {self.cash_from_sales:0.2f}")
            print(f"total_cost: {self.total_costs:0.2f}")
            print(f"total_trades: {self.total_trades}")
            print(f"Begin_portfolio_value: {self.cash_initial }")
            print(f"End stock holdings: {self.stock_holding_memory[-1]}")
            print(f"Last stock price: {self.current_price}")
            print(f"End_portfolio_value: {self._calculate_assets()}")
            print(f"Invalid Actions {self.invalid_action_count}")
            print("=================================")
            # print(f"Sortino: {self._calculate_sortino_ratio():0.3f}")
            # print(f"Max_drawdown: {self._calculate_drawdown():0.3f}%")
            # print(f"Avg_pos_duration: {self._calculate_average_position_duration():0.3f}")
            # print(f"Avg_pos_adjustment: {self._calculate_average_position_adjustment():0.3f}")

            # print(f"begin_total_asset: {self.asset_memory[0]:0.2f}")
            # print(f"end_total_asset: {end_total_asset:0.2f}")
            # print(f"Total Profit(%): {total_profit:0.2f}")
            # print(f"Total Loss(%): {total_loss:0.2f}")
            # print(f"Action Distribution: BUY: {buys} SELL: {sells}  HOLD: {holds}")


#Out for now only

    # def _immediate_pnl(self):
    #     """Calculate the immediate PnL, adjusting for any cash reallocations."""
    #     current_portfolio_value = self._calculate_assets()
    #     pnl = current_portfolio_value - self.previous_portfolio_value - self.cash_adjustment
    #     self.cash_adjustment = 0  # Reset after calculating PnL
    #     self.previous_portfolio_value = current_portfolio_value
    #     return pnl


    
    # def _calculate_rolling_average(self):
    #     if len(self.pnl_history) == 0:
    #         return 0.0
    #     return np.float32(sum(self.pnl_history) / len(self.pnl_history))

    # def _calculate_drawdown(self):
    #     # Adjusting for compounded returns
    #     cumulative_returns = np.cumprod(1 + np.array(self.return_memory)) - 1
    #     running_max = np.maximum.accumulate(cumulative_returns)
    #     drawdown = running_max - cumulative_returns
    #     return np.max(drawdown)
    
    # def _calculate_average_position_duration(self):
    #     if len(self.position_change_times) < 2:
    #         return np.nan
    #     position_durations = np.diff(self.position_change_times)
    #     return np.mean(position_durations)

    # def _calculate_average_position_adjustment(self):
    #     if len(self.position_change_sizes) == 0:
    #         return np.nan
    #     return np.mean(self.position_change_sizes)
    
    # def _calculate_total_profit_vs_total_loss(self):
    #     profits = [r for r in self.return_memory if r > 0]
    #     losses = [r for r in self.return_memory if r < 0]
    #     total_profit = np.sum(profits)
    #     total_loss = np.sum(losses)
    #     return total_profit, total_loss
    
    # def _calculate_sortino_ratio(self):
    #     returns = np.array(self.return_memory)
    #     expected_return = np.mean(returns)
    #     downside_returns = [ret for ret in returns if ret < 0]
    #     downside_var = np.mean([ret**2 for ret in returns if ret < 0])
    #     downside_deviation = np.sqrt(downside_var)
    #     sortino_ratio = expected_return / (downside_deviation + 1e-9)
    #     return sortino_ratio
    

    # def _calculate_action_distribution(self):
    #     buys = self.action_type.count(1)
    #     sells = self.action_type.count(2)
    #     holds = self.action_type.count(0)
    #     return buys,  sells, holds

    #! Trading more complex
    #  def _handle_trading(self, action_type, action_amount):
    #     if self.current_price == 0:
    #         return 0

    #     shares_to_sell = shares_to_buy = 0
        
    #     if action_type == 2:
    #         if self.shares_held <= 0.9:
    #             penalty = 1
    #             self.invalid_action_count += 1
    #             return 0
    #         shares_to_sell = min(self.shares_held, int(action_amount*self.shares_held))
    #         shares_to_sell_left = shares_to_sell
    #         while shares_to_sell_left > 0 and self.ledger:
    #             oldest_entry = self.ledger[0]
    #             if oldest_entry["shares"] <= shares_to_sell_left:
    #                 shares_to_sell_left -= oldest_entry["shares"]
    #                 self.ledger.pop(0)
    #             else:
    #                 oldest_entry["shares"] -= shares_to_sell_left
    #                 shares_to_sell_left = 0

            
    #         sell_amount = self.current_price * shares_to_sell * (1 - self.trading_cost)
    #         self.cash_from_sales += sell_amount
    #         self.current_cash += np.float32(sell_amount)
    #         self.total_trades +=1
           
    #     elif action_type == 1:
    #         if self.current_cash <= 0.9:
    #             penalty = 1
    #             self.invalid_action_count += 1
    #         max_affordable_shares = self.current_cash / (self.current_price * (1 + self.trading_cost))
    #         shares_to_buy = int(min(max_affordable_shares, action_amount * self.current_cash / self.current_price))    
    #         buy_amount = self.current_price * shares_to_buy * (1 + self.trading_cost)
    #         self.cash_spent += buy_amount 
    #         self.current_cash -= np.float32(buy_amount)
    #         self.total_trades +=1
    #         self.ledger.append({"price": self.current_price, "shares": shares_to_buy})

      
    #     # Calculate the net position after the trade
    #     net_position = np.float32(self.shares_held * self.current_price)
    #     self.position_memory.append(net_position)

    #     # Check if there's a significant change in position
    #     current_action_sign = 1 if action_type == 1 else (-1 if action_type == 2 else 0)
    #     if len(self.position_memory) > 1 and current_action_sign != self.previous_action_sign:
    #         # Record the time and size of the position change
    #         self.position_change_times.append(self.day)
    #         self.position_change_sizes.append(abs(self.position_memory[-1] - self.position_memory[-2]))

    #     self.previous_action_sign = current_action_sign
    #     self.shares_held += shares_to_buy - shares_to_sell
    #     self.current_step_cost += self.current_price * (abs(shares_to_sell) + abs(shares_to_buy)) * self.trading_cost
    #     self.total_costs += np.float32(self.current_step_cost) 
    #     total_cost = sum(entry["price"] * entry["shares"] for entry in self.ledger)
    #     total_shares = sum(entry["shares"] for entry in self.ledger)
    #     self.average_stock_cost = total_cost / total_shares if total_shares != 0 else 0
    #     return np.float32(shares_to_buy - shares_to_sell)