from __future__ import annotations
from env.env_components.reward import RewardFunction, RewardTradeBudgetPenalty, RewardTradeBudgetPenalty
from env.env_components.trading import TradingFunction

from typing import List
from datetime import datetime
import gymnasium as gym
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gymnasium import spaces
from gymnasium.utils import seeding


from collections import deque, defaultdict
matplotlib.use("Agg")


class Spec(object):
    def __init__(self, id, max_episode_steps):
        self.id = id
        self.max_episode_steps = max_episode_steps


class TradingOrdinary():
    def __init__(self, env):
        self.env = env

    def ordinary_sell_stock(self, ticker, action):
        index = self.env.tickers_list.index(ticker)
        current_share_holdings = self.env.get_current_share_holding(index)

        close_price = self.env.get_closing_price(index)
        if close_price == 0:
            return 0
        if current_share_holdings > 0:
            sell_num_shares = min(
                abs(action), current_share_holdings
            )

            sell_amount = (
                close_price
                * sell_num_shares
                * (1 - self.env.trading_cost_pct)

            )

            self.env.state[0] += sell_amount
            self.env.update_num_shares(index, -sell_num_shares)

            self.env.current_step_cost += -(
                close_price
                * sell_num_shares
                * self.env.trading_cost_pct
            )
            self.env.trades += 1
        else:
            sell_num_shares = 0

        return sell_num_shares

    def ordinary_buy_stock(self, ticker, action):
        index = self.env.tickers_list.index(ticker)
        close_price = self.env.get_closing_price(index)
        if close_price == 0:
            return 0

        current_total_assets = self.env.calculate_assets()
        max_stock_value = self.env.max_position_weight * current_total_assets
        current_stock_value = close_price * \
            self.env.get_current_share_holding(index)
        proposed_buy_value = close_price * action
        if current_stock_value + proposed_buy_value > max_stock_value:
            additional_buyable_value = max(
                0, max_stock_value - current_stock_value)
            action = additional_buyable_value / close_price
            action = int(action)

        max_shares_possible = self.env.state[0] // (
            close_price * (1 + self.env.trading_cost_pct)
        )

        buy_num_shares = min(max_shares_possible, action)

        if buy_num_shares > 0:
            buy_amount = close_price * buy_num_shares * \
                (1 + self.env.trading_cost_pct)
            self.env.state[0] -= buy_amount
            self.env.update_num_shares(index, buy_num_shares)
            self.env.current_step_cost += - \
                (close_price * buy_num_shares * self.env.trading_cost_pct)
            self.env.trades += 1

        return buy_num_shares


class Worker(gym.Env):

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        env_config: None,
        day=0,
        previous_state=[],
        model_name="",
        mode="",
        iteration="",
        asset_reward_weight=0.5,
        sharpe_reward_weight=0.5,
        dynamic_weighting=True,
        max_position_weight=0.30,
        trade_ratio: int = 0.17,
        trade_penalty_factor=-10000,
        reward_scaling: float = 1e-3,
        hmax: int = 70000,
    ):

        if env_config is None:
            env_config = {}
        self.day = day
        self.df = env_config["df"]
        self.spec = Spec(id="StockTradingEnvAdvanced-v0",
                         max_episode_steps=len(self.df.index.unique()) - 1)
        self.data = self.df.loc[self.day, :]
        self.tickers_list = list(self.df.tic.unique())
        self.number_of_trade_budget = int(
            len(self.df) * trade_ratio * len(self.tickers_list))

        self.trade_penalty_factor = trade_penalty_factor
        self.current_step_cost = 0
        self.stock_list = self.df.tic.unique()

        self.stock_dim = env_config.get("stock_dim")
        self.hmax = env_config.get("hmax", hmax)
        self.num_stock_shares = env_config.get(
            "num_stock_shares")

        self.initial_amount = env_config.get(
            "initial_amount")  # get the initial cash
        self.trading_cost_pct = env_config.get(
            "trading_cost_pct")

        self.max_position_weight = max_position_weight

        self.reward_scaling = env_config["reward_scaling"]
        self.state_space = env_config.get("state_space")
        self.action_space = env_config.get("action_space")
        self.tech_indicator_list = env_config.get(
            "tech_indicator_list")
        # ? Action & Observation Space
        # initalize state
        self.action_space = spaces.Box(
            low=-1, high=1, shape=(self.action_space,))

        self.timestep_since_last_trade = 0
        self.trades = 0
        self.state = self._initiate_state_hstack()
        self.state_components = [
            self._get_cash_balance,
            self._get_stock_prices,
            self._get_stock_owned,
            self._get_tech_indicators,
            self._get_current_step_cost,
            self._get_timestep_since_last_trade,
            self._get_number_of_trade_budget,
            self._get_number_of_done_trades

        ]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.calculate_state_shape(),), dtype=np.float32
        )
        self.terminal = False
        self.print_verbosity = env_config.get("print_verbosity")
        self.reward_function = RewardTradeBudgetPenalty(self)
        self.trading_function = TradingOrdinary(self)

        self.previous_state = previous_state
        self.model_name = model_name
        self.mode = mode
        self.iteration = iteration
        self.stocks_owned = {ticker: initial_holding for ticker, initial_holding in zip(
            self.tickers_list, self.num_stock_shares)}

        # ? initialize reward related stuff
        self.dynamic_weighting = dynamic_weighting
        self.asset_reward_weight = asset_reward_weight
        self.sharpe_reward_weight = sharpe_reward_weight
        self.portfolio_return_memory = [0]
        self.reward = 0
        # ? Track Trading behavior
        self.total_costs = 0
        self.episode = 0
        self.asset_memory = [
            self.initial_amount
            + np.sum(
                np.array(self.num_stock_shares)
                * np.array(self.state[1::(2 + len(self.tech_indicator_list))][:self.stock_dim])
            )
        ]
        self.portfolio_allocation_memory = []
        # ? Track Performance
        self.episode_end_date = []
        self.episode_start_date = []
        self.rewards_memory = []
        self.actions_memory = []
        self.state_memory = (
            []
        )  # we need sometimes to preserve the state in the middle of trading process
        self.portfolio_memory = {}
        for index, stock in enumerate(self.stock_list):
            # Initialize or reset the stock's memory in portfolio_memory with the initial number of shares
            self.portfolio_memory[stock] = [self.num_stock_shares[index]]

        self.date_memory = [self._get_date()]
        #         self.logger = Logger('results',[CSVOutputFormat])
        # self.reset()
        self._seed()

    def calculate_assets(self):

        assets = self.state[0]

        close_prices = self.state[1::(
            2 + len(self.tech_indicator_list))][:self.stock_dim]
        stocks_owned = self.state[2::(
            2 + len(self.tech_indicator_list))][:self.stock_dim]

        total_stock_value = np.sum(close_prices * stocks_owned)

        assets += total_stock_value
        return assets

    def step(self, actions):

        self.terminal = self.day >= len(self.df.index.unique()) - 1
        if self.terminal:

            self.episode_end_date.append(self._get_date())
            self.handle_terminate()
        else:
            self.current_step_cost = 0
            actions = (actions * self.hmax).astype(int)

            begin_total_asset = self.calculate_assets()

            # argsort_actions = np.argsort(actions)
            sell_tickers = [self.tickers_list[i]
                            for i in np.where(actions < 0)[0]]
            buy_tickers = [self.tickers_list[i]
                           for i in np.where(actions > 0)[0]]

            for ticker in sell_tickers:
                action_value = actions[self.tickers_list.index(ticker)]
                # actions[self.tickers_list.index(ticker)] = self._sell_stock(
                #     ticker, action_value) * (-1)
                actions[self.tickers_list.index(ticker)] = self.trading_function.ordinary_sell_stock(
                    ticker, action_value) * (-1)

            for ticker in buy_tickers:
                action_value = actions[self.tickers_list.index(ticker)]
                # actions[self.tickers_list.index(ticker)] = self._buy_stock(
                #     ticker, action_value)
                actions[self.tickers_list.index(ticker)] = self.trading_function.ordinary_buy_stock(
                    ticker, action_value)

            self.actions_memory.append(actions)
            self.total_costs += self.current_step_cost
            self.timestep_since_last_trade += 1

            # state: s -> s+1
            self.day += 1
            self.data = self.df.loc[self.day, :]

            # self.state = self._update_state()
            self.state = self._update_state_hstack()

            end_total_asset = self.calculate_assets()
            self.caclulate_metrics(begin_total_asset, end_total_asset)

            self.asset_memory.append(end_total_asset)
            self.date_memory.append(self._get_date())
            self.reward = self.reward_function.compute_reward(
                begin_total_asset, end_total_asset)

            self.reward = float(self.reward)

            self.rewards_memory.append(self.reward)

            for index, stock in enumerate(self.stock_list):
                # Update the stock's memory in portfolio_memory with the current number of shares
                # Assuming you want to append the current number of shares to the stock's memory list
                self.portfolio_memory[stock].append(
                    self.num_stock_shares[index])

            self.state_memory.append(
                self.state
            )  # add current state in state_recorder for each step
        info = {

        }

        # return self.state.astype(np.float32), self.reward, self.terminal, info
        truncated = False

        return self.state.astype(np.float32), self.reward, self.terminal, truncated, info

        # return self.state.astype(np.float32), self.reward, self.terminal, False, info

    def caclulate_metrics(self, begin_total_asset, end_total_asset):
        portfolio_return = (end_total_asset / begin_total_asset) - 1
        self.portfolio_return_memory.append(portfolio_return)

    def reset(
        self,
        *,
        seed=None,
        options=None,
    ):
        # initiate state
        self.episode_start_date.append(self._get_date())
        if len(self.episode_start_date) > 3:
            self.episode_start_date.pop(0)
        if len(self.episode_end_date) > 3:
            self.episode_end_date.pop(0)
        self.day = 0
        self.data = self.df.loc[self.day, :]
        self.state = self._initiate_state_hstack()
        self.stocks_owned = {ticker: initial_holding for ticker, initial_holding in zip(
            self.tickers_list, self.num_stock_shares)}

        self.current_step_cost = 0
        self.asset_memory = [
            self.initial_amount
            + np.sum(
                np.array(self.num_stock_shares)
                * np.array(self.state[1::(2 + len(self.tech_indicator_list))][:self.stock_dim])

            )
        ]
        # print("reset func - self.asset_memory:", self.asset_memory)
        self.portfolio_return_memory = [0]
        self.timestep_since_last_trade = 0
        self.total_costs = 0
        self.trades = 0
        self.terminal = False
        # self.iteration=self.iteration
        self.rewards_memory = []
        self.actions_memory = []
        self.date_memory = [self._get_date()]
        self.episode += 1
        for index, stock in enumerate(self.stock_list):
            # Reset the stock's memory in portfolio_memory to the initial number of shares
            self.portfolio_memory[stock] = [self.num_stock_shares[index]]

        worker_obs = self.state.astype(np.float32)
        worker_info = {}
        return worker_obs, worker_info
        # return {"worker": self.state.astype(np.float32)}, {}
        # return self.state.astype(np.float32), {}

    def calculate_sharpe_ratio(self):
        returns = np.array(self.portfolio_return_memory)
        return (np.mean(returns) / (np.std(returns) + 1e-9)) * np.sqrt(252)

    def render(self, mode="human", close=False):
        return self.state

    def _initiate_state_hstack(self):

        amount = np.array([self.initial_amount], dtype=np.float32)

        state_components = [amount]
        for idx, tic in enumerate(self.tickers_list):
            stock_data = self.data[self.data["tic"] == tic]
            if len(stock_data) == 0:
                print(f"No data found for ticker {tic} on date {self.day}")
            close_price = np.array(
                [stock_data["close"].values[0]], dtype=np.float32)
            stock_owned = np.array(
                [self.num_stock_shares[idx]], dtype=np.float32)
            tech_indicators = [np.array(
                [stock_data[tech].values[0]], dtype=np.float32) for tech in self.tech_indicator_list]
            state_components.extend(
                [close_price, stock_owned] + tech_indicators)
        state_components.append(
            np.array([self.current_step_cost], dtype=np.float32))
        state_components.append(self._get_timestep_since_last_trade())
        state_components.append(self._get_number_of_trade_budget())
        state_components.append(self._get_number_of_done_trades())

        state = np.hstack(state_components, dtype=np.float32)

        return state
        # return state

    def _update_state_hstack(self):
        current_amount = [self.state[0]]
        state_components = [current_amount]
        for tic in self.tickers_list:
            stock_data = self.data[(self.data["tic"] == tic)]
            if stock_data.empty:
                print(f"no data for ticker: {tic}")
                continue
            close_price = np.array(
                [stock_data["close"].values[0]], dtype=np.float32)
            stock_owned = np.array([self.stocks_owned[tic]], dtype=np.float32)
            tech_indicators = [np.array(
                [stock_data[tech].values[0]], dtype=np.float32) for tech in self.tech_indicator_list]
            state_components.extend(
                [close_price, stock_owned] + tech_indicators)
        state_components.append(
            np.array([self.current_step_cost], dtype=np.float32))

        state_components.append(self._get_timestep_since_last_trade())
        state_components.append(self._get_number_of_trade_budget())
        state_components.append(self._get_number_of_done_trades())

        state = np.hstack(state_components, dtype=np.float32)
        return state
        # return state

    def _get_cash_balance(self):
        return np.array([self.state[0]])

    def _get_stock_prices(self):
        return self.state[1::(2 + len(self.tech_indicator_list))][:self.stock_dim]

    def _get_stock_owned(self):
        return self.state[2::(2 + len(self.tech_indicator_list))][:self.stock_dim]

    def _get_tech_indicators(self):
        tech_indicators = []
        for i in range(self.stock_dim):
            start_idx = 3 + i * (2 + len(self.tech_indicator_list))
            end_idx = start_idx + len(self.tech_indicator_list)
            tech_indicators.extend(self.state[start_idx:end_idx])

        return np.array(tech_indicators)

    def _get_current_step_cost(self):
        return np.array([self.current_step_cost])

    def calculate_state_shape(self):
        return sum(comp().shape[0] for comp in self.state_components)

    def get_current_share_holding(self, index):
        return self.state[2 + index * (2 + len(self.tech_indicator_list))]

    def _get_timestep_since_last_trade(self):
        return np.array([self.timestep_since_last_trade], dtype=np.float32)

    def _get_number_of_trade_budget(self):
        return np.array([self.number_of_trade_budget], dtype=np.float32)

    def _get_number_of_done_trades(self):
        return np.array([self.trades], dtype=np.float32)

    def get_closing_price(self, index):

        return self.state[1 + index * (2 + len(self.tech_indicator_list))]

    def update_num_shares(self, index, amount):
        tic = self.tickers_list[index]
        self.state[2 + index *
                   (2 + len(self.tech_indicator_list))] += amount
        self.stocks_owned[tic] = self.state[2 +
                                            index * (2 + len(self.tech_indicator_list))]

    def _get_date(self):
        if len(self.df.tic.unique()) > 1:
            date = self.data.date.unique()[0]
        else:
            date = self.data.date
        return date

    def memory_collection(self):

        print("Date Memory Length:", len(self.date_memory))

        print("Asset Memory Length:", len(self.asset_memory))
        print("Portfolio Return Memory Length:",
              len(self.portfolio_return_memory))

        df_account_value = pd.DataFrame(
            {"date": self.date_memory, "account_value": self.asset_memory})
        df_return_memory = pd.DataFrame(
            {"date": self.date_memory, "daily_return": self.portfolio_return_memory})

        return df_account_value, df_return_memory

    def save_return_memory(self):

        df_return_memory = pd.DataFrame(
            {"date": self.date_memory, "daily_return": self.portfolio_return_memory})

        return df_return_memory

    def save_asset_memory(self):
        df_account_value = pd.DataFrame(
            {"date": self.date_memory, "account_value": self.asset_memory}
        )
        return df_account_value

    def save_portfolio_memory(self):
        df_portfolio = pd.DataFrame(self.portfolio_memory)
        df_portfolio['date'] = self.date_memory
        return df_portfolio

    def handle_terminate(self):
        end_total_asset = self.calculate_assets()
        df_total_value = pd.DataFrame(self.asset_memory)
        tot_reward = end_total_asset - self.asset_memory[0]

        df_total_value.columns = ["account_value"]
        df_total_value["date"] = self.date_memory
        df_total_value["daily_return"] = df_total_value["account_value"].pct_change(
            1
        )

        annualized_return = (
            ((end_total_asset / self.asset_memory[0]) ** (365 / (self.day)))-1)*100

        if df_total_value["daily_return"].std() != 0:
            sharpe = (
                (252**0.5)
                * df_total_value["daily_return"].mean()
                / df_total_value["daily_return"].std()
            )
        df_rewards = pd.DataFrame(self.rewards_memory)
        df_rewards.columns = ["account_rewards"]
        df_rewards["date"] = self.date_memory[:-1]
        if self.episode % self.print_verbosity == 0:
            print(f"day: {self.day}, episode: {self.episode}")
            print(f"begin_total_asset: {self.asset_memory[0]:0.2f}")
            print(f"end_total_asset: {end_total_asset:0.2f}")
            print(f"total_reward: {tot_reward:0.2f}")
            print(f"total_cost: {self.total_costs:0.2f}")
            print(f"total_trades: {self.trades}")
            if df_total_value["daily_return"].std() != 0:
                print(f"Sharpe: {sharpe:0.3f}")
            # print(f"Equal wgt portfolio: {equal_weighted_ann_return:0.2f}%")
            print(f"Annual Return: {annualized_return:0.2f}%")
            print("=================================")

        if (self.model_name != "") and (self.mode != ""):
            df_actions = self.save_action_memory()
            df_actions.to_csv(
                "results/actions_{}_{}_{}.csv".format(
                    self.mode, self.model_name, self.iteration
                )
            )
            df_total_value.to_csv(
                "results/account_value_{}_{}_{}.csv".format(
                    self.mode, self.model_name, self.iteration
                ),
                index=False,
            )
            df_rewards.to_csv(
                "results/account_rewards_{}_{}_{}.csv".format(
                    self.mode, self.model_name, self.iteration
                ),
                index=False,
            )
            plt.plot(self.asset_memory, "r")
            plt.savefig(
                "results/account_value_{}_{}_{}.png".format(
                    self.mode, self.model_name, self.iteration
                )
            )
            plt.close()

        return self.state, self.reward, self.terminal, False, {}

    def save_action_memory(self):
        if len(self.df.tic.unique()) > 1:
            # date and close price length must match actions length
            date_list = self.date_memory[:-1]
            df_date = pd.DataFrame(date_list)
            df_date.columns = ["date"]

            action_list = self.actions_memory
            df_actions = pd.DataFrame(action_list)
            df_actions.columns = self.data.tic.values
            df_actions.index = df_date.date
            # df_actions = pd.DataFrame({'date':date_list,'actions':action_list})
        else:
            date_list = self.date_memory[:-1]
            action_list = self.actions_memory
            df_actions = pd.DataFrame(
                {"date": date_list, "actions": action_list})
        return df_actions

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_sb_env(self):
        e = DummyVecEnv([lambda: self])
        obs = e.reset()
        return e, obs
    # def calculate_reward(self, end_total_asset, begin_total_asset):

    #     asset_reward = (end_total_asset - begin_total_asset)

    #     trade_penalty = self.trade_penalty_factor * \
    #         (self._get_number_of_done_trades()[0] /
    #          self._get_number_of_trade_budget()[0])

    #     return (asset_reward + trade_penalty +
    #             self.current_step_cost)*self.reward_scaling

    # def _sell_stock(self, ticker, action):
    #     # print("sell-function - action/ticker:", action, ticker)
    #     index = self.tickers_list.index(ticker)
    #     current_share_holdings = self.get_current_share_holding(index)

    #     close_price = self.get_closing_price(index)
    #     if close_price == 0:
    #         return 0
    #     if current_share_holdings > 0:
    #         sell_num_shares = min(
    #             abs(action), current_share_holdings
    #         )

    #         sell_amount = (
    #             close_price
    #             * sell_num_shares
    #             * (1 - self.trading_cost_pct)

    #         )

    #         self.state[0] += sell_amount
    #         self.update_num_shares(index, -sell_num_shares)

    #         self.current_step_cost += -(
    #             close_price
    #             * sell_num_shares
    #             * self.trading_cost_pct
    #         )
    #         self.trades += 1
    #     else:
    #         sell_num_shares = 0

    #     return sell_num_shares

    # def _buy_stock(self, ticker, action):
    #     # print("buy-function - action/ticker:", action, ticker)
    #     index = self.tickers_list.index(ticker)
    #     close_price = self.get_closing_price(index)
    #     if close_price == 0:
    #         return 0

    #     current_total_assets = self.calculate_assets()
    #     max_stock_value = self.max_position_weight * current_total_assets
    #     current_stock_value = close_price * \
    #         self.get_current_share_holding(index)
    #     proposed_buy_value = close_price * action
    #     if current_stock_value + proposed_buy_value > max_stock_value:
    #         additional_buyable_value = max(
    #             0, max_stock_value - current_stock_value)
    #         action = additional_buyable_value / close_price
    #         action = int(action)

    #     max_shares_possible = self.state[0] // (
    #         close_price * (1 + self.trading_cost_pct)
    #     )

    #     buy_num_shares = min(max_shares_possible, action)

    #     buy_amount = (
    #         close_price
    #         * buy_num_shares
    #         * (1 + self.trading_cost_pct)
    #     )

    #     self.state[0] -= buy_amount

    #     self.update_num_shares(index, buy_num_shares)

    #     self.current_step_cost += -(
    #         close_price * buy_num_shares *
    #         self.trading_cost_pct
    #     )
    #     self.trades += 1

    #     return buy_num_shares

    # def _update_state(self):
    #     if len(self.df.tic.unique()) > 1:
    #         # for multiple stock
    #         state = (
    #             [self.state[0]]
    #             + self.data.close.values.tolist()
    #             + list(self.state[(self.stock_dim + 1)
    #                    : (self.stock_dim * 2 + 1)])
    #             + sum(
    #                 (
    #                     self.data[tech].values.tolist()
    #                     for tech in self.tech_indicator_list
    #                 ),
    #                 [],
    #             )
    #         )

    #     else:
    #         # for single stock
    #         state = (
    #             [self.state[0]]
    #             + [self.data.close]
    #             + list(self.state[(self.stock_dim + 1)
    #                    : (self.stock_dim * 2 + 1)])
    #             + sum(([self.data[tech]]
    #                   for tech in self.tech_indicator_list), [])
    #         )

    #     return state

    # def _initiate_state(self):

    #     if len(self.df.tic.unique()) > 1:
    #         # for multiple stock
    #         state = (
    #             [self.initial_amount]
    #             + self.data.close.values.tolist()
    #             + self.num_stock_shares
    #             + sum(
    #                 (
    #                     self.data[tech].values.tolist()
    #                     for tech in self.tech_indicator_list
    #                 ),
    #                 [],
    #             )
    #         )  # append initial stocks_share to initial state, instead of all zero
    #     else:
    #         # for single stock
    #         state = (
    #             [self.initial_amount]
    #             + [self.data.close]
    #             + [0] * self.stock_dim
    #             + sum(([self.data[tech]]
    #                    for tech in self.tech_indicator_list), [])
    #         )

    #     return state
