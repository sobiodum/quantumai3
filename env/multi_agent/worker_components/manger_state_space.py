import numpy as np
import pandas as pd


class ManagerStateClass():
    def __init__(self, env):
        self.env = env

    def calculate_state_shape(self):
        # Assuming cash balance, portfolio value, and sharpe ratio are scalars.
        # stock_holdings is an array with length equal to n_stocks.
        return (3 + len(self.env.n_stocks), )

    def _initiate_state(self, info):
        state_components = [
            info["cash_balance"][0] if isinstance(
                info["cash_balance"], np.ndarray) else info["cash_balance"],
            info["portfolio_value"],
            # using * to unpack the array into the list
            *info["stock_holdings"],
            info["sharpe_ratio"]
        ]

        state = np.array(state_components, dtype=np.float32)

        return state

    def _update_state(self, info):
        state_components = [
            info["cash_balance"][0] if isinstance(
                info["cash_balance"], np.ndarray) else info["cash_balance"],
            info["portfolio_value"],
            *info["stock_holdings"],
            info["sharpe_ratio"]
        ]

        state = np.array(state_components, dtype=np.float32)

        return state

    #     self.state_components = [
    #         self.cash_balance,
    #         self.portfolio_value,
    #         self.stock_holdings,
    #         self.sharpe_ratio,
    #     ]

    # def cash_balance(self):
    #     pass

    # def portfolio_value(self):
    #     pass

    # def stock_holdings(self):
    #     pass

    # def sharpe_ratio(self):
    #     pass

    # def calculate_state_shape(self):
    #     return sum(comp().shape[0] for comp in self.state_components)
