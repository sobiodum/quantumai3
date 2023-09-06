import numpy as np
import pandas as pd


class ManagerRiskMetrics():
    def __init__(self, env):
        self.env = env

    def _get_cash_balance(self):
        return np.array([self.state[0]])

    def _get_sharpe_ratio(self):
        return self.state[1::(2 + len(self.tech_indicator_list))][:self.stock_dim]

    def _get_total_trades(self):
        return self.state[2::(2 + len(self.tech_indicator_list))][:self.stock_dim]

    def _get_current_step_cost(self):
        return np.array([self.current_step_cost])

    def calculate_state_shape(self):
        return sum(comp().shape[0] for comp in self.state_components)

    def _initiate_state(self):
        state_components = []
        state_components.append(self._get_cash_balance())
        state_components.append(self._get_sharpe_ratio())
        state_components.append(self._get_total_trades())
        state_components.append(self._get_current_step_cost())

        state = np.hstack(state_components, dtype=np.float32)

        return state

    def calculate_state_shape(self):
        return sum(comp().shape[0] for comp in self.state_components)

    def _update_state(self):
        state_components = []
        state_components.append(self._get_cash_balance())
        state_components.append(self._get_sharpe_ratio())
        state_components.append(self._get_total_trades())
        state_components.append(self._get_current_step_cost())

        state = np.hstack(state_components, dtype=np.float32)

        return state
