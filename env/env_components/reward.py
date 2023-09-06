from abc import ABC, abstractmethod


class RewardFunction(ABC):

    @abstractmethod
    def compute_reward(self, *args, **kwargs):
        pass


class RewardTradeBudgetPenalty(RewardFunction):
    def __init__(self, env):
        self.env = env

    def compute_reward(self, begin_total_asset, end_total_asset):
        asset_reward = (end_total_asset - begin_total_asset)
        trade_penalty = self.env.trade_penalty_factor * \
            (self.env._get_number_of_done_trades()[0] /
             self.env._get_number_of_trade_budget()[0])
        return (asset_reward + trade_penalty +
                self.env.current_step_cost) * self.env.reward_scaling


class RewardFunctionB(RewardFunction):

    def compute_reward(self, *args, **kwargs):
        # Implement the logic for reward function B
        pass
