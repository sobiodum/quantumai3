from abc import ABC, abstractmethod


class TradingFunction(ABC):

    @abstractmethod
    def ordinary_sell_stock(self, *args, **kwargs):
        pass

    @abstractmethod
    def ordinary_buy_stock(self, *args, **kwargs):
        pass


class TradingOrdinary(TradingFunction):
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
