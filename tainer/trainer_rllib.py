import ray
from ray.air.config import RunConfig, ScalingConfig
from ray.train.rl import RLTrainer
from ray.rllib.algorithms.bc.bc import BC
from ray.rllib.algorithms import a2c
from ray.tune.registry import register_env
from env.env_ST_advanced_raylib import StockTradingEnvAdvanced
from preprocessors.data_cleaner import forward_fill_missing_stock_data
# That works


def env_creator(env_config):
    # Assuming this is your environment
    return StockTradingEnvAdvanced(env_config)


register_env("stock_trading_env", env_creator)

df = pd.read_pickle("pharma.pkl")
df = df[df['tic'] != 'NVO.US']
df = df[df['tic'] != 'DHR.US']
df
df = forward_fill_missing_stock_data(df)

train_start = "1986-01-02"
train_end = "2010-12-31"
validate_start = "2011-01-01"
validate_end = "2015-12-31"
test_start = "2016-01-01"
test_end = "2023-08-17"


indicators = ['avgvol_50',
              'sma_10', 'sma_50', 'sma_100', 'sma_200', 'wma_50', 'rsi_14',
              'volatility_30', 'volatility_100', 'stddev_30', 'dmi_14', 'adx_14',
              'macd', 'atr_14', "volume"]


def data_split(df, start, end, target_date_col="date"):
    data = df[(df[target_date_col] >= start) & (df[target_date_col] < end)]
    data = data.sort_values([target_date_col, "tic"], ignore_index=True)
    data.index = data[target_date_col].factorize()[0]
    return data


train_df = data_split(df, train_start, train_end)
validate_df = data_split(df, validate_start, validate_end)
test_df = data_split(df, test_start, test_end)
stock_dimension = len(train_df.tic.unique())
state_space = 1 + 2*stock_dimension + len(indicators)*stock_dimension

trainer = RLTrainer(
    run_config=RunConfig(stop={"training_iteration": 5}),
    scaling_config=ScalingConfig(
        num_workers=2,
        use_gpu=False,
    ),

    algorithm=a2c.A2C,
    config={
        "env": "stock_trading_env",
        "env_config": {
            "hmax": 50000,
            "initial_amount": 1000000,
            "trading_cost_pct": 0.001,
            "state_space": state_space,
            "stock_dim": stock_dimension,
            "tech_indicator_list": indicators,
            "action_space": stock_dimension,
            "reward_scaling": 1e-4,
            "print_verbosity": 1,
            "num_stock_shares": np.zeros(stock_dimension).tolist(),
            "df": train_df,
        },
        "framework": "tf",
        "evaluation_num_workers": 1,
        "evaluation_interval": 1,
        "evaluation_config": {"input": "sampler"},
    },
)
result = trainer.fit()
