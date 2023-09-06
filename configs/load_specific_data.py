import pandas as pd
from preprocessors.data_cleaner import forward_fill_missing_stock_data


def pharma_basket():
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
    return train_df, validate_df, test_df, stock_dimension, state_space, indicators
