import pandas as pd


def fill_at_once(df: pd.DataFrame) -> pd.DataFrame:
    df = df.groupby('tic').apply(lambda group: group.fillna(method="backfill"))
    df = df.reset_index(drop=True)
    df.index = df["date"].factorize()[0]
    return df


def backward_fill(df: pd.DataFrame) -> pd.DataFrame:
    df = df.groupby('tic').apply(lambda group: group.fillna(method="backfill"))
    df = df.reset_index(drop=True)
    df.index = df["date"].factorize()[0]
    return df


def forward_fill_missing_stock_data(df: pd.DataFrame) -> pd.DataFrame:
    # Create all possible (date, tic) combinations
    all_dates = df['date'].unique()
    all_tics = df['tic'].unique()
    all_combinations = pd.MultiIndex.from_product(
        [all_dates, all_tics], names=['date', 'tic']).to_frame(index=False)

    # Merge with the original data
    df = pd.merge(all_combinations, df, on=['date', 'tic'], how='left')

    # Sort by tic and date for forward filling
    df.sort_values(by=['tic', 'date'], inplace=True)

    # Group by 'tic' and forward fill
    df = df.groupby('tic').apply(lambda group: group.fillna(method="ffill"))
    df.index = df["date"].factorize()[0]

    return df.reset_index(drop=True)


def fill_with_zeros(df: pd.DataFrame) -> pd.DataFrame:
    return df.fillna(0)
