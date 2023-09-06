from eod import EodHistoricalData
from datetime import datetime, timedelta
import pandas as pd
import requests
from data_cleaner import forward_fill_missing_stock_data

tickers = ["MSFT.US", "AMZN.US", "AAPL.US", "IBM.US", "F.US"]
start_date = "2000-01-05"
end_date = "2023-08-10"

TECH_INDICATORS = {
    "avgvol": [50],
    'sma': [10, 50, 100, 200],
    "wma": [50],
    'rsi': [14],
    'volatility': [30, 100],
    # 'stochastic': [None],
    'stddev': [30],

    'dmi': [14],
    'adx': [14],
    'macd': [None],
    'atr': [14],
    # 'bbands': [20],
}


def fetch_eod_1d_histroical(tickers, start_date, end_date):
    client = EodHistoricalData("64cbaf10539312.00750558")
    data_df = pd.DataFrame()
    for ticker in tickers:
        try:
            resp = client.get_prices_eod(
                ticker, period="d", from_=start_date, to=end_date)
            temp_df = pd.DataFrame(resp)
            if len(temp_df) > 0:
                temp_df['tic'] = ticker
                temp_df = temp_df.drop(columns=["close"])
                temp_df = temp_df.rename(columns={'adjusted_close': 'close'})

                for indicator, periods in TECH_INDICATORS.items():
                    try:
                        if not periods:  # If the list is empty or None
                            resp_technical = client.get_instrument_ta(
                                ticker, function=indicator, from_=start_date, to=end_date)
                            tech_df = pd.DataFrame(resp_technical)
                            print(resp_technical)
                            temp_df[indicator] = tech_df['value']
                        else:
                            for period in periods:
                                resp_technical = client.get_instrument_ta(
                                    ticker, function=indicator, from_=start_date, to=end_date, period=period)
                                tech_df = pd.DataFrame(resp_technical)
                                temp_df[f'{indicator}_{period}'] = tech_df['value']
                    except Exception as e:
                        print(
                            f"Error fetching Technical data for ticker {ticker} and indicator: {indicator} in period: {period}: {e}")
                data_df = pd.concat([data_df, temp_df], axis=0)
            else:
                print(f"Dataframe empty for ticker: {ticker}")
        except Exception as e:
            print(f"Error fetching data for ticker {ticker}: {e}")
    data_df = data_df.sort_values(["date", "tic"])
    data_df.index = data_df["date"].factorize()[0]
    return data_df

# ... (rest of your code above)


BASE_URL = "https://eodhistoricaldata.com/api/technical"
API_TOKEN = "64cbaf10539312.00750558"


def fetch_eod_1d_histroical_new(tickers, start_date, end_date):
    client = EodHistoricalData(API_TOKEN)
    data_df = pd.DataFrame()
    for ticker in tickers:
        try:
            resp = client.get_prices_eod(
                ticker, period="d", from_=start_date, to=end_date)
            temp_df = pd.DataFrame(resp)
            if len(temp_df) > 0:
                temp_df['tic'] = ticker
                temp_df = temp_df.drop(columns=["close"])
                temp_df = temp_df.rename(columns={'adjusted_close': 'close'})
                if 'date' in temp_df.columns:
                    temp_df.set_index('date', inplace=True)

                for indicator, periods in TECH_INDICATORS.items():
                    try:
                        if not periods:  # If the list is empty or None
                            # Construct and print the request URL
                            request_url = f"{BASE_URL}/{ticker}?order=d&fmt=json&from={start_date}&to={end_date}&function={indicator}&&api_token={API_TOKEN}"

                            resp_technical = requests.get(request_url)
                            resp_technical.raise_for_status()
                            tech_df = pd.DataFrame(
                                resp_technical.json()).set_index('date')
                            temp_df = temp_df.join(
                                tech_df[[indicator]], how='left')
                        else:
                            for period in periods:
                                # Construct and print the request URL
                                period_str = f"&period={period}" if period is not None else ""
                                request_url = f"{BASE_URL}/{ticker}?order=d&fmt=json&from={start_date}&to={end_date}&function={indicator}{period_str}&&api_token={API_TOKEN}"

                                resp_technical = requests.get(request_url)
                                resp_technical.raise_for_status()
                                tech_df = pd.DataFrame(
                                    resp_technical.json()).set_index('date')
                                column_name = f"{indicator}_{period}" if period else indicator
                                temp_df = temp_df.join(tech_df[[indicator]].rename(
                                    columns={indicator: column_name}), how='left')
                    except Exception as e:
                        print(
                            f"Error fetching Technical data for ticker {ticker} and indicator: {indicator} in period: {period}: {e}")
                data_df = pd.concat([data_df, temp_df], axis=0)
            else:
                print(f"Dataframe empty for ticker: {ticker}")
        except Exception as e:
            print(f"Error fetching data for ticker {ticker}: {e}")
    data_df = data_df.reset_index()
    data_df = data_df.sort_values(["date", "tic"])
    data_df.index = data_df["date"].factorize()[0]
    return data_df


def fetch_eod_1d_histroical_API_incl_Technicals(tickers, start_date, end_date):
    client = EodHistoricalData(API_TOKEN)
    data_df = pd.DataFrame()
    for ticker in tickers:
        try:
            resp = client.get_prices_eod(
                ticker, period="d", from_=start_date, to=end_date)
            temp_df = pd.DataFrame(resp)
            if len(temp_df) > 0:
                temp_df['tic'] = ticker
                temp_df = temp_df.drop(columns=["close"])
                temp_df = temp_df.rename(columns={'adjusted_close': 'close'})
                if 'date' in temp_df.columns:
                    temp_df.set_index('date', inplace=True)

                for indicator, periods in TECH_INDICATORS.items():
                    try:
                        if not periods:  # If the list is empty or None
                            # Construct and print the request URL
                            request_url = f"{BASE_URL}/{ticker}?order=d&fmt=json&from={start_date}&to={end_date}&function={indicator}&&api_token={API_TOKEN}"

                            resp_technical = requests.get(request_url)
                            resp_technical.raise_for_status()
                            tech_df = pd.DataFrame(
                                resp_technical.json()).set_index('date')
                            temp_df = temp_df.join(
                                tech_df[[indicator]], how='left')
                        else:
                            for period in periods:
                                # Construct and print the request URL
                                period_str = f"&period={period}" if period is not None else ""
                                request_url = f"{BASE_URL}/{ticker}?order=d&fmt=json&from={start_date}&to={end_date}&function={indicator}{period_str}&&api_token={API_TOKEN}"

                                resp_technical = requests.get(request_url)
                                resp_technical.raise_for_status()
                                tech_df = pd.DataFrame(
                                    resp_technical.json()).set_index('date')
                                column_name = f"{indicator}_{period}" if period else indicator
                                temp_df = temp_df.join(tech_df[[indicator]].rename(
                                    columns={indicator: column_name}), how='left')
                    except Exception as e:
                        print(
                            f"Error fetching Technical data for ticker {ticker} and indicator: {indicator} in period: {period}: {e}")
                data_df = pd.concat([data_df, temp_df], axis=0)
            else:
                print(f"Dataframe empty for ticker: {ticker}")
        except Exception as e:
            print(f"Error fetching data for ticker {ticker}: {e}")
    data_df = data_df.reset_index()
    data_df = data_df.sort_values(["date", "tic"])
    data_df = forward_fill_missing_stock_data(data_df)
    data_df.index = data_df["date"].factorize()[0]
    return data_df


def forward_fill_all_columns(df):
    # Step 1: Create a multi-index of all possible combinations of `date` and `tic`.
    all_dates = df['date'].unique()
    all_tickers = df['tic'].unique()
    multi_index = pd.MultiIndex.from_product(
        [all_dates, all_tickers], names=['date', 'tic'])

    # Step 2: Reindex the DataFrame with this multi-index.
    df_reindexed = df.set_index(['date', 'tic']).reindex(multi_index)

    # Step 3: Group by the `tic` column and apply a forward fill.
    filled_df = df_reindexed.groupby('tic').ffill().reset_index()

    return filled_df
