from datetime import timedelta
import pandas as pd
from definitions.constants import BOND_TICKERS, END_DATE_DATA_DOWNLOAD, START_DATE_DATA_DOWNLOAD
from definitions.constants_V import SHARE_OUTSTANDING_DATA_250_TESTING_PKL_L, SHARES_OUTSTANDING_DATA_250_UPDATED_PKL_L, EPS_DATA_TESTING_PKL_L, EPS_DATA_UPDATED_PKL_L, DIVIDEND_DATA_TESTING_PKL_L, DIVIDEND_DATA_UPDATED_PKL_L
import yfinance as yf
import math
import pickle
from joblib import Parallel, delayed
from definitions.constants import N_JOBS
import numpy as np
# import definitions.constants as const
import matplotlib.pyplot as plt
import requests
import time
import copy

# --------------------------------------------------------------------
# Functions for downloading, processing, and analyzing financial data
# --------------------------------------------------------------------

def download_data(*, tickers=BOND_TICKERS, start_date=START_DATE_DATA_DOWNLOAD, end_date=(pd.to_datetime(END_DATE_DATA_DOWNLOAD) + timedelta(days=1)).strftime('%Y-%m-%d'), bonds_data_path_raw):
    """
    Download historical data for a list of tickers from Yahoo Finance.

    Args:
        tickers (list): List of ticker symbols to download data for.
        start_date (str): Start date for the data in 'YYYY-MM-DD' format.
        end_date (str): End date for the data in 'YYYY-MM-DD' format.

    Returns:
        None: Saves the data as a pickle file at BONDS_DATA_PATH_RAW.
    """
    data_frames = []

    for ticker in tickers:
        data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=False)
        data = data.xs(ticker, axis=1, level='Ticker')

        # Calculate Volume Weighted Average Price (VWAP)
        data['VWAP'] = (
            (data['Close'] * data['Volume']).groupby(data.index.date).cumsum() /
            data['Volume'].groupby(data.index.date).cumsum()
        )

        # Rename columns to include the ticker symbol
        data = data.rename(columns={
            'Open': f'{ticker}_Open',
            'High': f'{ticker}_High',
            'Low': f'{ticker}_Low',
            'Close': f'{ticker}_Close',
            'Adj Close': f'{ticker}_Adj_Close',
            'Volume': f'{ticker}_Volume',
            'VWAP': f'{ticker}_VWAP'
        })

        data_frames.append(data)

    # Save all data to a pickle file
    
    with open(bonds_data_path_raw, 'wb') as file:
        pickle.dump(data_frames, file)

def earning_data_downloader(tickers,api_key):
    
    stocks = tickers
    api_token = api_key
    
    
    # Dictionary to store EPS data for each stock
    eps_data_dict = {}

    for stock in stocks:
        ticker = f"{stock}.US"  # Add '.US' to the stock name for API
        url = f"https://eodhd.com/api/fundamentals/{ticker}?api_token={api_token}&fmt=json"
        try:
            response = requests.get(url)
            data = response.json()

            # Extract EPS history
            if "Earnings" in data and "History" in data["Earnings"]:
                eps_data = data["Earnings"]["History"]
                eps_records = [
                    {
                        "reportDate": record["reportDate"],
                        "epsActual": record["epsActual"],
                    }
                    for record in eps_data.values()
                ]

                # Convert to a DataFrame and store it
                eps_df = pd.DataFrame(eps_records).sort_values(by="reportDate", ascending=False)
                eps_data_dict[stock] = eps_df
        except Exception as e:
            print(f"Failed to fetch data for {stock}: {e}")
        
    with open(EPS_DATA_TESTING_PKL_L, 'wb') as file:
        pickle.dump(eps_data_dict, file)


    return eps_data_dict


def divident_data_downloader(tickers):

    dividend_data = {}

    for ticker in tickers:
        stock = yf.Ticker(ticker)
        try:
            dividends = stock.dividends  # Fetch historical dividend data
            dividends.index = dividends.index.tz_localize(None)
            dividend_data[ticker] = dividends  # Store in dictionary
            print(f"Fetched dividends for {ticker}")
        except Exception as e:
            print(f"Failed to fetch data for {ticker}: {e}")

    # Save the data to a pickle file
    with open(DIVIDEND_DATA_TESTING_PKL_L, 'wb') as file:
        pickle.dump(dividend_data, file)

    return dividend_data


def shares_outstanding_data_downloader(tickers,api_key):

        # Dictionary to store data for each ticker
    shares_data = {}
    for ticker in tickers:
        # Define the URL for fundamental data
        url = f'https://eodhistoricaldata.com/api/fundamentals/{ticker}.US'

        # Parameters
        params = {
            'api_token': api_key,
        }

        # Fetch data
        response = requests.get(url, params=params)

        # Check if response is valid
        if response.status_code == 200:
            data = response.json()

            # Extract quarterly shares outstanding data
            shares_outstanding_data = data.get('outstandingShares', {}).get('quarterly', {}).values()

            # Convert to DataFrame if data is available
            if shares_outstanding_data:
                df = pd.DataFrame(shares_outstanding_data)

                # Use 'dateFormatted' as date and 'shares' as Shares Outstanding
                df = df[['dateFormatted', 'shares']].rename(columns={'dateFormatted': 'Date', 'shares': 'Shares Outstanding'})

                # Convert 'Date' column to datetime format
                df['Date'] = pd.to_datetime(df['Date'])

                # Sort DataFrame by date
                df.sort_values(by='Date', inplace=True)

                # Reset index
                df.reset_index(drop=True, inplace=True)

                # Convert 'Shares Outstanding' to integer
                df['Shares Outstanding'] = df['Shares Outstanding'].astype(int)

                # Store the DataFrame in the dictionary
                shares_data[ticker] = df
    #             print(f"AHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH {ticker}")
            else:
                print(f"No shares outstanding data found for {ticker}")
        else:
            print(f"Failed to fetch data for {ticker}")

    # Save the data to a pickle file
    with open(SHARE_OUTSTANDING_DATA_250_TESTING_PKL_L, 'wb') as f:
        pickle.dump(shares_data, f)

    print("Data saved to 'shares_outstanding_data_250.pkl'")
    return shares_data

def equity_data_downloader(tickers,api_key):
    equity = {}
    # Loop through each stock to fetch and store its equity data
    for stock in tickers:
        print(f"Fetching data for {stock}...")
        # Construct the API URL for the stock
        url = f'https://eodhistoricaldata.com/api/fundamentals/{stock}.{exchange}?api_token={api_key}'
        
        # Make the API request
        response = requests.get(url)
        
        if response.status_code == 200:
            data = response.json()
            try:
                # Extract quarterly balance sheet data
                quarterly_data = data['Financials']['Balance_Sheet']['quarterly']
                equity_data = []
                
                # Process each quarterly entry
                for date, details in quarterly_data.items():
                    equity_value = details.get('totalStockholderEquity', 'N/A')
                    try:
                        # Convert equity to millions and ensure it’s an integer
                        equity_in_millions = int(float(equity_value) / 1000000)
                        equity_data.append({
                            'Date': date,
                            'Total Stockholder Equity (Millions)': equity_in_millions
                        })
                    except (TypeError, ValueError):
                        # Handle missing or invalid equity values
                        equity_data.append({
                            'Date': date,
                            'Total Stockholder Equity (Millions)': None
                        })
                
                # Create a DataFrame from the processed data
                df = pd.DataFrame(equity_data)
                # Convert 'Date' column to datetime format
                df['Date'] = pd.to_datetime(df['Date'])
                # Sort by date in ascending order
                df = df.sort_values('Date')
                df['Total Stockholder Equity (Millions)_shifted'] = df['Total Stockholder Equity (Millions)'].shift(1)
                # Store the DataFrame in the equity dictionary
                equity[stock] = df
                print(f"Data for {stock} successfully fetched and processed.")
            except KeyError as e:
                print(f"Error for {stock}: Could not find expected data structure. Key error: {e}")
        else:
            print(f"Failed to fetch data for {stock}. Status code: {response.status_code}")


def update_data(file_path, df, output_file):
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    
    for _, row in df.iterrows():
        alternative = row['Alternative/Status']
        original = row['Original Ticker']
        if original in data:
            data[alternative] = data.pop(original)
    
    with open(output_file, 'wb') as file:
        pickle.dump(data, file)


def update_dividend_data(file_path, df, output_file):
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    
    for _, row in df.iterrows():
        alternative = row['Alternative/Status']
        original = row['Original Ticker']
        if original in data:
            data[alternative] = data.pop(original)
    
    # Specific changes for dividend data
    data['FB'] = data.pop('META')
    data['META'] = data['FB']
    
    with open(output_file, 'wb') as file:
        pickle.dump(data, file)


def add_shift_columns_to_all(*, all_data):
    """
    Add shifted columns for Close and Open prices to all dataframes.

    Args:
        all_data (list): List of pandas DataFrames containing stock data.

    Returns:
        list: Updated list of DataFrames with shift columns added.
    """
    for df in all_data:
        prefixes = set(col.split('_')[0] for col in df.columns)
        for prefix in prefixes:
            if f"{prefix}_Open" in df.columns and f"{prefix}_Close" in df.columns:
                df.sort_index(inplace=True)
                df[f"{prefix}_Volume_shift"] = df[f"{prefix}_Volume"].shift(-1)
                df[f"{prefix}_Close_shift"] = df[f"{prefix}_Close"].shift(-1)
                df[f"{prefix}_Open_shift"] = df[f"{prefix}_Open"].shift(-1)
    return all_data

def turnovern(window_data, index,ticker):
    
    with open('/home/iyad/aliquidity/V1/updated_shares_outstanding_data_250.pkl', 'rb') as f:
        shares_data = pickle.load(f)
    ticker = ticker
    if ticker in shares_data:
        shar = shares_data[ticker]
        shar['Shares Outstanding'] = shar['Shares Outstanding'].astype(str).str[:-4].astype(int)

#         print(f"Shares Outstanding Data for {ticker}:")
#         print(df)
    else:
#         print(f"Shares Outstanding Data for {ticker} not found!")
        return 0  # Or any default value you want to return
     
    filtered_da = pd.DataFrame(window_data)
    filtered_da.reset_index(inplace=True)
    
    filtered_da['Date'] = pd.to_datetime(filtered_da['Date'])
    shar['period'] = pd.to_datetime(shar['Date'])
    
    shar = shar.sort_values('Date')
    
    daily_dates = pd.date_range(start=shar['Date'].min(), end=shar['Date'].max(), freq='D')
    daily_df = pd.DataFrame(daily_dates, columns=['Date'])
    # Merge and forward fill the 'Shares Outstanding' values
    daily_df = pd.merge(daily_df, shar, on='Date', how='left')
    daily_df['Shares Outstanding'] = daily_df['Shares Outstanding'].fillna(method='ffill')
    combined_df = pd.merge(daily_df,filtered_da,on='Date')
    combined_df

    num = len(window_data)
    if num == 0:
        # If the window data is empty, return 0
        return 0
    else:
        Volume = window_data
        arr = []
        
        for i in np.arange(num-1):
            # Ensure you are handling the case when no matching 'Date' is found
            turnov_row = combined_df[combined_df['Date'] == str(window_data.index[i].date())]
            
            if not turnov_row.empty:
                turnov = turnov_row['Shares Outstanding'].values[0]
            else:
                # Handle the case where no match is found
                print(f"No sharesOutstanding data found for date: {window_data.index[i].date()}")
                turnov = 0 # Default value to avoid division by zero

            # Proceed only if turnov is not zero
            if turnov != 0:
                val = Volume[i] / turnov
                arr = np.append(arr, val)
            else:
#                 print(f"Warning: sharesOutstanding is zero for date: {window_data.index[i].date()}")
                turnov = 0

        # Check if the sum of arr is greater than zero before applying math.log
        if sum(arr) > 0:
            result = math.log(sum(arr))
        else:
            result = 0  # Avoid log of zero or negative numbers
            
        return result

    
def turnovern3M(window_data3m, index,ticker):
    
#     print('turnover22')
    num_days_in_month = 30  # Approximate trading days in a month
    num_months = 3  # We are dealing with 3 months of data
    monthly_turnovers = []

    # Loop through each month and calculate turnover
    for month in range(num_months-1):
        start = month * num_days_in_month #0,30
        end = (month + 1) * num_days_in_month # 30,60
        val = math.exp(turnovern(window_data3m[start:end], index,ticker))  # Get turnover for each month
        monthly_turnovers.append(val)

    # Calculate the average turnover for the 3 months and return the logarithmic value
    result = math.log(sum(monthly_turnovers) / num_months)
    return result

def turnovern12M(window_data12m, index,ticker):
    print(3)
    num_days_in_month = 21  # Approximate number of trading days in a month
    num_months = 12  # We're working with 12 months of data
    monthly_turnovers = []

    # Loop through each month and calculate turnover
    for month in range(num_months):
        start = month * num_days_in_month
        end = (month + 1) * num_days_in_month
        # Extract the relevant window for the month and calculate turnover
        value = turnovern(window_data12m[start:end], index,ticker)
        monthly_turnovers.append(value)

    total_turnover = sum(monthly_turnovers)

    # Return 0 if the total turnover is zero or less to avoid math domain errors
    if total_turnover <= 0:
        print(f"Total turnover is zero or negative: {total_turnover}. Returning 0.")
        return 0  # Return 0 if total turnover is zero or negative

    # Calculate the average turnover for the 12 months and return the logarithmic value
    result = math.log(total_turnover / num_months)
    return result



def momentum(*, point, half_life):
    """
    Calculate momentum using logarithmic returns and exponential weighting.

    Args:
        point (pd.Series): Series of stock prices.
        half_life (int): Exponential decay parameter.

    Returns:
        float: Calculated momentum value.
    """
    arr2 = []
    for i in range(len(point) - 1):
        inter = (point.iloc[i + 1] / point.iloc[i] - 1)
        val = (math.log(inter + 1)) * math.exp(-4 + i / half_life)
        arr2.append(val)
    return sum(arr2)


def momentum_liquidity(window_data,window_data3m,window_data12m, index,ticker):
    a = turnovern(window_data, index,ticker)
    b = turnovern3M(window_data3m, index,ticker)
    c = turnovern12M(window_data12m, index,ticker)
    val = 0.35 * a + 0.35 * b + 0.3 * c
    return val



def calculate_monthly_momentum(*, df, close_col_name, open_col_name, close_col_name_shift, open_col_name_shift, window, half_life, number=0):
    """
    Calculate monthly momentum for a given DataFrame.

    Args:
        df (pd.DataFrame): Stock data DataFrame.
        close_col_name (str): Column name for Close prices.
        open_col_name (str): Column name for Open prices.
        close_col_name_shift (str): Column name for shifted Close prices.
        open_col_name_shift (str): Column name for shifted Open prices.
        window (int): Lookback window for momentum calculation.
        half_life (int): Half-life parameter for exponential weighting.

    Returns:
        pd.DataFrame: DataFrame containing momentum values and associated metrics.
    """
    tail = len(df)
    momentum_values = []

    while tail - window > 0:
        tail = tail - number

        last_date = df.iloc[tail - 1].name
        last_close = df[close_col_name].iloc[tail - 1]
        last_open = df[open_col_name].iloc[tail - 1]
        last_close_shift = df[close_col_name_shift].iloc[tail - 1]
        last_open_shift = df[open_col_name_shift].iloc[tail - 1]

        window_data = df[close_col_name].iloc[tail - window: tail]
        mom_value = momentum(point = window_data, half_life = half_life)
        momentum_values.append((last_date, mom_value, last_close, last_open, last_close_shift, last_open_shift))

        tail -= 1
        current_month = df.iloc[tail].name.month
        while df.iloc[tail - 1].name.month == current_month:
            tail -= 1

    return pd.DataFrame(momentum_values, columns=['Date', f'Momentum_{window}_{half_life}', 'Close', 'Open', 'Close_shift', 'Open_shift'])


def calculate_monthly_liquidity(*, df, close_col_name,open_col_name,close_col_name_shift,open_col_name_shift,Volume_col_name,volume_col_name_shift, window, half_life,ticker, number=0):
    """
    Calculate monthly liquidity for a given DataFrame.

    Args:
        df (pd.DataFrame): Stock data DataFrame.

"""
        
    tail = len(df)
    momentum_values = []
    index = 0
    window3m = window+60
    window12m = window+330

    while tail - window > 0:
        tail = tail - number
        last_date = df.iloc[tail - 1].name
        last_close = df[Volume_col_name].iloc[tail - 1]
        last_open = df[open_col_name].iloc[tail - 1]
        last_close_shift = df[volume_col_name_shift].iloc[tail - 1]
        last_open_shift = df[open_col_name_shift].iloc[tail - 1]
#         print('last_date',last_date)
#         print('tail,window',tail,window)
        window_data = df[Volume_col_name].iloc[tail - window: tail]
        window_data3m = df[close_col_name].iloc[tail - window3m: tail]
        window_data12m = df[close_col_name].iloc[tail - window12m: tail]

#         print('df', df)
#         print('window_data',window_data)
        mom_value = momentum_liquidity(window_data,window_data3m,window_data12m, index,ticker)
#         print('mom_value',mom_value)
        momentum_values.append((last_date, mom_value,last_close,last_open,last_close_shift,last_open_shift))
#         print('momentum_values',momentum_values)
        tail -= 1
        current_month = df.iloc[tail].name.month
#         print(momentum_values)
        while df.iloc[tail - 1].name.month == current_month:
            tail -= 1

    return pd.DataFrame(momentum_values, columns=['Date', f'Momentum_{window}_{half_life}','Close','Open','Close_shift','Open_shift'])




def process_single_dataframe(*, df, momentum_windows, half_lives, number=0):
    """
    Process a single DataFrame to calculate momentum metrics.

    Args:
        df (pd.DataFrame): Stock data DataFrame.
        momentum_windows (list): List of momentum windows to use.
        half_lives (list): List of half-lives to use.

    Returns:
        pd.DataFrame: DataFrame with combined momentum metrics.
    """
    ticker = [col.split('_')[0] for col in df.columns if '_Close' in col][0]
    close_col_name = f"{ticker}_Close"
    open_col_name = f"{ticker}_Open"
    close_col_name_shift = f"{ticker}_Close_shift"
    open_col_name_shift = f"{ticker}_Open_shift"

    momentum_dfs = []
    counter = 0
    for window in momentum_windows:
        for half_life in half_lives:
            if counter == 0:
                momentum_df = calculate_monthly_momentum(
                    df = df, close_col_name=close_col_name, open_col_name=open_col_name, close_col_name_shift=close_col_name_shift, open_col_name_shift=open_col_name_shift, window=window, half_life=half_life, number=number
                )
                momentum_df.set_index('Date', inplace=True)
                momentum_dfs.append(momentum_df)
            else:
                momentum_df = calculate_monthly_momentum(
                    df = df, close_col_name = close_col_name, open_col_name = open_col_name, close_col_name_shift = close_col_name_shift, open_col_name_shift = open_col_name_shift, window = window, half_life = half_life, number = number
                )
                momentum_df.set_index('Date', inplace=True)
                momentum_dfs.append(momentum_df[[f'Momentum_{window}_{half_life}']])
            counter += 1

    combined_df = pd.concat(momentum_dfs, axis=1)
    combined_df['Stock'] = ticker
    combined_df.sort_index(inplace=True)
    return combined_df

def resid_vol(dat, half_life, num):
    arr = []
    for i in np.arange(num-1):   #num==252
        val = (dat[i+1]/dat[i] - 1) * math.exp(-num/half_life + i/half_life) * 100
        arr = np.append(arr, val)
    return arr

def vol(*, dat, half_life, mult, weight):
    """
    Calculate vol using logarithmic returns and exponential weighting.

    Args:
        point (pd.Series): Series of stock prices.
        half_life (int): Exponential decay parameter.

    Returns:
        float: Calculated vol value.
    """
    try:
        num = len(dat)
        vol_42 = resid_vol(dat, half_life, num)
        vol_63 = resid_vol(dat, half_life*mult, num)

        vol_42_p = vol_42[vol_42>0]
        vol_63_p = vol_63[vol_63>0]

        vol_42_n = vol_42[vol_42<0]
        vol_63_n = vol_63[vol_63<0]

        sigma1 = ((np.std(vol_42_p))-(np.std(vol_42_n)))
        sigma2 = ((np.std(vol_63_p))-(np.std(vol_63_n)))
        score = weight * sigma1 + (1-weight) * sigma2  #resiid --> calculate score
        return score
    except:
        return -1000

def calculate_monthly_vol(*, df, close_col_name, open_col_name, close_col_name_shift, open_col_name_shift, window, half_life, mult, weight, number=0):
    """
    Calculate monthly vol for a given DataFrame.

    Args:
        df (pd.DataFrame): Stock data DataFrame.
        close_col_name (str): Column name for Close prices.
        open_col_name (str): Column name for Open prices.
        close_col_name_shift (str): Column name for shifted Close prices.
        open_col_name_shift (str): Column name for shifted Open prices.
        window (int): Lookback window for vol calculation.
        half_life (int): Half-life parameter for exponential weighting.

    Returns:
        pd.DataFrame: DataFrame containing vol values and associated metrics.
    """
    tail = len(df)
    momentum_values = []

    while tail - window > 0:
        tail = tail - number

        last_date = df.iloc[tail - 1].name
        last_close = df[close_col_name].iloc[tail - 1]
        last_open = df[open_col_name].iloc[tail - 1]
        last_close_shift = df[close_col_name_shift].iloc[tail - 1]
        last_open_shift = df[open_col_name_shift].iloc[tail - 1]

        window_data = df[close_col_name].iloc[tail - window: tail]
        mom_value = vol(dat = window_data, half_life = half_life, mult = mult, weight = weight)
        momentum_values.append((last_date, mom_value, last_close, last_open, last_close_shift, last_open_shift))

        tail -= 1
        current_month = df.iloc[tail].name.month
        while df.iloc[tail - 1].name.month == current_month:
            tail -= 1

    return pd.DataFrame(momentum_values, columns=['Date', f'Momentum_{window}_{half_life}_{mult}_{weight}', 'Close', 'Open', 'Close_shift', 'Open_shift'])

def process_single_dataframe_V(*, df, momentum_windows, half_lives, mult, weight, number=0):
    """
    Process a single DataFrame to calculate momentum metrics.

    Args:
        df (pd.DataFrame): Stock data DataFrame.
        momentum_windows (list): List of momentum windows to use.
        half_lives (list): List of half-lives to use.

    Returns:
        pd.DataFrame: DataFrame with combined momentum metrics.
    """
    ticker = [col.split('_')[0] for col in df.columns if '_Close' in col][0]
    close_col_name = f"{ticker}_Close"
    open_col_name = f"{ticker}_Open"
    close_col_name_shift = f"{ticker}_Close_shift"
    open_col_name_shift = f"{ticker}_Open_shift"

    momentum_dfs = []
    counter = 0
    for window in momentum_windows:
        for half_life in half_lives:
            for mu in mult:
                for w in weight:
                    if counter == 0:
                        momentum_df = calculate_monthly_vol(
                            df = df, close_col_name=close_col_name, open_col_name=open_col_name, close_col_name_shift=close_col_name_shift, open_col_name_shift=open_col_name_shift, window=window, half_life=half_life, mult=mu, weight=w, number=number
                        )
                        momentum_df.set_index('Date', inplace=True)
                        momentum_dfs.append(momentum_df)
                    else:
                        momentum_df = calculate_monthly_vol(
                            df = df, close_col_name = close_col_name, open_col_name = open_col_name, close_col_name_shift = close_col_name_shift, open_col_name_shift = open_col_name_shift, window = window, half_life = half_life, mult=mu, weight=w, number = number
                        )
                        momentum_df.set_index('Date', inplace=True)
                        momentum_dfs.append(momentum_df[[f'Momentum_{window}_{half_life}_{mu}_{w}']])
                    counter += 1

    combined_df = pd.concat(momentum_dfs, axis=1)
    combined_df['Stock'] = ticker
    combined_df.sort_index(inplace=True)
    return combined_df




def process_single_dataframe_L(df, momentum_windows, half_lives, mult, weight, number=0):
    """
    Process a single DataFrame to calculate momentum metrics.

    Args:
        df (pd.DataFrame): Stock data DataFrame.
        momentum_windows (list): List of momentum windows to use.
        half_lives (list): List of half-lives to use.

    Returns:
        pd.DataFrame: DataFrame with combined momentum metrics.
    """
    ticker = [col.split('_')[0] for col in df.columns if '_Close' in col][0]
    close_col_name = f"{ticker}_Close"
    open_col_name = f"{ticker}_Open"
    Volume_col_name = f"{ticker}_Volume"
    
    close_col_name_shift = f"{ticker}_Close_shift"
    open_col_name_shift = f"{ticker}_Open_shift"
    volume_col_name_shift = f"{ticker}_Volume_shift"
    
    momentum_dfs = []
    counter = 0
    for window in momentum_windows:
        for half_life in half_lives:
            if counter ==0:
                momentum_df = calculate_monthly_liquidity(df = df, close_col_name = close_col_name,open_col_name = open_col_name,close_col_name_shift = close_col_name_shift,
                                                       open_col_name_shift = open_col_name_shift,Volume_col_name = Volume_col_name,volume_col_name_shift = volume_col_name_shift, window = window, half_life = half_life,ticker = ticker, number = number)
                momentum_df.set_index('Date', inplace=True)
                momentum_dfs.append(momentum_df)
            else:
                momentum_df = calculate_monthly_liquidity(df = df, close_col_name = close_col_name,open_col_name = open_col_name,close_col_name_shift = close_col_name_shift,
                                                       open_col_name_shift = open_col_name_shift,Volume_col_name = Volume_col_name,volume_col_name_shift = volume_col_name_shift, window = window, half_life = half_life,ticker = ticker, number = number)
                momentum_df.set_index('Date', inplace=True)
                momentum_dfs.append(momentum_df[[f'Momentum_{window}_{half_life}']])
            counter = counter+1
                
                
                
    combined_df = pd.concat(momentum_dfs, axis=1)
    combined_df['Stock'] = ticker
    combined_df.sort_index(inplace=True)
    return combined_df


def process_single_dataframe_E(df, number=0):
    """
    Process a single DataFrame to calculate momentum metrics.

    Args:
        df (pd.DataFrame): Stock data DataFrame.
        momentum_windows (list): List of momentum windows to use.
        half_lives (list): List of half-lives to use.

    Returns:
        pd.DataFrame: DataFrame with combined momentum metrics.
    """


    with open(DIVIDEND_DATA_UPDATED_PKL_L, 'rb') as file:
        divident = pickle.load(file)
    with open(EPS_DATA_UPDATED_PKL_L, 'rb') as file:
        eps = pickle.load(file)
    with open(SHARES_OUTSTANDING_DATA_250_UPDATED_PKL_L, 'rb') as file:
        shares = pickle.load(file)


    ticker = [col.split('_')[0] for col in df.columns if '_Close' in col][0]
    print(ticker)
    if ticker in shares:
        print('here2')
        print(ticker)
        shar = shares[ticker]
        eq = equity[ticker]
        
        shar['Shares Outstanding'] = shar['Shares Outstanding'].astype(str).str[:-4].astype(int)

        filtered_da = df
        filtered_da.reset_index(inplace=True)


        filtered_da['Date'] = pd.to_datetime(filtered_da['Date'])
        shar['Date'] = pd.to_datetime(shar['Date'])
        eq['Date'] = pd.to_datetime(eq['Date'])


        shar = shar.sort_values('Date')
        eq = eq.sort_values('Date')

        daily_dates_shares = pd.date_range(start=shar['Date'].min(), end=shar['Date'].max(), freq='D')

        daily_df_shares = pd.DataFrame(daily_dates_shares, columns=['Date'])

        daily_shares = pd.merge(daily_df_shares, shar, on='Date', how='left')
#         daily_divident = pd.merge(daily_df_shares, div, on='Date', how='left')
        daily_equity = pd.merge(daily_df_shares, eq, on='Date', how='left')
        
        daily_shares['Shares Outstanding'] = daily_shares['Shares Outstanding'].fillna(method='ffill')
#         daily_divident['Dividends'] = daily_divident['Dividends'].fillna(method='ffill')
        daily_equity['Total Stockholder Equity (Millions)_shifted'] = daily_equity['Total Stockholder Equity (Millions)_shifted'].fillna(method='ffill')

        
        #     print(filtered_da)
    #     print('*************')
    #     print(daily_df_shares)
        combined_df = pd.merge(daily_shares,filtered_da,on='Date')
        combined_df = pd.merge(daily_equity,combined_df,on='Date')

        a = combined_df
        a['market cap'] = a['Shares Outstanding'] * a[f'{ticker}_Close']
        a['book_to_price'] = a['Total Stockholder Equity (Millions)_shifted']/a['market cap']

        
        a['Stock'] = ticker
        print(a)
        a = a[['Date','Stock','book_to_price',f'{ticker}_Open_shift']]

        a.rename(columns={f'{ticker}_Open_shift': 'Open_shift'}, inplace=True)
        return a




def paralelizer(*, data, year, BOND_TICKERS, LEN_YEARS_DV_LOOKBACK, DV_QUANTILE_THRESHOLD):
    """
    Parallelize the processing of data for a given year.

    Args:
        data (pd.DataFrame): Stock data DataFrame.
        year (int): Year for which to process the data.

    Returns:
        dict: Ticker and its corresponding median daily volume.
    """
    ticker = data.columns[0].split('_')[0]
    if ticker in BOND_TICKERS:
        return {f'{ticker}': 0}

    data = data[(data.index.year <= year) & (data.index.year >= year - (LEN_YEARS_DV_LOOKBACK - 1))]
    data.sort_index(inplace=True)

    if len(data) <= 245 * LEN_YEARS_DV_LOOKBACK:
        return {f'{ticker}': 0}

    try:
        data = data[data.index.year == year]
        data.reset_index(inplace=True)
        DV = data[f'{ticker}_vw'] * data[f'{ticker}_Volume']
        medianDailyVolume = DV.quantile(DV_QUANTILE_THRESHOLD)
        return {f'{ticker}': medianDailyVolume}
    except:
        return {f'{ticker}': 0}


def stock_selector(*, all_data, yearStocks, YEARS, BOND_TICKERS, LEN_YEARS_DV_LOOKBACK, DV_QUANTILE_THRESHOLD, N_JOBS, SELECTED_TOP_VOL_STOCKS, FOR_LIVE = False, YEARSTOCKS_PATH_LIVE, YEARSTOCKS_PATH):
    """
    Select stocks based on their median daily volume for each year.

    Args:
        all_data (list): List of DataFrames containing stock data.
        yearStocks (dict): Dictionary to store selected stocks by year.
        YEARS (list): List of years to process.

    Returns:
        dict: Updated yearStocks with selected stocks.
    """

    stockdict1 = {}
    for year in YEARS:
        fitnesses = Parallel(n_jobs=N_JOBS)(delayed(paralelizer)(data = i, year = year, BOND_TICKERS = BOND_TICKERS, LEN_YEARS_DV_LOOKBACK=LEN_YEARS_DV_LOOKBACK, DV_QUANTILE_THRESHOLD=DV_QUANTILE_THRESHOLD) for i in all_data)
        keys = []
        values = []

        for item in fitnesses:
            key, value = item.popitem()
            keys.append(key)
            values.append(value)

        resDf = pd.DataFrame({'Key': keys, 'Value': values})
        resDf = resDf[resDf['Value'] > 0]

        sorted_stocklist = resDf.sort_values('Value', ascending=False).iloc[:SELECTED_TOP_VOL_STOCKS, :]

        stockdict1[year] = sorted_stocklist

    for i in list(stockdict1.keys()):
        if i >= 2005:
            for idx, row in stockdict1[i].iterrows():
                yearStocks[i + 1].append(row['Key'])

    if FOR_LIVE:
        pickle.dump(yearStocks, open(YEARSTOCKS_PATH_LIVE, 'wb'))
    else:
        pickle.dump(yearStocks, open(YEARSTOCKS_PATH, 'wb'))

    return yearStocks

def stock_selector_sensitivity(*, all_data, yearStocks, YEARS, BOND_TICKERS, LEN_YEARS_DV_LOOKBACK, DV_QUANTILE_THRESHOLD, N_JOBS, SELECTED_TOP_VOL_STOCKS, YEARSTOCKS_PATH):
    """
    Select stocks based on their median daily volume for each year.

    Args:
        all_data (list): List of DataFrames containing stock data.
        yearStocks (dict): Dictionary to store selected stocks by year.
        YEARS (list): List of years to process.

    Returns:
        dict: Updated yearStocks with selected stocks.
    """
    results = {}  # To store precomputed results for each dv_quantile
    
    # Process for each DV_QUANTILE_THRESHOLD
    for dv_quantile in DV_QUANTILE_THRESHOLD:

        stockdict1 = {}
        for year in YEARS:
            # Parallel processing for fitnesses
            fitnesses = Parallel(n_jobs=N_JOBS)(
                delayed(paralelizer)(data=i, year=year, BOND_TICKERS=BOND_TICKERS, LEN_YEARS_DV_LOOKBACK=LEN_YEARS_DV_LOOKBACK, DV_QUANTILE_THRESHOLD=dv_quantile) for i in all_data
            )
            
            # Extract keys and values
            keys = []
            values = []
            for item in fitnesses:
                key, value = item.popitem()
                keys.append(key)
                values.append(value)

            # Create DataFrame and filter by value > 0
            resDf = pd.DataFrame({'Key': keys, 'Value': values})
            resDf = resDf[resDf['Value'] > 0]

            # Store the filtered DataFrame for the year
            stockdict1[year] = resDf

        # Store the full result for this dv_quantile
        results[dv_quantile] = stockdict1

    # Generate files for each SELECTED_TOP_VOL_STOCKS
    for dv_quantile, stockdict1 in results.items():
        for top_stocks in SELECTED_TOP_VOL_STOCKS:
            yearStockstemp = copy.deepcopy(yearStocks)

            for i in list(stockdict1.keys()):
                if i >= 2005:
                    sorted_stocklist = stockdict1[i].sort_values('Value', ascending=False).iloc[:top_stocks, :]
                    for idx, row in sorted_stocklist.iterrows():
                        yearStockstemp[i + 1].append(row['Key'])

            # Save yearStockstemp for this combination
            file_suffix = f"/stockstobeused1_dv_{dv_quantile}_top_{top_stocks}.pkl"
            pickle.dump(yearStockstemp, open(YEARSTOCKS_PATH+file_suffix, 'wb'))

def exponential_weights(*, length, alpha=0.85):
    """
    Generate exponential weights for a given length.

    Args:
        length (int): Number of weights to generate.
        alpha (float): Decay factor for the weights, typically between 0 and 1.

    Returns:
        np.array: Array of weights, where weights exponentially decrease.
    """
    indices = np.arange(length)
    weights = alpha ** indices
    weights /= weights.sum()  # Normalize weights to sum to 1
    return weights[::-1]  # Return weights in reverse order

def load_and_preprocess_data_M(*, file1, file2, FOR_LIVE=False):
    """
    Load and preprocess data from two CSV files, concatenate them, and save the result.

    Args:
        file1 (str): Path to the first CSV file.
        file2 (str): Path to the second CSV file.

    Returns:
        pd.DataFrame: Concatenated DataFrame after preprocessing.
    """
    if FOR_LIVE:
        data1 = pd.read_csv(file1)  # Drop rows with missing values
        data2 = pd.read_csv(file2)
    else:
        data1 = pd.read_csv(file1).dropna(axis=0)  # Drop rows with missing values
        data2 = pd.read_csv(file2).dropna(axis=0)

    concatenated = pd.concat([data1, data2])  # Concatenate both DataFrames
    return concatenated

def load_and_preprocess_data(*, file2, FOR_LIVE=False):
    """
    Load and preprocess data from two CSV files, concatenate them, and save the result.

    Args:
        file1 (str): Path to the first CSV file.
        file2 (str): Path to the second CSV file.

    Returns:
        pd.DataFrame: Concatenated DataFrame after preprocessing.
    """
    if FOR_LIVE:  # Drop rows with missing values
        data2 = pd.read_csv(file2)
    else:
        data2 = pd.read_csv(file2).dropna(axis=0)

    concatenated = pd.concat([data2])  # Concatenate both DataFrames

    return concatenated


def calculate_metrics(*, returns):
    """
    Calculate performance metrics including annual return, Sharpe ratio, 
    max drawdown, Calmar ratio, and Sortino ratio.

    Args:
        returns (pd.Series): Series of monthly returns.

    Returns:
        tuple: Annual return, Sharpe ratio, Max drawdown, Calmar ratio, Sortino ratio.
    """
    monthly_returns = [x / 100 for x in returns.values]  # Convert to decimals

    # Calculate annual return
    annual_return = np.prod([1 + r for r in monthly_returns]) ** (12 / len(monthly_returns)) - 1

    # Calculate Sharpe ratio
    sharpe_ratio = np.mean(monthly_returns) / np.std(monthly_returns) * np.sqrt(12)

    def max_drawdown(*, returns):
        """Helper function to calculate max drawdown."""
        cumulative_returns = np.cumprod([1 + r for r in returns])
        peak = np.maximum.accumulate(cumulative_returns)
        drawdown = (peak - cumulative_returns) / peak
        return np.max(drawdown)

    max_drawdown_value = max_drawdown(returns=monthly_returns)

    # Calculate Calmar ratio
    calmar_ratio = annual_return / max_drawdown_value if max_drawdown_value != 0 else float('inf')

    # Calculate Sortino ratio
    negative_returns = [r for r in monthly_returns if r < 0]
    downside_std = np.std(negative_returns) if negative_returns else 0
    sortino_ratio = np.mean(monthly_returns) / downside_std * np.sqrt(12) if downside_std != 0 else float('inf')

    return annual_return, sharpe_ratio, max_drawdown_value, calmar_ratio, sortino_ratio


def calculate_stock_selection(*, df, SELECTED_MOM_WINDOW=252, SELECTED_HALF_LIFE_WINDOW=252, SELECTED_N_STOCK_POSITIVE=3, SELECTED_N_STOCK_CHOSE=1):
    """
    Select stocks based on their momentum values.

    Args:
        df (pd.DataFrame): DataFrame with stock data and momentum metrics.
        SELECTED_MOM_WINDOW (int): Selected momentum window.
        SELECTED_HALF_LIFE_WINDOW (int): Selected half-life window.
        SELECTED_N_STOCK_POSITIVE (int): Minimum number of stocks with positive momentum.
        SELECTED_N_STOCK_CHOSE (int): Number of top stocks to select.

    Returns:
        dict: Dictionary with dates as keys and selected stocks as values.
    """
    dt = np.array(df[df.Stock == 'SPY'].Date)
    stock_dict = {}

    for i in dt:
        tmp = df[df.Date == i].copy()
        sorted_tmp = tmp.sort_values(f'Momentum_{SELECTED_MOM_WINDOW}_{SELECTED_HALF_LIFE_WINDOW}', ascending=False).drop_duplicates()
        positive_momentum_stocks = sorted_tmp[sorted_tmp[f'Momentum_{SELECTED_MOM_WINDOW}_{SELECTED_HALF_LIFE_WINDOW}'] > 0]

        # Select top stocks with positive momentum
        stock_dict[i] = positive_momentum_stocks.head(SELECTED_N_STOCK_CHOSE).Stock.values if len(positive_momentum_stocks) >= SELECTED_N_STOCK_POSITIVE else np.array([])

    return stock_dict

def calculate_stock_selection_L(*, df, df_M, SELECTED_N_STOCK_POSITIVE=3, SELECTED_N_STOCK_CHOSE=1, SELECTED_MOM_WINDOW_M=252, SELECTED_HALF_LIFE_WINDOW_M=252, SELECTED_N_STOCK_POSITIVE_M=3):
    """
    Select stocks based on their momentum values.

    Args:
        df (pd.DataFrame): DataFrame with stock data and momentum metrics.
        SELECTED_MOM_WINDOW (int): Selected momentum window.
        SELECTED_HALF_LIFE_WINDOW (int): Selected half-life window.
        SELECTED_N_STOCK_POSITIVE (int): Minimum number of stocks with positive momentum.
        SELECTED_N_STOCK_CHOSE (int): Number of top stocks to select.

    Returns:
        dict: Dictionary with dates as keys and selected stocks as values.
    """
    # dt = np.array(df[df.Stock == 'SPY'].Date)
    dt = df.Date.unique()
    stock_dict = {}

    for i in dt:
        tmp = df[df.Date == i].copy()
        sorted_tmp = tmp.sort_values(f'earnings_yield', ascending=False).drop_duplicates()
        positive_momentum_stocks = sorted_tmp[sorted_tmp['earnings_yield'] > 0]
        tmp_M = df_M[df_M.Date == i].copy()
        sorted_tmp_M = tmp_M.sort_values(f'Momentum_{SELECTED_MOM_WINDOW_M}_{SELECTED_HALF_LIFE_WINDOW_M}', ascending=False).drop_duplicates()
        positive_momentum_stocks_M = sorted_tmp_M[sorted_tmp_M[f'Momentum_{SELECTED_MOM_WINDOW_M}_{SELECTED_HALF_LIFE_WINDOW_M}'] > 0]
        # Select top stocks with positive momentum
        stock_dict[i] = positive_momentum_stocks.head(SELECTED_N_STOCK_CHOSE).Stock.values if ((len(positive_momentum_stocks) >= SELECTED_N_STOCK_POSITIVE) and (len(positive_momentum_stocks_M) >= SELECTED_N_STOCK_POSITIVE_M)) else np.array([])

    return stock_dict


def calculate_stock_selection_V(*, df, df_M, SELECTED_MOM_WINDOW=252, SELECTED_HALF_LIFE_WINDOW=252, SELECTED_MULT=1.01, SELECTED_WEIGHT=0.9, SELECTED_N_STOCK_POSITIVE=3, SELECTED_N_STOCK_CHOSE=1, SELECTED_MOM_WINDOW_M=252, SELECTED_HALF_LIFE_WINDOW_M=252, SELECTED_N_STOCK_POSITIVE_M=3):
    """
    Select stocks based on their momentum values.

    Args:
        df (pd.DataFrame): DataFrame with stock data and momentum metrics.
        SELECTED_MOM_WINDOW (int): Selected momentum window.
        SELECTED_HALF_LIFE_WINDOW (int): Selected half-life window.
        SELECTED_N_STOCK_POSITIVE (int): Minimum number of stocks with positive momentum.
        SELECTED_N_STOCK_CHOSE (int): Number of top stocks to select.

    Returns:
        dict: Dictionary with dates as keys and selected stocks as values.
    """
    dt = np.array(df[df.Stock == 'SPY'].Date)
    stock_dict = {}

    for i in dt:
        tmp = df[df.Date == i].copy()
        sorted_tmp = tmp.sort_values(f'Momentum_{SELECTED_MOM_WINDOW}_{SELECTED_HALF_LIFE_WINDOW}_{SELECTED_MULT}_{SELECTED_WEIGHT}', ascending=False).drop_duplicates()
        positive_momentum_stocks = sorted_tmp[sorted_tmp[f'Momentum_{SELECTED_MOM_WINDOW}_{SELECTED_HALF_LIFE_WINDOW}_{SELECTED_MULT}_{SELECTED_WEIGHT}'] > 0]

        tmp_M = df_M[df_M.Date == i].copy()
        sorted_tmp_M = tmp_M.sort_values(f'Momentum_{SELECTED_MOM_WINDOW_M}_{SELECTED_HALF_LIFE_WINDOW_M}', ascending=False).drop_duplicates()
        positive_momentum_stocks_M = sorted_tmp_M[sorted_tmp_M[f'Momentum_{SELECTED_MOM_WINDOW_M}_{SELECTED_HALF_LIFE_WINDOW_M}'] > 0]

        # Select top stocks with positive momentum
        stock_dict[i] = positive_momentum_stocks.head(SELECTED_N_STOCK_CHOSE).Stock.values if ((len(positive_momentum_stocks) >= SELECTED_N_STOCK_POSITIVE) and (len(positive_momentum_stocks_M) >= SELECTED_N_STOCK_POSITIVE_M)) else np.array([])

    return stock_dict


def calculate_returns(*, stock_dict, df, weights, mom, half):
    """
    Calculate portfolio returns based on selected stocks and weights.

    Args:
        stock_dict (dict): Dictionary with dates as keys and selected stocks as values.
        df (pd.DataFrame): DataFrame containing stock data.
        weights (list): List of weights for the selected stocks.

    Returns:
        pd.Series: Series of portfolio returns.
    """
    returns = {}
    for date, stocks in stock_dict.items():
        tmp = df[df.Date == date].copy()
        tmp = tmp[tmp['Stock'].isin(stocks)]
        tmp = tmp.sort_values(by=f'Momentum_{mom}_{half}', ascending=True)

        if len(tmp) == 0:
            returns[date] = 0
        else:
            tmp['weights'] = weights
            tmp['w_rets'] = tmp['weights'] * tmp['Returns']
            returns[date] = tmp['w_rets'].sum()

    return pd.Series(returns) * 100

def calculate_returns_L(*, stock_dict, df, weights,):
    """
    Calculate portfolio returns based on selected stocks and weights.

    Args:
        stock_dict (dict): Dictionary with dates as keys and selected stocks as values.
        df (pd.DataFrame): DataFrame containing stock data.
        weights (list): List of weights for the selected stocks.

    Returns:
        pd.Series: Series of portfolio returns.
    """
    returns = {}
    for date, stocks in stock_dict.items():
        tmp = df[df.Date == date].copy()
        tmp = tmp[tmp['Stock'].isin(stocks)]
        tmp = tmp.sort_values(by='earnings_yield', ascending=True)

        if len(tmp) == 0:
            returns[date] = 0
        else:
            tmp['weights'] = weights
            tmp['w_rets'] = tmp['weights'] * tmp['Returns']
            returns[date] = tmp['w_rets'].sum()

    return pd.Series(returns) * 100


def calculate_returns_V(*, stock_dict, df, weights, mom, half, mult, w):
    """
    Calculate portfolio returns based on selected stocks and weights.

    Args:
        stock_dict (dict): Dictionary with dates as keys and selected stocks as values.
        df (pd.DataFrame): DataFrame containing stock data.
        weights (list): List of weights for the selected stocks.

    Returns:
        pd.Series: Series of portfolio returns.
    """
    returns = {}
    for date, stocks in stock_dict.items():
        tmp = df[df.Date == date].copy()
        tmp = tmp[tmp['Stock'].isin(stocks)]
        tmp = tmp.sort_values(by=f'Momentum_{mom}_{half}_{mult}_{w}', ascending=True)

        if len(tmp) == 0:
            returns[date] = 0
        else:
            tmp['weights'] = weights
            tmp['w_rets'] = tmp['weights'] * tmp['Returns']
            returns[date] = tmp['w_rets'].sum()

    return pd.Series(returns) * 100


def makeFinalDf(*, parallel_results,number=0):
    """
    Combine parallel processing results into a final DataFrame and calculate returns.

    Args:
        parallel_results (list): List of DataFrames from parallel processing.

    Returns:
        pd.DataFrame: Final combined DataFrame with returns calculated.
    """
    all_results = []

    # Combine results
    for momentum_df in parallel_results:
        all_results.append(momentum_df)

    final_df = pd.concat(all_results)
    final_df.reset_index(inplace=True)
    
    number=number+1
    df = final_df
    df['Date'] = pd.to_datetime(df['Date'])
    df = df[df['Date']<"2023-08-01"]

    # Group by month and get unique dates
    df['YearMonth'] = df['Date'].dt.to_period('M')
    second_last_dates = df.groupby('YearMonth')['Date'].unique().apply(lambda x: sorted(x)[-number])
    # # print(second_last_dates)

    # Filter the DataFrame to get all stocks on second last trading days
    second_last_day_df = df[df['Date'].isin(second_last_dates)]
    df = second_last_day_df.sort_values('Date')
    df.dropna(axis=0,inplace=True)

    


    # Calculate returns
    lstt = []
    for stock in df.Stock.unique():
        temp_df = df[df['Stock'] == stock]
        temp_df.sort_values('Date', inplace=True)
        temp_df['Returns'] = temp_df.Open_shift.pct_change()
        temp_df['Returns'] = temp_df['Returns'].shift(-1)
        lstt.append(temp_df)

    final_df = pd.concat(lstt)

    final_dff = final_df[final_df['Date']<'2024-12-01']
    return final_dff


def makeCorrectedDf(*, final_df, selected_stocks, FOR_LIVE=False):
    """
    Correct the final DataFrame to include only selected stocks for each year.

    Args:
        final_df (pd.DataFrame): Final DataFrame with all stocks and metrics.
        selected_stocks (dict): Dictionary of selected stocks by year.

    Returns:
        pd.DataFrame: Corrected DataFrame with only selected stocks.
    """
    corrected_stocks_df = pd.DataFrame()

    for year, stock_list in selected_stocks.items():
        # Filter for selected stocks and add the year column
        yearly_data = final_df[(final_df['Stock'].isin(stock_list)) & (final_df['Date'].dt.year == year)]
        yearly_data['Year'] = year
        corrected_stocks_df = pd.concat([corrected_stocks_df, yearly_data])

    corrected_stocks_df.reset_index(drop=True, inplace=True)
    corrected_stocks_df['Date'] = pd.to_datetime(corrected_stocks_df['Date'])
    if FOR_LIVE:
        pass
    else:
        corrected_stocks_df.dropna(axis=0, inplace=True)
    corrected_stocks_df = corrected_stocks_df.sort_values('Date')

    return corrected_stocks_df


def plot_returns(*, returns, diffRebal=False):
    """
    Plot cumulative returns over time.

    Args:
        returns (pd.Series): Series of portfolio returns.

    Returns:
        None: Displays the cumulative returns plot.
    """
    cumr = (((returns / 100) + 1).cumprod())
    cumr.index = pd.to_datetime(cumr.index)

    if diffRebal:
        plt.figure(figsize=(6, 5))
    else:
        plt.figure(figsize=(18, 16))

    # Plot cumulative returns on a logarithmic scale
    plt.plot(np.log(cumr), marker='o')
    plt.title('Cumulative Returns Over Time')
    plt.xlabel('Months')
    plt.ylabel('Cumulative Returns')
    plt.grid(True)
    plt.show()


class TickerAPI:
    """
    A class to interact with the Polygon.io API for fetching stock tickers.
    """

    def __init__(self, *, api_key):
        self.api_key = api_key

    def get_stock_tickers(self, *, start_date):
        """
        Fetches stock tickers from the Polygon.io API.

        Args:
            start_date (str): The start date to filter active stocks.

        Returns:
            list: A list of unique stock tickers.
        """
        tickers = []
        base_url = 'https://api.polygon.io/v3/reference/tickers'
        params = {
            "market": "stocks",
            "date": start_date,
            "active": "true",
            "sort": "ticker",
            "order": "asc",
            "limit": 1000,
            "apiKey": self.api_key
        }

        while base_url:
            response = requests.get(base_url, params=params).json()
            tickers.extend([
                ticker['ticker'].split('.')[0]
                for ticker in response.get('results', [])
            ])
            base_url = response.get('next_url')
            if base_url:
                base_url += f'&apiKey={self.api_key}'

        return list(set(tickers))


class AggregatesAPI:
    """
    A class to fetch aggregate stock data using Polygon.io API.
    """

    def __init__(self, *, api_key):
        self.api_key = api_key
        self.base_url = 'https://api.polygon.io/v2/aggs/ticker'

    def fetch_aggregates(self, *, ticker, from_date, to_date):
        """
        Fetches aggregate data for a single ticker.

        Args:
            ticker (str): Stock ticker symbol.
            from_date (str): Start date for data retrieval.
            to_date (str): End date for data retrieval.

        Returns:
            dict: The aggregate data for the ticker.
        """
        url = f'{self.base_url}/{ticker}/range/1/day/{from_date}/{to_date}?adjusted=true&sort=asc&limit=50000&apiKey={self.api_key}'
        response = requests.get(url)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error for {ticker}: Unable to fetch data. Status code: {response.status_code}")
            return None

    # def fetch_multiple_tickers(self, tickers, from_date, to_date):
    #     results = {}

    #     with concurrent.futures.ThreadPoolExecutor() as executor:
    #         futures = {executor.submit(self.fetch_aggregates, ticker, from_date, to_date): ticker for ticker in tickers}
    #         for future in tqdm(concurrent.futures.as_completed(futures)):
    #             ticker = futures[future]
    #             data = future.result()
    #             if data:
    #                 results[ticker] = data

    def fetch_multiple_tickers(self, *, tickers, from_date, to_date):
        """
        Fetches aggregate data for multiple tickers.

        Args:
            tickers (list): List of stock ticker symbols.
            from_date (str): Start date for data retrieval.
            to_date (str): End date for data retrieval.

        Returns:
            dict: A dictionary of tickers and their corresponding data.
        """
        results = {}
        for ticker in tickers:
            print(f"Fetching data for {ticker}...")
            data = self.fetch_aggregates(ticker=ticker, from_date = from_date, to_date = to_date)
            if data:
                results[ticker] = data
            time.sleep(1)  # Avoid hitting the API rate limit
        return results


def create_chunks(*, lst, chunk_size):
    """
    Splits a list into smaller chunks.

    Args:
        lst (list): The list to split.
        chunk_size (int): The size of each chunk.

    Returns:
        list: A list of chunks.
    """
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def process_data(*, 
    momentum_windows,
    half_lives,
    all_data,
    selected_stocks,
    output_filename,
    is_stock_data = False,
    stock_selection = None
):
    """Generic data processing function for both stocks and bonds."""
    filtered_data = [
        df for df in all_data
        if df.columns[0].split('_')[0] in selected_stocks
    ]

    parallel_results = Parallel(n_jobs=N_JOBS)(
        delayed(process_single_dataframe)(
            df=df.copy(),
            momentum_windows=momentum_windows,
            half_lives=half_lives
        )
        for df in filtered_data
    )

    final_df = makeFinalDf(parallel_results=parallel_results)

    if is_stock_data:
        final_df = makeCorrectedDf(final_df=final_df, selected_stocks=stock_selection)

    final_df.to_csv(output_filename, index=False)
    print(f'Data processing complete. Results saved to: {output_filename}')

def process_data_V(*, 
    momentum_windows,
    half_lives,
    mult,
    weight,
    all_data,
    selected_stocks,
    output_filename,
    is_stock_data = False,
    stock_selection = None
):
    """Generic data processing function for both stocks and bonds."""
    filtered_data = [
        df for df in all_data
        if df.columns[0].split('_')[0] in selected_stocks
    ]

    parallel_results = Parallel(n_jobs=N_JOBS)(
        delayed(process_single_dataframe_V)(
            df=df.copy(),
            momentum_windows=momentum_windows,
            half_lives=half_lives,
            mult=mult,
            weight=weight
        )
        for df in filtered_data
    )

    final_df = makeFinalDf(parallel_results=parallel_results)

    if is_stock_data:
        final_df = makeCorrectedDf(final_df=final_df, selected_stocks=stock_selection)

    final_df.to_csv(output_filename, index=False)
    print(f'Data processing complete. Results saved to: {output_filename}')


def log_test_parameters(*, params, metrics, logger):
    """Log test parameters and results."""
    params_to_log = {
        'mom_window': params['mom_window'],
        'half_life': params['half_life'],
        'n_stock_positive': params['n_stock_positive'],
        'n_stock_chose': params['n_stock_chose'],
        'exp_weight': params['exp_weight'],
        'dvqt': params['dvqt'],
        'topvolstocks': params['topvolstocks']
    }
    param_str = ", ".join(f"{k}: {v}" for k, v in params_to_log.items())
    metric_names = ["annual_return", "sharpe_ratio", "max_drawdown_value", "calmar_ratio", "sortino_ratio"]
    metric_str = ", ".join(f"{name}: {val}" for name, val in zip(metric_names, metrics))
    
    logger.info(f"Parameters: {param_str}")
    logger.info(f"Results: {metric_str}")

def run_parameter_sweep(*, 
    parameter_name,
    parameter_values,
    default_params,
    all_data,
    all_data_bonds,
    SENSITIVITY_DIR,
    run_sensitivity_test
):
    """Run sensitivity tests for different parameter values."""
    results = {
        'anrets': {},
        'sharpes': {},
        'maxdraws': {},
        'calmars': {},
        'sortinos': {}
    }

    results_full = {
        'anrets': {},
        'sharpes': {},
        'maxdraws': {},
        'calmars': {},
        'sortinos': {}
    }

    for value in parameter_values:
        params = default_params.copy()
        params[parameter_name] = value
        
        metrics = run_sensitivity_test(
            **params,
            all_data=all_data,
            all_data_bonds=all_data_bonds
        )

        for metric, result in zip(['anrets', 'sharpes', 'maxdraws', 'calmars', 'sortinos'], metrics[0]):
            results[metric][str(value)] = result

        for metric, result in zip(['anrets', 'sharpes', 'maxdraws', 'calmars', 'sortinos'], metrics[1]):
            results_full[metric][str(value)] = result

    for metric_name, data in results.items():
        filename=f'{metric_name}_{parameter_name}_tillGS.pkl'
        with open(f'{SENSITIVITY_DIR}/{filename}', 'wb') as file:
            pickle.dump(data, file)

    for metric_name, data in results_full.items():
        filename=f'{metric_name}_{parameter_name}_full.pkl'
        with open(f'{SENSITIVITY_DIR}/{filename}', 'wb') as file:
            pickle.dump(data, file)