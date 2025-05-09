import multiprocessing
import pickle
import pandas as pd
import yfinance as yf
import sys
import os
import time

# Append the project root to sys.path
project_root = os.path.dirname(os.path.abspath(__file__))  # Current file directory
project_root = os.path.join(project_root, '..')  # Move one level up to the root
sys.path.append(project_root)

from definitions.constants import API_KEY, START_DATE_GET_TICKERS_AND_DATA_DOWNLOAD_FOR_LIVE, STOCKS_DATA_RAW_LIVE_PKL, CHUNK_SIZE, END_DATE_FOR_LIVE
from utils.custom import TickerAPI, AggregatesAPI, create_chunks

# Step 1: Fetch dates from SPY data
df = yf.download('SPY')
dates = [str(i.date()) for i in df[df.index > START_DATE_GET_TICKERS_AND_DATA_DOWNLOAD_FOR_LIVE].index.tolist()]

# Step 2: Initialize TickerAPI instance
ticker_api = TickerAPI(api_key=API_KEY)

# Step 3: Fetch tickers using multiprocessing
def get_tickers(date, ticker_api):
    return ticker_api.get_stock_tickers(start_date=date)

with multiprocessing.Pool() as pool:
    results = pool.starmap(get_tickers, [(date, ticker_api) for date in dates])

# Step 4: Process and clean results
new_results = []
for i in results:
    new_results.extend(i)
new_results = list(set(new_results))
new_results = [i.replace('/', '.') for i in new_results]

# Step 5: Create chunks for parallel processing
chunks = create_chunks(lst=new_results, chunk_size=CHUNK_SIZE)

# Step 6: Fetch aggregate data for tickers
chunked_results = []
aggregates_api = AggregatesAPI(api_key=API_KEY)

for chunk in chunks:
    ticker_data = aggregates_api.fetch_multiple_tickers(tickers=chunk, from_date=START_DATE_GET_TICKERS_AND_DATA_DOWNLOAD_FOR_LIVE, to_date=END_DATE_FOR_LIVE)
    chunked_results.append(ticker_data)
    time.sleep(10)

# Step 7: Convert results to DataFrames
all_dfs = []
for ticker_data in chunked_results:
    for ticker, data in ticker_data.items():
        try:
            hist = pd.DataFrame(data['results'])
            hist.rename(columns={
                'c': 'Close',
                't': 'Date',
                'v': 'Volume',
                'o': 'Open',
                'h': 'High',
                'l': 'Low'
            }, inplace=True)
            hist['Date'] = pd.to_datetime(hist['Date'], unit='ms')
            hist['Date'] = hist['Date'].dt.tz_localize('UTC')
            hist['Date'] = hist['Date'].dt.tz_convert('US/Eastern')
            hist['Date'] = hist['Date'].dt.tz_localize(None)
            hist.set_index('Date', inplace=True)
            hist.dropna(inplace=True)
            hist = hist.add_prefix(ticker + '_')
            all_dfs.append(hist)
        except Exception as e:
            print(f"Error processing data for {ticker}: {e}")

# Step 8: Save DataFrames to a pickle file
with open(STOCKS_DATA_RAW_LIVE_PKL, 'wb') as f:
    pickle.dump(all_dfs, f)
print(f"All data saved to {STOCKS_DATA_RAW_LIVE_PKL}.")