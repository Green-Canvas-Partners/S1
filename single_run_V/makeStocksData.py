import pandas as pd
import pickle
import sys
import os
# from grid_search_V.makeGSDatas import process_single_dataframe_V_remote
from joblib import Parallel, delayed
import time
import ray

# Append the project root to sys.path
project_root = os.path.dirname(os.path.abspath(__file__))  # Current file directory
project_root = os.path.join(project_root, '..')  # Move one level up to the root
sys.path.append(project_root)

# Import constants and custom utility functions
from definitions.constants import BOND_TICKERS, FOR_LIVE, LEN_YEARS_DV_LOOKBACK, N_JOBS, STOCKS_DATA_RAW_LIVE_PKL, STOCKS_DATA_RAW_PKL, YEARS, YEARSTOCKS, USE_RAY

from definitions.constants_V import (
    MOMENTUM_WINDOWS_V, HALF_LIVES_V, DV_QUANTILE_THRESHOLD_V, SELECTED_TOP_VOL_STOCKS_V, 
    SINGLE_RUN_STOCKS_DATA_ENRICHED_CSV_L, SINGLE_RUN_STOCKS_DATA_ENRICHED_LIVE_CSV_L, 
    SINGLE_RUN_YEARSTOCKS_LIVE_PKL_L, SINGLE_RUN_YEARSTOCKS_PKL_L, MULT_V, WEIGHT_V,etfs_to_exclude
)

from utils.custom import (
    add_shift_columns_to_all, process_single_dataframe_E, 
    stock_selector, makeFinalDf, makeCorrectedDf
)

if USE_RAY:
    # Define the runtime environment
    runtime_env = {
        "working_dir": "/home/iyad/E1_DIR/E1"
    }

    # Initialize Ray with the runtime environment
    ray.init(_redis_password='password', runtime_env=runtime_env)

if USE_RAY:
    # Wrap the function with Ray's remote decorator
    @ray.remote
    def process_single_dataframe_V_remote(*args, **kwargs):
        return process_single_dataframe_E(*args, **kwargs)

if FOR_LIVE:
    # Step 1: Load raw stock data from pickle file
    with open(STOCKS_DATA_RAW_LIVE_PKL, 'rb') as f:
        all_data = pickle.load(f)
else:
    # Step 1: Load raw stock data from pickle file
    with open(STOCKS_DATA_RAW_PKL, 'rb') as f:
        all_data = pickle.load(f)

# Step 2: Select stocks for the specified year
year = stock_selector(
    all_data=all_data,
    yearStocks=YEARSTOCKS,
    YEARS=YEARS,
    BOND_TICKERS=BOND_TICKERS,
    LEN_YEARS_DV_LOOKBACK=LEN_YEARS_DV_LOOKBACK,
    DV_QUANTILE_THRESHOLD=DV_QUANTILE_THRESHOLD_V,
    N_JOBS=N_JOBS,
    SELECTED_TOP_VOL_STOCKS=SELECTED_TOP_VOL_STOCKS_V,
    FOR_LIVE=FOR_LIVE,
    YEARSTOCKS_PATH_LIVE=SINGLE_RUN_YEARSTOCKS_LIVE_PKL_L,
    YEARSTOCKS_PATH=SINGLE_RUN_YEARSTOCKS_PKL_L
)

stock_lists = year.values()
combined_stocks = list(set(stock for sublist in stock_lists for stock in sublist))

# Step 3: Add shift columns to the loaded stock data
all_data = add_shift_columns_to_all(all_data = all_data)

# Main function
def main(all_data, selected_stocks, stockstobeused):
    """
    Main function to process stock data.

    Args:
        momentum_windows (list): List of momentum windows for calculations.
        half_lives (list): List of half-lives for exponential weighting.
        all_data (list): List of DataFrames containing stock data.
        selected_stocks (list): List of selected stocks to process.
        stockstobeused (dict): Dictionary of stocks selected for the specified year.

    Returns:
        None
    """
    # Step 4: Filter the data for selected stocks
    filtered_data = [
        df for df in all_data 
        if df.columns[0].split('_')[0] in selected_stocks
    ]
    
    if USE_RAY:
        remote_functions = [
            process_single_dataframe_V_remote.remote(df=df.copy(), momentum_windows=momentum_windows, half_lives=half_lives, mult=mult, weight=weight)
            for df in filtered_data
        ]
        # Fetch results from Ray
        parallel_results = ray.get(remote_functions)

    else:
        print("here")
        # Step 5: Process each filtered DataFrame in parallel
        parallel_results = Parallel(n_jobs=N_JOBS)(
            delayed(process_single_dataframe_E)(df = df.copy())
            for df in filtered_data
        )
    # Step 6: Combine all processed DataFrames into a final DataFrame
    final_df = makeFinalDf(parallel_results = parallel_results)
    # Step 7: Correct the final DataFrame to ensure it matches specified criteria
    corrected_stocks_df = makeCorrectedDf(final_df = final_df, selected_stocks = stockstobeused, FOR_LIVE=FOR_LIVE)
    # Step 8: Save the corrected DataFrame to a CSV file
    if FOR_LIVE:
        corrected_stocks_df.to_csv(SINGLE_RUN_STOCKS_DATA_ENRICHED_LIVE_CSV_L, index=False)
        print(f'Data processing complete. Results saved to: {SINGLE_RUN_STOCKS_DATA_ENRICHED_LIVE_CSV_L}')
    else:
        corrected_stocks_df.to_csv(SINGLE_RUN_STOCKS_DATA_ENRICHED_CSV_L, index=False)
        print(f'Data processing complete. Results saved to: {SINGLE_RUN_STOCKS_DATA_ENRICHED_CSV_L}')

# Execute the main function
if __name__ == "__main__":
    """
    Script entry point.

    This script processes stock data by applying momentum and exponential weighting
    calculations, correcting the final data, and saving it to a CSV file.
    """
    start_time = time.time()  # Start timing
    main(
        all_data, 
        combined_stocks,
        year
    )
    end_time = time.time()  # End timing
    elapsed_time = end_time - start_time  # Calculate elapsed time
    print(f"Script execution time: {elapsed_time:.2f} seconds")
    if USE_RAY:
        ray.shutdown()
