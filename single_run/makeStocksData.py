import pandas as pd
import pickle
import sys
import os
from joblib import Parallel, delayed

# Append the project root to sys.path
project_root = os.path.dirname(os.path.abspath(__file__))  # Current file directory
project_root = os.path.join(project_root, '..')  # Move one level up to the root
sys.path.append(project_root)

# Import constants and custom utility functions
from definitions.constants import BOND_TICKERS, DV_QUANTILE_THRESHOLD, FOR_LIVE, HALF_LIVES, LEN_YEARS_DV_LOOKBACK, MOMENTUM_WINDOWS, N_JOBS, SELECTED_TOP_VOL_STOCKS, SINGLE_RUN_STOCKS_DATA_ENRICHED_CSV, SINGLE_RUN_STOCKS_DATA_ENRICHED_LIVE_CSV, SINGLE_RUN_YEARSTOCKS_LIVE_PKL, SINGLE_RUN_YEARSTOCKS_PKL, STOCKS_DATA_RAW_LIVE_PKL, STOCKS_DATA_RAW_PKL, YEARS, YEARSTOCKS

from utils.custom import (
    add_shift_columns_to_all, process_single_dataframe, 
    stock_selector, makeFinalDf, makeCorrectedDf
)

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
    DV_QUANTILE_THRESHOLD=DV_QUANTILE_THRESHOLD,
    N_JOBS=N_JOBS,
    SELECTED_TOP_VOL_STOCKS=SELECTED_TOP_VOL_STOCKS,
    FOR_LIVE=FOR_LIVE,
    YEARSTOCKS_PATH_LIVE=SINGLE_RUN_YEARSTOCKS_LIVE_PKL,
    YEARSTOCKS_PATH=SINGLE_RUN_YEARSTOCKS_PKL
)
stock_lists = year.values()
combined_stocks = list(set(stock for sublist in stock_lists for stock in sublist))

# Step 3: Add shift columns to the loaded stock data
all_data = add_shift_columns_to_all(all_data = all_data)

# Main function
def main(momentum_windows, half_lives, all_data, selected_stocks, stockstobeused):
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
    
    # Step 5: Process each filtered DataFrame in parallel
    parallel_results = Parallel(n_jobs=N_JOBS)(
        delayed(process_single_dataframe)(df = df.copy(), momentum_windows = momentum_windows, half_lives = half_lives)
        for df in filtered_data
    )

    # Step 6: Combine all processed DataFrames into a final DataFrame
    final_df = makeFinalDf(parallel_results = parallel_results)

    # Step 7: Correct the final DataFrame to ensure it matches specified criteria
    corrected_stocks_df = makeCorrectedDf(final_df = final_df, selected_stocks = stockstobeused, FOR_LIVE=FOR_LIVE)

    # Step 8: Save the corrected DataFrame to a CSV file
    if FOR_LIVE:
        corrected_stocks_df.to_csv(SINGLE_RUN_STOCKS_DATA_ENRICHED_LIVE_CSV, index=False)
        print(f'Data processing complete. Results saved to: {SINGLE_RUN_STOCKS_DATA_ENRICHED_LIVE_CSV}')
    else:
        corrected_stocks_df.to_csv(SINGLE_RUN_STOCKS_DATA_ENRICHED_CSV, index=False)
        print(f'Data processing complete. Results saved to: {SINGLE_RUN_STOCKS_DATA_ENRICHED_CSV}')

# Execute the main function
if __name__ == "__main__":
    """
    Script entry point.

    This script processes stock data by applying momentum and exponential weighting
    calculations, correcting the final data, and saving it to a CSV file.
    """
    main(
        MOMENTUM_WINDOWS, 
        HALF_LIVES, 
        all_data, 
        combined_stocks,
        year
    )
