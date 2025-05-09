import pandas as pd
import pickle
import sys
import os
from joblib import Parallel, delayed
from datetime import timedelta

# Append the project root to sys.path
project_root = os.path.dirname(os.path.abspath(__file__))  # Current file directory
project_root = os.path.join(project_root, '..')  # Move one level up to the root
sys.path.append(project_root)

# Import constants and custom utility functions
from definitions.constants import (
    BOND_TICKERS, START_DATE_DATA_DOWNLOAD, END_DATE_DATA_DOWNLOAD, MOMENTUM_WINDOWS, 
    HALF_LIVES, YEARS, SINGLE_RUN_BONDS_DATA_RAW_PKL, SINGLE_RUN_BONDS_DATA_ENRICHED_CSV, N_JOBS, FOR_LIVE, 
    START_DATE_GET_TICKERS_AND_DATA_DOWNLOAD_FOR_LIVE, END_DATE_FOR_LIVE, SINGLE_RUN_BONDS_DATA_RAW_LIVE_PKL, 
    SINGLE_RUN_BONDS_DATA_ENRICHED_LIVE_CSV
)
from utils.custom import (
    download_data, add_shift_columns_to_all, 
    process_single_dataframe, makeFinalDf
)

if FOR_LIVE:
    # Step 1: Download bond data for specified tickers and date range
    download_data(tickers=BOND_TICKERS, start_date=START_DATE_GET_TICKERS_AND_DATA_DOWNLOAD_FOR_LIVE, end_date=END_DATE_FOR_LIVE, bonds_data_path_raw=SINGLE_RUN_BONDS_DATA_RAW_LIVE_PKL)
else:
    # Step 1: Download bond data for specified tickers and date range
    download_data(tickers=BOND_TICKERS, start_date=START_DATE_DATA_DOWNLOAD, end_date=(pd.to_datetime(END_DATE_DATA_DOWNLOAD) + timedelta(days=1)).strftime('%Y-%m-%d'), bonds_data_path_raw=SINGLE_RUN_BONDS_DATA_RAW_PKL)

if FOR_LIVE:
    with open(SINGLE_RUN_BONDS_DATA_RAW_LIVE_PKL, 'rb') as f:
        all_data_bonds = pickle.load(f)
else:
    # Step 2: Load the raw bond data from a pickle file
    with open(SINGLE_RUN_BONDS_DATA_RAW_PKL, 'rb') as f:
        all_data_bonds = pickle.load(f)

# Step 3: Add shift columns to the loaded bond data
all_data_bonds = add_shift_columns_to_all(all_data = all_data_bonds)

# Main function
def main(momentum_windows, half_lives, years, all_data, selected_stocks):
    """
    Main function to process bond data.

    Args:
        momentum_windows (list): List of momentum windows for calculations.
        half_lives (list): List of half-lives for exponential weighting.
        years (int): Number of years of data to process.
        all_data (list): List of DataFrames containing bond data.
        selected_stocks (list): List of selected bond tickers to process.

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
        delayed(process_single_dataframe)(df = df.copy(), momentum_windows=momentum_windows, half_lives = half_lives) 
        for df in filtered_data
    )

    # Step 6: Combine all processed DataFrames into a final DataFrame
    final_df = makeFinalDf(parallel_results=parallel_results)

    # Step 7: Save the final DataFrame to a CSV file
    if FOR_LIVE:
        filename = SINGLE_RUN_BONDS_DATA_ENRICHED_LIVE_CSV
    else:
        filename = SINGLE_RUN_BONDS_DATA_ENRICHED_CSV
    final_df.to_csv(filename, index=False)
    print(f'Data processing complete. Results saved to: {filename}')

if __name__ == "__main__":
    """
    Script entry point.

    This script processes bond data by applying momentum and exponential weighting
    calculations, and saves the enriched data to a CSV file.
    """
    main(
        MOMENTUM_WINDOWS, 
        HALF_LIVES, 
        YEARS, 
        all_data_bonds, 
        BOND_TICKERS
    )
