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
    BOND_TICKERS, START_DATE_DATA_DOWNLOAD, END_DATE_DATA_DOWNLOAD, YEARS, N_JOBS
)
from definitions.constants_V import (
    GS_BONDS_DATA_ENRICHED_CSV_V, GS_BONDS_DATA_RAW_PKL_V, MOMENTUM_WINDOWS_GS_V, 
    HALF_LIVES_GS_V, MULT_GS_V, WEIGHT_GS_V
)
from utils.custom import (
    download_data, add_shift_columns_to_all, 
    process_single_dataframe_V, makeFinalDf
)

# Step 1: Download bond data for specified tickers and date range
download_data(tickers = BOND_TICKERS, start_date = START_DATE_DATA_DOWNLOAD, end_date=(pd.to_datetime(END_DATE_DATA_DOWNLOAD) + timedelta(days=1)).strftime('%Y-%m-%d'), bonds_data_path_raw=GS_BONDS_DATA_RAW_PKL_V)

# Step 2: Load the raw bond data from a pickle file
with open(GS_BONDS_DATA_RAW_PKL_V, 'rb') as f:
    all_data_bonds = pickle.load(f)

# Step 3: Add shift columns to the loaded bond data
all_data_bonds = add_shift_columns_to_all(all_data=all_data_bonds)

# Main function
def main(momentum_windows, half_lives, mult, weight, years, all_data, selected_stocks):
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
        delayed(process_single_dataframe_V)(df=df.copy(), momentum_windows=momentum_windows, half_lives=half_lives, mult=mult, weight=weight) 
        for df in filtered_data
    )

    # Step 6: Combine all processed DataFrames into a final DataFrame
    final_df = makeFinalDf(parallel_results=parallel_results)

    # Step 7: Save the final DataFrame to a CSV file
    filename = GS_BONDS_DATA_ENRICHED_CSV_V
    final_df.to_csv(filename, index=False)
    print(f'Data processing complete. Results saved to: {filename}')

if __name__ == "__main__":
    """
    Script entry point.

    This script processes bond data by applying momentum and exponential weighting
    calculations, and saves the enriched data to a CSV file.
    """
    main(
        MOMENTUM_WINDOWS_GS_V, 
        HALF_LIVES_GS_V, 
        MULT_GS_V,
        WEIGHT_GS_V,
        YEARS, 
        all_data_bonds, 
        BOND_TICKERS
    )
