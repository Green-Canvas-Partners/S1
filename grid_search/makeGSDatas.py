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
from definitions.constants import (FOR_LIVE, GS_BONDS_DATA_ENRICHED_CSV, N_JOBS, MOMENTUM_WINDOWS, HALF_LIVES, STOCKS_DATA_RAW_PKL, GS_STOCKS_DATA_ENRICHED_CSV, SENSITIVITY_DIR, DV_QUANTILE_THRESHOLD_MAKE_YS, SELECTED_TOP_VOL_STOCKS_MAKE_YS, GS_COMBINED_DIR)
from utils.custom import process_single_dataframe, makeFinalDf, makeCorrectedDf, load_and_preprocess_data, add_shift_columns_to_all
# Main function
def main_stocks(*, momentum_windows, half_lives, all_data, selected_stocks, stockstobeused):
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
        delayed(process_single_dataframe)(df=df.copy(), momentum_windows = momentum_windows, half_lives = half_lives)
        for df in filtered_data
    )

    # Step 6: Combine all processed DataFrames into a final DataFrame
    final_df = makeFinalDf(parallel_results = parallel_results)

    # Step 7: Correct the final DataFrame to ensure it matches specified criteria
    corrected_stocks_df = makeCorrectedDf(final_df = final_df, selected_stocks = stockstobeused, FOR_LIVE = FOR_LIVE)

    # Step 8: Save the corrected DataFrame to a CSV file
    filename = GS_STOCKS_DATA_ENRICHED_CSV
    corrected_stocks_df.to_csv(filename, index=False)
    print(f'Data processing complete. Results saved to: {filename}')


def makeGSDatas(*, all_data, dvqt, topvolstocks):
    file_suffix = f"/stockstobeused1_dv_{dvqt}_top_{topvolstocks}.pkl"
    year = pd.read_pickle(SENSITIVITY_DIR + file_suffix)

    stock_lists = year.values()
    combined_stocks = list(set(stock for sublist in stock_lists for stock in sublist))

    main_stocks(
        momentum_windows=MOMENTUM_WINDOWS, 
        half_lives=HALF_LIVES, 
        all_data=all_data, 
        selected_stocks=combined_stocks, 
        stockstobeused=year
    )

    df = load_and_preprocess_data(file1=GS_BONDS_DATA_ENRICHED_CSV, file2=GS_STOCKS_DATA_ENRICHED_CSV, FOR_LIVE=FOR_LIVE)
    df.to_csv(GS_COMBINED_DIR+f"/{dvqt}_{topvolstocks}.csv", index=False)  # Save concatenated data to file

if __name__ == "__main__":
    """
    Script entry point.

    This script processes bond data by applying momentum and exponential weighting
    calculations, and saves the enriched data to a CSV file.
    """
    # Step 1: Load raw stock data from pickle file
    with open(STOCKS_DATA_RAW_PKL, 'rb') as f:
        all_data = pickle.load(f)

    # Step 3: Add shift columns to the loaded stock data
    all_data = add_shift_columns_to_all(all_data=all_data)

    for qts in DV_QUANTILE_THRESHOLD_MAKE_YS:
        for topnstockvolume in SELECTED_TOP_VOL_STOCKS_MAKE_YS:
            makeGSDatas(all_data=all_data, dvqt=qts, topvolstocks=topnstockvolume)
