import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

# Append the project root to sys.path
project_root = os.path.dirname(os.path.abspath(__file__))  # Current file directory
project_root = os.path.join(project_root, '..')  # Move one level up to the root
sys.path.append(project_root)

from definitions.constants import DIFF_REBALANCING_BONDS_DATA_ENRICHED_CSV, DIFF_REBALANCING_COMBINED_DATA_CSV, DIFF_REBALANCING_STOCK_DICT_PKL, EXP_WEIGHT, DIFF_REBALANCING_RETURNS_PKL,  SELECTED_HALF_LIFE_WINDOW, SELECTED_MOM_WINDOW, SELECTED_N_STOCK_CHOSE, DIFF_REBALANCING_STOCKS_DATA_ENRICHED_CSV, SELECTED_N_STOCK_POSITIVE
from utils.custom import calculate_returns, calculate_stock_selection, exponential_weights, load_and_preprocess_data_M

import pickle

import numpy as np

# -------------------------------------------------
# Main script for loading, processing, and analyzing data
# -------------------------------------------------

def main(number):
    """
    Main function to preprocess data, calculate stock selections, compute returns,
    and save the results.

    Steps:
        1. Load and preprocess data from bonds and stocks.
        2. Filter and sort data for momentum and returns.
        3. Calculate stock selection based on momentum metrics.
        4. Compute portfolio weights.
        5. Calculate and save returns.
    """
    # Step 1: Load and preprocess data from bonds and stocks
    bonds_filename=DIFF_REBALANCING_BONDS_DATA_ENRICHED_CSV+ str(number) + ".csv"

    stocks_filename=DIFF_REBALANCING_STOCKS_DATA_ENRICHED_CSV + str(number) + ".csv"

    df = load_and_preprocess_data_M(file1 = bonds_filename, file2=stocks_filename)

    filename=DIFF_REBALANCING_COMBINED_DATA_CSV + str(number) + ".csv"
    df.to_csv(filename, index=False)

    # Step 2: Filter and sort data to include only required columns
    df = df[
        ['Date', f'Momentum_{SELECTED_MOM_WINDOW}_{SELECTED_HALF_LIFE_WINDOW}', 'Stock', 'Returns']
    ].sort_values('Date').reset_index(drop=True)

    # Step 3: Calculate stock selection based on momentum metrics
    stock_dict = calculate_stock_selection(df = df, SELECTED_MOM_WINDOW=SELECTED_MOM_WINDOW, SELECTED_HALF_LIFE_WINDOW=SELECTED_HALF_LIFE_WINDOW, SELECTED_N_STOCK_POSITIVE=SELECTED_N_STOCK_POSITIVE, SELECTED_N_STOCK_CHOSE=SELECTED_N_STOCK_CHOSE)

    stock_dict_filename=DIFF_REBALANCING_STOCK_DICT_PKL + str(number) + ".pkl"
    with open(stock_dict_filename, 'wb') as file:
        pickle.dump(stock_dict, file)

    # Step 4: Determine portfolio weights
    # Exponential weights are used if EXP_WEIGHT is True, otherwise a near-uniform weight is applied.
    weights = exponential_weights(
        length = SELECTED_N_STOCK_CHOSE, 
        alpha= EXP_WEIGHT
    )

    # Step 5: Calculate portfolio returns
    returns = calculate_returns(stock_dict = stock_dict, df = df, weights = weights, mom = SELECTED_MOM_WINDOW, half = SELECTED_HALF_LIFE_WINDOW)

    returns_diff_rebalancing = DIFF_REBALANCING_RETURNS_PKL + str(number) + ".pkl"
    with open(returns_diff_rebalancing, 'wb') as file:
        pickle.dump(returns, file)


if __name__ == "__main__":
    """
    Entry point for the script. Executes the main function.
    """
    for number in range(18):
        main(number)

