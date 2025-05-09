import pandas as pd
import matplotlib.pyplot as plt
import sys
import os



# Append the project root to sys.path
project_root = os.path.dirname(os.path.abspath(__file__))  # Current file directory
project_root = os.path.join(project_root, '..')  # Move one level up to the root
sys.path.append(project_root)


import pickle
from definitions.constants import SELECTED_HALF_LIFE_WINDOW, SELECTED_MOM_WINDOW, DIFF_REBALANCING_COMBINED_DATA_CSV, SELECTED_N_STOCK_POSITIVE
from definitions.constants_V import  (SELECTED_MULT_V, SELECTED_WEIGHT_V, EXP_WEIGHT_V, SELECTED_HALF_LIFE_WINDOW_V, SELECTED_MOM_WINDOW_V, 
                                      SELECTED_N_STOCK_CHOSE_V, DIFF_REBALANCING_BONDS_DATA_ENRICHED_CSV_L, 
                                      DIFF_REBALANCING_COMBINED_DATA_CSV_L, DIFF_REBALANCING_STOCKS_DATA_ENRICHED_CSV_L, 
                                      DIFF_REBALANCING_STOCK_DICT_PKL_L, DIFF_REBALANCING_RETURNS_PKL_L,
                                      SELECTED_N_STOCK_POSITIVE_V)
from utils.custom import calculate_returns_L, calculate_stock_selection_L, exponential_weights, load_and_preprocess_data

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
    bonds_filename=DIFF_REBALANCING_BONDS_DATA_ENRICHED_CSV_L+ str(number) + ".csv"
    stocks_filename=DIFF_REBALANCING_STOCKS_DATA_ENRICHED_CSV_L + str(number) + ".csv"

    df = load_and_preprocess_data(file2=stocks_filename)

    filename=DIFF_REBALANCING_COMBINED_DATA_CSV_L + str(number) + ".csv"
    df.to_csv(filename, index=False)  # Save concatenated data to file

    filename=DIFF_REBALANCING_COMBINED_DATA_CSV + str(number) + ".csv"
    df_M = pd.read_csv(filename)
        
    # Step 2: Filter and sort data to include only required columns

    df_M = df_M[
        ['Date', f'Momentum_{SELECTED_MOM_WINDOW}_{SELECTED_HALF_LIFE_WINDOW}', 'Stock', 'Returns']
    ].sort_values('Date').reset_index(drop=True)

    df_M= df_M[df_M['Date']>'2008-01-01']
    df= df[df['Date']>'2008-01-01']

    print(df.head())

    # Step 3: Calculate stock selection based on momentum metrics
    stock_dict = calculate_stock_selection_L(df = df, df_M = df_M, SELECTED_N_STOCK_POSITIVE=SELECTED_N_STOCK_POSITIVE_V, SELECTED_N_STOCK_CHOSE=SELECTED_N_STOCK_CHOSE_V, SELECTED_MOM_WINDOW_M=SELECTED_MOM_WINDOW, SELECTED_HALF_LIFE_WINDOW_M=SELECTED_HALF_LIFE_WINDOW, SELECTED_N_STOCK_POSITIVE_M=SELECTED_N_STOCK_POSITIVE)

    stock_dict_filename=DIFF_REBALANCING_STOCK_DICT_PKL_L + str(number) + ".pkl"
    with open(stock_dict_filename, 'wb') as file:
        pickle.dump(stock_dict, file)

    # Step 4: Determine portfolio weights
    # Exponential weights are used if EXP_WEIGHT is True, otherwise a near-uniform weight is applied.
    weights = exponential_weights(
        length = SELECTED_N_STOCK_CHOSE_V, 
        alpha= EXP_WEIGHT_V
    )

    # Step 5: Calculate portfolio returns
    returns = calculate_returns_L(stock_dict = stock_dict, df = df, weights = weights)

    returns_diff_rebalancing = DIFF_REBALANCING_RETURNS_PKL_L + str(number) + ".pkl"
    with open(returns_diff_rebalancing, 'wb') as file:
        pickle.dump(returns, file)


if __name__ == "__main__":
    """
    Entry point for the script. Executes the main function.
    """
    for number in range(18):
        main(number)
