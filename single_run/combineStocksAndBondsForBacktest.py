import pandas as pd
import matplotlib.pyplot as plt
import sys
import os



# Append the project root to sys.path
project_root = os.path.dirname(os.path.abspath(__file__))  # Current file directory
project_root = os.path.join(project_root, '..')  # Move one level up to the root
sys.path.append(project_root)


import pickle
from definitions.constants import  (EXP_WEIGHT, FOR_LIVE, SELECTED_HALF_LIFE_WINDOW, SELECTED_MOM_WINDOW, SELECTED_N_STOCK_CHOSE, 
                                    SINGLE_RUN_BONDS_DATA_ENRICHED_CSV, SINGLE_RUN_BONDS_DATA_ENRICHED_LIVE_CSV, SINGLE_RUN_COMBINED_DATA_CSV, 
                                    SINGLE_RUN_COMBINED_DATA_LIVE_CSV, SINGLE_RUN_LIVE_RETURNS_PKL, SINGLE_RUN_LIVE_STOCK_DICT_PKL, SINGLE_RUN_RETURNS_PKL, 
                                    SINGLE_RUN_STOCK_DICT_PKL, SINGLE_RUN_STOCKS_DATA_ENRICHED_CSV, SINGLE_RUN_STOCKS_DATA_ENRICHED_LIVE_CSV,
                                    SELECTED_N_STOCK_POSITIVE)
from utils.custom import calculate_returns, calculate_stock_selection, exponential_weights, load_and_preprocess_data

import numpy as np

# -------------------------------------------------
# Main script for loading, processing, and analyzing data
# -------------------------------------------------

def main():
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
    if FOR_LIVE:
        df = load_and_preprocess_data(file1 = SINGLE_RUN_BONDS_DATA_ENRICHED_LIVE_CSV, file2 = SINGLE_RUN_STOCKS_DATA_ENRICHED_LIVE_CSV, FOR_LIVE = FOR_LIVE)
        df.to_csv(SINGLE_RUN_COMBINED_DATA_LIVE_CSV, index=False)
    else:
        df = load_and_preprocess_data(file1=SINGLE_RUN_BONDS_DATA_ENRICHED_CSV, file2=SINGLE_RUN_STOCKS_DATA_ENRICHED_CSV, FOR_LIVE=FOR_LIVE)
        df.to_csv(SINGLE_RUN_COMBINED_DATA_CSV, index=False)  # Save concatenated data to file

        
    # Step 2: Filter and sort data to include only required columns
    df = df[
        ['Date', f'Momentum_{SELECTED_MOM_WINDOW}_{SELECTED_HALF_LIFE_WINDOW}', 'Stock', 'Returns']
    ].sort_values('Date').reset_index(drop=True)

    # Step 3: Calculate stock selection based on momentum metrics
    stock_dict = calculate_stock_selection(df = df, SELECTED_MOM_WINDOW=SELECTED_MOM_WINDOW, SELECTED_HALF_LIFE_WINDOW=SELECTED_HALF_LIFE_WINDOW, SELECTED_N_STOCK_POSITIVE=SELECTED_N_STOCK_POSITIVE, SELECTED_N_STOCK_CHOSE=SELECTED_N_STOCK_CHOSE)

    if FOR_LIVE:
        with open(SINGLE_RUN_LIVE_STOCK_DICT_PKL, 'wb') as file:
            pickle.dump(stock_dict, file)
    else:
        with open(SINGLE_RUN_STOCK_DICT_PKL, 'wb') as file:
            pickle.dump(stock_dict, file)

    # Step 4: Determine portfolio weights
    # Exponential weights are used if EXP_WEIGHT is True, otherwise a near-uniform weight is applied.
    weights = exponential_weights(
        length = SELECTED_N_STOCK_CHOSE, 
        alpha= EXP_WEIGHT
    )

    # Step 5: Calculate portfolio returns
    returns = calculate_returns(stock_dict = stock_dict, df = df, weights = weights, mom = SELECTED_MOM_WINDOW, half = SELECTED_HALF_LIFE_WINDOW)


    if FOR_LIVE:
        # Step 6: Save returns to a pickle file
        with open(SINGLE_RUN_LIVE_RETURNS_PKL, 'wb') as file:
            pickle.dump(returns, file)
    else:
        with open(SINGLE_RUN_RETURNS_PKL, 'wb') as file:

            pickle.dump(returns, file)


if __name__ == "__main__":
    """
    Entry point for the script. Executes the main function.
    """
    main()
