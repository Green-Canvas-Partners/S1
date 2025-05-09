import pandas as pd
import pickle
import sys
import os
from joblib import Parallel, delayed
from datetime import timedelta
import random

# Append the project root to sys.path
project_root = os.path.dirname(os.path.abspath(__file__))  # Current file directory
project_root = os.path.join(project_root, '..')  # Move one level up to the root
sys.path.append(project_root)

# Import constants and custom utility functions
from definitions.constants import GS_COMBINED_DIR, DATE_GS_CUTOFF, EXP_WEIGHT, GS_ITS, GS_RES, N_JOBS

from utils.custom import (
    calculate_stock_selection,
    exponential_weights, calculate_returns, calculate_metrics
)

def run_gs(*, qt, topnStocksByVolume, stockPos, topnStockSelect, momentum_windows, half_lives, exp_weight):
    df=pd.read_csv(GS_COMBINED_DIR+f"/{qt}_{topnStocksByVolume}.csv")

    df = df[
        ['Date', f'Momentum_{momentum_windows}_{half_lives}', 'Stock', 'Returns']
    ].sort_values('Date').reset_index(drop=True)

    df=df[df.Date<=DATE_GS_CUTOFF]

    # Step 3: Calculate stock selection based on momentum metrics
    stock_dict = calculate_stock_selection(df = df, SELECTED_MOM_WINDOW=momentum_windows, SELECTED_HALF_LIFE_WINDOW=half_lives, SELECTED_N_STOCK_POSITIVE=stockPos, SELECTED_N_STOCK_CHOSE=topnStockSelect)

    # Step 4: Determine portfolio weights
    # Exponential weights are used if EXP_WEIGHT is True, otherwise a near-uniform weight is applied.
    weights = exponential_weights(
        length = topnStockSelect, 
        alpha= exp_weight
    )

    # Step 5: Calculate portfolio returns
    returns = calculate_returns(stock_dict=stock_dict, df=df, weights=weights, mom=momentum_windows, half=half_lives)

    annual_return, sharpe_ratio, max_drawdown_value, calmar_ratio, sortino_ratio = calculate_metrics(returns = returns)

    return annual_return, sharpe_ratio, max_drawdown_value, calmar_ratio, sortino_ratio


def run_iteration(it):
    a, s, m, c, ss = run_gs(qt=it[0], topnStocksByVolume=it[1], stockPos=it[2], topnStockSelect=it[3], momentum_windows=it[4], half_lives=it[5], exp_weight=it[6])
    return {
        str(it): [a, s, m, c, ss],
    }


if __name__ == "__main__":
    """
    Script entry point.

    This script processes bond data by applying momentum and exponential weighting
    calculations, and saves the enriched data to a CSV file.
    """
    its=pd.read_pickle(GS_ITS)

    results = Parallel(n_jobs=N_JOBS)(delayed(run_iteration)(it) for it in its)

    dictt = {}
    for result in results:
        for key, value in result.items():
            dictt[key] = value

    rows = []
    for key, values in dictt.items():
        # Parse the key which is in tuple format but as a string, eval is used to convert it to an actual tuple
        key_tuple = eval(key)
        # Unpack all values and combine them with the keys into one list
        row = list(key_tuple) + values
        rows.append(row)

    # Define the column names
    columns = [
        'qt', 'topnStocksByVolume', 'nStockspos_plus_topnStockSelect', 'topnStockSelect', 'momentum_windows', 'half_lives', 'exp_weight',
        'annual_returns', 'sharpe_ratios', 'max_drawdown_values', 'calmar_ratios', 'sortino_ratios'
    ]

    # Creating the DataFrame
    df = pd.DataFrame(rows, columns=columns)
    df.to_csv(GS_RES, index=False)

