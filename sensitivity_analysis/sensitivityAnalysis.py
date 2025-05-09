import pandas as pd
import pickle
import sys
import os
import logging
from joblib import Parallel, delayed
from datetime import timedelta
from typing import Dict, List, Tuple

# Append the project root to sys.path
project_root = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(project_root, '..')
sys.path.append(project_root)

# Import constants and custom utility functions
from definitions.constants import (
    BOND_TICKERS, SENSITIVITY_BONDS_DATA_ENRICHED_CSV, SENSITIVITY_BONDS_DATA_RAW_PKL,
    DV_QUANTILE_THRESHOLD, DV_QUANTILE_THRESHOLD_SENS, END_DATE_DATA_DOWNLOAD, EXP_WEIGHT,
    EXP_WEIGHT_SENS, HALF_LIVES, HALF_LIVES_SENS, MOMENTUM_WINDOWS,
    MOMENTUM_WINDOWS_SENS, SELECTED_HALF_LIFE_WINDOW, SELECTED_MOM_WINDOW,
    SELECTED_N_STOCK_CHOSE, SELECTED_N_STOCK_CHOSE_SENS, SELECTED_N_STOCK_POSITIVE,
    SELECTED_N_STOCK_POSITIVE_SENS, SELECTED_TOP_VOL_STOCKS, SELECTED_TOP_VOL_STOCKS_SENS,
    SENSITIVITY_COMBINED_DATA_CSV, SENSITIVITY_DIR, SENSITIVITY_STOCKS_DATA_ENRICHED_CSV,
    START_DATE_DATA_DOWNLOAD, STOCKS_DATA_RAW_PKL, DATES_SENS
)

from utils.custom import (
    download_data, add_shift_columns_to_all, load_and_preprocess_data,
    calculate_stock_selection, exponential_weights, calculate_returns, calculate_metrics, process_data, log_test_parameters, run_parameter_sweep
)

# Configure logger
logging.basicConfig(
    filename='logfile.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def run_sensitivity_test(
    mom_window,
    half_life,
    n_stock_positive,
    n_stock_chose,
    exp_weight,
    dvqt,
    topvolstocks,
    all_data,
    all_data_bonds
):
    """Run a single sensitivity test and return metrics."""
    file_suffix = f"/stockstobeused1_dv_{dvqt}_top_{topvolstocks}.pkl"
    year_data = pd.read_pickle(SENSITIVITY_DIR + file_suffix)
    combined_stocks = list(set(stock for sublist in year_data.values() for stock in sublist))

    # Process bonds
    process_data(
        momentum_windows=list(set(mom_window + [max(MOMENTUM_WINDOWS)])),
        half_lives=list(set(half_life + [max(HALF_LIVES)])),
        all_data=all_data_bonds,
        selected_stocks=BOND_TICKERS,
        output_filename=SENSITIVITY_BONDS_DATA_ENRICHED_CSV
    )

    # Process stocks
    process_data(
        momentum_windows=list(set(mom_window + [max(MOMENTUM_WINDOWS)])),
        half_lives=list(set(half_life + [max(HALF_LIVES)])),
        all_data=all_data,
        selected_stocks=combined_stocks,
        output_filename=SENSITIVITY_STOCKS_DATA_ENRICHED_CSV,
        is_stock_data=True,
        stock_selection=year_data
    )

    # Process combined data
    df = load_and_preprocess_data(
        file1=SENSITIVITY_BONDS_DATA_ENRICHED_CSV,
        file2=SENSITIVITY_STOCKS_DATA_ENRICHED_CSV
    )
    df.to_csv(SENSITIVITY_COMBINED_DATA_CSV, index=False)

    # Calculate metrics
    filtered_df = df[['Date', f'Momentum_{mom_window[0]}_{half_life[0]}', 'Stock', 'Returns']]
    stock_dict = calculate_stock_selection(
        df=filtered_df,
        SELECTED_MOM_WINDOW=mom_window[0],
        SELECTED_HALF_LIFE_WINDOW=half_life[0],
        SELECTED_N_STOCK_POSITIVE=n_stock_positive,
        SELECTED_N_STOCK_CHOSE=n_stock_chose
    )

    weights = exponential_weights(
        length=n_stock_chose,
        alpha= exp_weight
    )

    returns = calculate_returns(
        stock_dict=stock_dict,
        df=filtered_df,
        weights=weights,
        mom=mom_window[0],
        half=half_life[0]
    )
    ms=[]
    for i in DATES_SENS:
        metrics = calculate_metrics(returns=returns[returns.index<=i])
        ms.append(metrics)

    log_test_parameters(params=locals(), metrics=ms[-1], logger=logging)
    return ms

if __name__ == "__main__":
    # Initial data loading and processing
    download_data(
        tickers=BOND_TICKERS,
        start_date=START_DATE_DATA_DOWNLOAD,
        end_date=(pd.to_datetime(END_DATE_DATA_DOWNLOAD) + timedelta(days=1)).strftime('%Y-%m-%d'),
        bonds_data_path_raw=SENSITIVITY_BONDS_DATA_RAW_PKL
    )

    # Load and process bond data
    with open(SENSITIVITY_BONDS_DATA_RAW_PKL, 'rb') as f:
        all_data_bonds = pickle.load(f)
    all_data_bonds = add_shift_columns_to_all(all_data=all_data_bonds)

    # Load and process stock data
    with open(STOCKS_DATA_RAW_PKL, 'rb') as f:
        all_data = pickle.load(f)
    all_data = add_shift_columns_to_all(all_data=all_data)

    # Base configuration
    default_params = {
        'mom_window': [SELECTED_MOM_WINDOW],
        'half_life': [SELECTED_HALF_LIFE_WINDOW],
        'n_stock_positive': SELECTED_N_STOCK_POSITIVE,
        'n_stock_chose': SELECTED_N_STOCK_CHOSE,
        'exp_weight': EXP_WEIGHT,
        'dvqt': DV_QUANTILE_THRESHOLD,
        'topvolstocks': SELECTED_TOP_VOL_STOCKS
    }

    # Run parameter sweeps
    parameter_sweeps = [
        ('mom_window', [mw for mw in MOMENTUM_WINDOWS_SENS]),
        ('half_life', [hl for hl in HALF_LIVES_SENS]),
        ('dvqt', DV_QUANTILE_THRESHOLD_SENS),
        ('topvolstocks', SELECTED_TOP_VOL_STOCKS_SENS),
        ('exp_weight', EXP_WEIGHT_SENS),
        ('n_stock_chose', SELECTED_N_STOCK_CHOSE_SENS),
        ('n_stock_positive', SELECTED_N_STOCK_POSITIVE_SENS)
    ]

    for param_name, param_values in parameter_sweeps:
        run_parameter_sweep(
            parameter_name=param_name,
            parameter_values=param_values,
            default_params=default_params,
            all_data=all_data,
            all_data_bonds=all_data_bonds,
            SENSITIVITY_DIR=SENSITIVITY_DIR,
            run_sensitivity_test=run_sensitivity_test
        )