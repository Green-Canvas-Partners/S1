import pickle
import sys
import os

# Append the project root to sys.path
project_root = os.path.dirname(os.path.abspath(__file__))  # Current file directory
project_root = os.path.join(project_root, '..')  # Move one level up to the root
sys.path.append(project_root)

# Import constants and custom utility functions
from definitions.constants import BOND_TICKERS, LEN_YEARS_DV_LOOKBACK, N_JOBS, STOCKS_DATA_RAW_PKL, YEARS, YEARSTOCKS

from definitions.constants_V import DV_QUANTILE_THRESHOLD_MAKE_YS_V, SELECTED_TOP_VOL_STOCKS_MAKE_YS_V, SENSITIVITY_DIR_V

from utils.custom import (
    stock_selector_sensitivity
)

if __name__ == "__main__":
    """
    Script entry point.

    This script processes bond data by applying momentum and exponential weighting
    calculations, and saves the enriched data to a CSV file.
    """
    # Step 1: Load raw stock data from pickle file
    with open(STOCKS_DATA_RAW_PKL, 'rb') as f:
        all_data = pickle.load(f)

    # Step 2: Select stocks for the specified year
    filename=SENSITIVITY_DIR_V
    stock_selector_sensitivity(all_data=all_data, yearStocks = YEARSTOCKS, YEARS=YEARS, BOND_TICKERS=BOND_TICKERS, LEN_YEARS_DV_LOOKBACK=LEN_YEARS_DV_LOOKBACK, DV_QUANTILE_THRESHOLD=DV_QUANTILE_THRESHOLD_MAKE_YS_V, N_JOBS = N_JOBS, SELECTED_TOP_VOL_STOCKS=SELECTED_TOP_VOL_STOCKS_MAKE_YS_V, YEARSTOCKS_PATH=filename)