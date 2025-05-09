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
from definitions.constants import DV_QUANTILE_THRESHOLD_MAKE_YS, GS_ITS, SELECTED_TOP_VOL_STOCKS_MAKE_YS, SELECTED_N_STOCK_POSITIVE_GS, SELECTED_N_STOCK_CHOSE_GS, MOMENTUM_WINDOWS_GS, HALF_LIVES_GS, EXP_WEIGHT_GS

if __name__ == "__main__":
    """
    Script entry point.

    This script processes bond data by applying momentum and exponential weighting
    calculations, and saves the enriched data to a CSV file.
    """
    its=[]
    for qt in DV_QUANTILE_THRESHOLD_MAKE_YS:
        for topnStocksByVolume in SELECTED_TOP_VOL_STOCKS_MAKE_YS:
            for nStockspos in SELECTED_N_STOCK_POSITIVE_GS:
                for topnStockSelect in SELECTED_N_STOCK_CHOSE_GS:
                        for momentum_windows in MOMENTUM_WINDOWS_GS:
                            for half_lives in HALF_LIVES_GS:
                                if nStockspos+topnStockSelect<topnStocksByVolume-1:
                                    for exp_weight in EXP_WEIGHT_GS:
                                        its.append((qt, topnStocksByVolume, (nStockspos+topnStockSelect), topnStockSelect, momentum_windows, half_lives, exp_weight))


    pd.to_pickle(its, GS_ITS)
