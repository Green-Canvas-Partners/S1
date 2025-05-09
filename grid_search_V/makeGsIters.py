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
from definitions.constants_V import (DV_QUANTILE_THRESHOLD_MAKE_YS_V, GS_ITS_V, SELECTED_TOP_VOL_STOCKS_MAKE_YS_V, 
                                     SELECTED_N_STOCK_POSITIVE_GS_V, SELECTED_N_STOCK_CHOSE_GS_V, MOMENTUM_WINDOWS_GS_V, 
                                     HALF_LIVES_GS_V, EXP_WEIGHT_GS_V, MULT_GS_V, WEIGHT_GS_V
)

M2_PARAMS=[
    (0.75, 75, 252, 250, 30),
    (0.5, 20, 252, 126, 16),
    (0.5, 27, 252, 126, 16),
    (0.2, 11, 252, 126, 7),
    (0.2, 11, 252, 250, 10),
    (0.66, 75, 252, 150, 55),
    (0.05, 150, 252, 250, 51)
]

if __name__ == "__main__":
    """
    Script entry point.

    This script processes bond data by applying momentum and exponential weighting
    calculations, and saves the enriched data to a CSV file.
    """
    its=[]
    for qt in DV_QUANTILE_THRESHOLD_MAKE_YS_V:
        for topnStocksByVolume in SELECTED_TOP_VOL_STOCKS_MAKE_YS_V:
            for nStockspos in SELECTED_N_STOCK_POSITIVE_GS_V:
                for topnStockSelect in SELECTED_N_STOCK_CHOSE_GS_V:
                        for momentum_windows in MOMENTUM_WINDOWS_GS_V:
                            for half_lives in HALF_LIVES_GS_V:
                                if nStockspos+topnStockSelect<topnStocksByVolume-1:
                                    for exp_weight in EXP_WEIGHT_GS_V:
                                        for mult in MULT_GS_V:
                                            for weight in WEIGHT_GS_V:
                                                for m2 in M2_PARAMS:
                                                    its.append((qt, topnStocksByVolume, (nStockspos+topnStockSelect), topnStockSelect, momentum_windows, half_lives, exp_weight, mult, weight, m2))


    pd.to_pickle(its, GS_ITS_V)
