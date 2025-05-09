import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

# Append the project root to sys.path
project_root = os.path.dirname(os.path.abspath(__file__))  # Current file directory
project_root = os.path.join(project_root, '..')  # Move one level up to the root
sys.path.append(project_root)


import pickle
from definitions.constants_V import (DIFF_REBALANCING_COMBINED_DATA_ALL_PKL_L, DIFF_REBALANCING_COMBINED_DATA_CSV_L, 
                                     DIFF_REBALANCING_RETURNS_ALL_PKL_L, DIFF_REBALANCING_STOCK_DICT_ALL_PKL_L, DIFF_REBALANCING_RETURNS_PKL_L, 
                                     DIFF_REBALANCING_STOCK_DICT_PKL_L, DIFF_REBALANCING_PICASSO_DIR_L, DV_QUANTILE_THRESHOLD_V, 
                                     SELECTED_TOP_VOL_STOCKS_V, SELECTED_MOM_WINDOW_V, SELECTED_HALF_LIFE_WINDOW_V, SELECTED_N_STOCK_POSITIVE_V, 
                                     SELECTED_N_STOCK_CHOSE_V, EXP_WEIGHT_V, MULT_V, WEIGHT_V)

from definitions.constants import (DV_QUANTILE_THRESHOLD, SELECTED_TOP_VOL_STOCKS, SELECTED_MOM_WINDOW, SELECTED_HALF_LIFE_WINDOW, SELECTED_N_STOCK_POSITIVE)
# -------------------------------------------------
# Main script for loading, processing, and analyzing data
# -------------------------------------------------

if __name__ == "__main__":
    """
    Entry point for the script. Executes the main function.
    """
    retss=[]
    stks=[]
    picasodfs=[]
    for number in range(18):
        filename=DIFF_REBALANCING_COMBINED_DATA_CSV_L + str(number) + ".csv"
        df=pd.read_csv(filename)
        picasodfs.append(df)

        stock_dict_filename=DIFF_REBALANCING_STOCK_DICT_PKL_L + str(number) + ".pkl"
        stk=pd.read_pickle(stock_dict_filename)
        stks.append(stk)

        returns_filename=DIFF_REBALANCING_RETURNS_PKL_L + str(number) + ".pkl"
        rets=pd.read_pickle(returns_filename)

        rets=pd.DataFrame(rets, columns=['returns'])
        rets.reset_index(inplace=True)
        rets.rename(columns={'index':'t'}, inplace=True)
        rets.set_index('t', inplace=True)
        retss.append(rets)

    dfsfilename=DIFF_REBALANCING_COMBINED_DATA_ALL_PKL_L
    pd.to_pickle(picasodfs, dfsfilename)

    stock_dict_filename=DIFF_REBALANCING_STOCK_DICT_ALL_PKL_L
    pd.to_pickle(stks, stock_dict_filename)

    rets_filename=DIFF_REBALANCING_RETURNS_ALL_PKL_L
    pd.to_pickle(retss, rets_filename)


    
    strategy_params = f"{DV_QUANTILE_THRESHOLD_V}_{SELECTED_TOP_VOL_STOCKS_V}_{SELECTED_MOM_WINDOW_V}_{SELECTED_HALF_LIFE_WINDOW_V}_{SELECTED_N_STOCK_POSITIVE_V}_{SELECTED_N_STOCK_CHOSE_V}_{EXP_WEIGHT_V}_{MULT_V}_{WEIGHT_V}__{DV_QUANTILE_THRESHOLD}_{SELECTED_TOP_VOL_STOCKS}_{SELECTED_MOM_WINDOW}_{SELECTED_HALF_LIFE_WINDOW}_{SELECTED_N_STOCK_POSITIVE}"
    pd.to_pickle(picasodfs, f"{DIFF_REBALANCING_PICASSO_DIR_L}/picasodfs_{strategy_params}.pkl")
    pd.to_pickle(stks, f"{DIFF_REBALANCING_PICASSO_DIR_L}/stks_{strategy_params}.pkl")
    pd.to_pickle(retss, f"{DIFF_REBALANCING_PICASSO_DIR_L}/retss_{strategy_params}.pkl")
