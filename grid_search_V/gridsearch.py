import pandas as pd
import pickle
import sys
import os
from joblib import Parallel, delayed
from datetime import timedelta
import random
import time
import ray
from definitions.constants import USE_RAY

if USE_RAY:
    # Update runtime_env to INCLUDE the files
    runtime_env = {
        "working_dir": "/home/iyad/V1_DIR/V1",
        # "includes": [
        #     "/home/iyad/V1_DIR/data_V/grid_search/combined"
        # ]
    }

    # Initialize Ray with the runtime environment
    ray.init(_redis_password='password', runtime_env=runtime_env)

# Append the project root to sys.path
project_root = os.path.dirname(os.path.abspath(__file__))  # Current file directory
project_root = os.path.join(project_root, '..')  # Move one level up to the root
sys.path.append(project_root)

# Import constants and custom utility functions
from definitions.constants import GS_COMBINED_DIR, DATE_GS_CUTOFF, N_JOBS
from definitions.constants_V import GS_COMBINED_DIR_V, GS_ITS_V, GS_RES_V

from utils.custom import (
    calculate_stock_selection_V,
    exponential_weights, calculate_returns_V, calculate_metrics
)

def run_gs(*, qt, topnStocksByVolume, stockPos, topnStockSelect, momentum_windows, half_lives, exp_weight, mult, weight, m2):
    file_path_v = os.path.join(GS_COMBINED_DIR_V, f"{qt}_{topnStocksByVolume}.csv")
    file_path = os.path.join(GS_COMBINED_DIR, f"{m2[0]}_{m2[1]}.csv")

    # print(f"Reading file: {file_path_v}")
    # print(f"Reading file: {file_path}")
    
    df=pd.read_csv(file_path_v)

    df_M=pd.read_csv(file_path)

    df = df[
        ['Date', f'Momentum_{momentum_windows}_{half_lives}_{mult}_{weight}', 'Stock', 'Returns']
    ].sort_values('Date').reset_index(drop=True)

    df_M = df_M[
        ['Date', f'Momentum_{m2[2]}_{m2[3]}', 'Stock', 'Returns']
    ].sort_values('Date').reset_index(drop=True)

    df=df[df.Date<=DATE_GS_CUTOFF]
    df_M=df_M[df_M.Date<=DATE_GS_CUTOFF]

    # Step 3: Calculate stock selection based on momentum metrics
    stock_dict = calculate_stock_selection_V(df = df, df_M = df_M, SELECTED_MOM_WINDOW=momentum_windows, SELECTED_HALF_LIFE_WINDOW=half_lives, SELECTED_MULT=mult, SELECTED_WEIGHT=weight, SELECTED_N_STOCK_POSITIVE=stockPos, SELECTED_N_STOCK_CHOSE=topnStockSelect, SELECTED_MOM_WINDOW_M=m2[2], SELECTED_HALF_LIFE_WINDOW_M=m2[3], SELECTED_N_STOCK_POSITIVE_M=m2[4])

    # Step 4: Determine portfolio weights
    # Exponential weights are used if EXP_WEIGHT is True, otherwise a near-uniform weight is applied.
    weights = exponential_weights(
        length = topnStockSelect, 
        alpha= exp_weight
    )

    # Step 5: Calculate portfolio returns
    returns = calculate_returns_V(stock_dict=stock_dict, df=df, weights=weights, mom=momentum_windows, half=half_lives, mult=mult, w=weight)

    annual_return, sharpe_ratio, max_drawdown_value, calmar_ratio, sortino_ratio = calculate_metrics(returns = returns)

    return annual_return, sharpe_ratio, max_drawdown_value, calmar_ratio, sortino_ratio


def run_iteration(it):
    a, s, m, c, ss = run_gs(qt=it[0], topnStocksByVolume=it[1], stockPos=it[2], topnStockSelect=it[3], momentum_windows=it[4], half_lives=it[5], exp_weight=it[6], mult=it[7], weight=it[8], m2=it[9])
    return {
        str(it): [a, s, m, c, ss],
    }

if USE_RAY:
    # Wrap the function with Ray's remote decorator
    @ray.remote
    def run_iteration_remote(*args, **kwargs):
        return run_iteration(*args, **kwargs)


if __name__ == "__main__":
    """
    Script entry point.

    This script processes bond data by applying momentum and exponential weighting
    calculations, and saves the enriched data to a CSV file.
    """
    its=pd.read_pickle(GS_ITS_V)
    
    # Shuffle the iterations
    random.shuffle(its)

    # Split the iterations into 10 chunks
    chunk_size = len(its) // 10
    chunks = [its[i:i + chunk_size] for i in range(0, len(its), chunk_size)]

    # Ensure no leftover iterations are missed
    if len(its) % 10 != 0:
        chunks[-1].extend(its[10 * chunk_size:])

    times=[]

    for chunk_number, chunk in enumerate(chunks, start=1):
        start_time = time.time()

        if USE_RAY:
            remote_functions = [
                run_iteration_remote.remote(it)
                for it in chunk
            ]
            # Fetch results from Ray
            results = ray.get(remote_functions)
        else:
            results = Parallel(n_jobs=N_JOBS)(delayed(run_iteration)(it) for it in chunk)

        end_time = time.time()  # End timing
        elapsed_time = end_time - start_time  # Calculate elapsed time
        times.append(elapsed_time)

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
            'qt', 'topnStocksByVolume', 'nStockspos_plus_topnStockSelect', 'topnStockSelect', 'momentum_windows', 'half_lives', 'exp_weight', 'mult', 'weight', 'm2',
            'annual_returns', 'sharpe_ratios', 'max_drawdown_values', 'calmar_ratios', 'sortino_ratios'
        ]

        # Creating the DataFrame
        df = pd.DataFrame(rows, columns=columns)
        df.to_csv(GS_RES_V + f"_{chunk_number}.csv", index=False)

    
    print(f"Script execution time: {times} seconds")
    if USE_RAY:
        ray.shutdown()

