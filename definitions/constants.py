from datetime import datetime, timedelta
import os

# General configuration
# for L1
N_JOBS = 30
FOR_LIVE=False

USE_RAY = False

"""
If USE_RAY is True, Make sure of the following (assuming 2202 head node):

sudo scp -r -P 2204 /home/iyad/V1_DIR iyad@ws01.zapto.org:/home/iyad/ || and the same for 2203, in order to ensure the data is present on each node

____

In head node currently 2202:

env: M1_SKELETON_ENV_RAY 

To start: ray start --head --port=6318 --redis-password='password' --num-cpus=28

To check status: ray status

To stop: ray stop --force

____

In worker node 2204:

env: M1_SKELETON_ENV_RAY 

To start: ray start --address='10.0.0.110:6318' --redis-password='password' --num-cpus=28

To check status: ray status

To stop: ray stop --force
"""


# Date range for analysis, last year of this range would be one less than the YEARSTOCKS below
YEARS = range(2004, 2025)#2004

# For Data Downloading Script
# API Key
API_KEY = '7J0OvsRHy0svTgU30022h58Y04nPds2s'

# Date range
START_DATE_GET_TICKERS = '2005-01-01'
START_DATE_DATA_DOWNLOAD = '2005-01-01'
END_DATE_DATA_DOWNLOAD = '2025-02-16'

START_DATE_GET_TICKERS_AND_DATA_DOWNLOAD_FOR_LIVE='2022-01-01'
END_DATE_FOR_LIVE=(datetime.now() + timedelta(days=10)).strftime('%Y-%m-%d')

# Chunk size for processing tickers
CHUNK_SIZE = 50

# Bond-related constants
BOND_TICKERS = [
    "TLT", "IEF", "LQD", "AGG", "DBC", "GSG", "GLD", "BND", "VNQ", "HYG", "EFA", "SLV", "UCO", "DBA",
    "ETHA", "ARKB"
]

YEARSTOCKS={key: [] for key in [2005+1, 2006+1, 2007+1, 2008+1,2009+1, 2010+1,2011+1, 2012+1, 2013+1, 2014+1,
2015+1, 2016+1, 2017+1, 2018+1, 2019+1, 2020+1, 2021+1, 2022+1, 2023+1, 2024+1]}#

# Root directories
ROOT_DIR = '/home/iyad/S1_DIR'
ROOT_DIR_V = '/home/iyad/V1_DIR'
DATA_DIR = os.path.join(ROOT_DIR, 'data')

# Paths for raw stock data
STOCKS_DATA_RAW_PKL = '/mnt/spare8tb/all_dataframes_fasih_2203.pkl'#'/mnt/spare8tb/all_dataframes_sequential.pkl'
STOCKS_DATA_RAW_LIVE_PKL = '/mnt/spare8tb/all_dataframes_sequential_for_live.pkl'

# single run dirs
SINGLE_RUN_DIR = os.path.join(DATA_DIR, 'single_run')
SINGLE_RUN_BONDS_DIR = os.path.join(SINGLE_RUN_DIR, 'bonds')
SINGLE_RUN_STOCKS_DIR = os.path.join(SINGLE_RUN_DIR, 'stocks')
SINGLE_RUN_COMBINED_DIR = os.path.join(SINGLE_RUN_DIR, 'combined')

# diff rebalancing dirs
DIFF_REBALANCING_DIR = os.path.join(DATA_DIR, 'different_rebalancing_dates')
DIFF_REBALANCING_BONDS_DIR = os.path.join(DIFF_REBALANCING_DIR, 'bonds')
DIFF_REBALANCING_STOCKS_DIR = os.path.join(DIFF_REBALANCING_DIR, 'stocks')
DIFF_REBALANCING_COMBINED_DIR = os.path.join(DIFF_REBALANCING_DIR, 'combined')
DIFF_REBALANCING_PICASSO_DIR = os.path.join(DIFF_REBALANCING_DIR, 'picasso')

# sensitivity dirs
SENSITIVITY_DIR = os.path.join(DATA_DIR, 'sensitivity_analysis')
SENSITIVITY_BONDS_DIR = os.path.join(SENSITIVITY_DIR, 'bonds')
SENSITIVITY_STOCKS_DIR = os.path.join(SENSITIVITY_DIR, 'stocks')
SENSITIVITY_COMBINED_DIR = os.path.join(SENSITIVITY_DIR, 'combined')

# GS dirs
GS_DIR = os.path.join(DATA_DIR, 'grid_search')
GS_BONDS_DIR = os.path.join(GS_DIR, 'bonds')
GS_STOCKS_DIR = os.path.join(GS_DIR, 'stocks')
GS_COMBINED_DIR = os.path.join(GS_DIR, 'combined')

# Create directories if they do not exist
dirs_to_create = [
    SINGLE_RUN_DIR, SINGLE_RUN_BONDS_DIR, SINGLE_RUN_STOCKS_DIR, SINGLE_RUN_COMBINED_DIR,
    DIFF_REBALANCING_DIR, DIFF_REBALANCING_BONDS_DIR, DIFF_REBALANCING_STOCKS_DIR, DIFF_REBALANCING_COMBINED_DIR, DIFF_REBALANCING_PICASSO_DIR,
    SENSITIVITY_DIR, SENSITIVITY_BONDS_DIR, SENSITIVITY_STOCKS_DIR, SENSITIVITY_COMBINED_DIR,
    GS_DIR, GS_BONDS_DIR, GS_STOCKS_DIR, GS_COMBINED_DIR
]

for directory in dirs_to_create:
    if not os.path.exists(directory):
        os.makedirs(directory)

# single run
# Paths for returns and stock dicts and year stocks in single run
SINGLE_RUN_LIVE_STOCK_DICT_PKL = os.path.join(SINGLE_RUN_DIR, 'stock_dict_for_live.pkl')
SINGLE_RUN_STOCK_DICT_PKL = os.path.join(SINGLE_RUN_DIR, 'stock_dict.pkl')

SINGLE_RUN_YEARSTOCKS_PKL = os.path.join(SINGLE_RUN_DIR, 'stockstobeused1.pkl')
SINGLE_RUN_YEARSTOCKS_LIVE_PKL = os.path.join(SINGLE_RUN_DIR, 'stockstobeused1_for_live.pkl')

SINGLE_RUN_RETURNS_PKL = os.path.join(SINGLE_RUN_DIR, 'returns.pkl')
SINGLE_RUN_LIVE_RETURNS_PKL = os.path.join(SINGLE_RUN_DIR, 'returns_for_live.pkl')

# Paths for bond data in single run
SINGLE_RUN_BONDS_DATA_RAW_PKL = os.path.join(SINGLE_RUN_BONDS_DIR, 'bonds_data.pkl')
SINGLE_RUN_BONDS_DATA_RAW_LIVE_PKL = os.path.join(SINGLE_RUN_BONDS_DIR, 'bonds_data_for_live.pkl')
SINGLE_RUN_BONDS_DATA_ENRICHED_CSV = os.path.join(SINGLE_RUN_BONDS_DIR, 'dummy1_return_bonds.csv')
SINGLE_RUN_BONDS_DATA_ENRICHED_LIVE_CSV = os.path.join(SINGLE_RUN_BONDS_DIR, 'dummy1_return_bonds_for_live.csv')

# Paths for stock data in single run
SINGLE_RUN_STOCKS_DATA_ENRICHED_CSV = os.path.join(SINGLE_RUN_STOCKS_DIR, 'dummy1_final.csv')
SINGLE_RUN_STOCKS_DATA_ENRICHED_LIVE_CSV = os.path.join(SINGLE_RUN_STOCKS_DIR, 'dummy1_final_for_live.csv')

# Paths for combined data in single run
SINGLE_RUN_COMBINED_DATA_CSV = os.path.join(SINGLE_RUN_COMBINED_DIR, 'dummy1_return_bonds_stocks_forPicasso.csv')
SINGLE_RUN_COMBINED_DATA_LIVE_CSV = os.path.join(SINGLE_RUN_COMBINED_DIR, 'dummy1_return_bonds_stocks_forPicasso_for_live.csv')

# Different rebalancing dates

# Paths for returns and stock dicts and year stocks in bonds
DIFF_REBALANCING_STOCK_DICT_PKL = os.path.join(DIFF_REBALANCING_DIR, 'stock_dict') # + str(number) + ".pkl" to be appended in file 
DIFF_REBALANCING_STOCK_DICT_ALL_PKL = os.path.join(DIFF_REBALANCING_DIR, 'stock_dict.pkl')

DIFF_REBALANCING_RETURNS_PKL = os.path.join(DIFF_REBALANCING_DIR, 'returns') # + str(number) + ".pkl" to be appended in file
DIFF_REBALANCING_RETURNS_ALL_PKL = os.path.join(DIFF_REBALANCING_DIR, 'returns.pkl')

# Paths for combined data in different rebalancing dates
DIFF_REBALANCING_COMBINED_DATA_CSV = os.path.join(DIFF_REBALANCING_COMBINED_DIR, 'dummy1_return_bonds_stocks_forPicasso') # + str(number) + ".csv" to be appended in file
DIFF_REBALANCING_COMBINED_DATA_ALL_PKL = os.path.join(DIFF_REBALANCING_COMBINED_DIR, 'dummy1_return_bonds_stocks_forPicasso.pkl')

# Paths for bond data in different rebalancing dates
DIFF_REBALANCING_BONDS_DATA_RAW_PKL = os.path.join(DIFF_REBALANCING_BONDS_DIR, 'bonds_data.pkl')
DIFF_REBALANCING_BONDS_DATA_ENRICHED_CSV = os.path.join(DIFF_REBALANCING_BONDS_DIR, 'dummy1_return_bonds') # + str(number) + ".csv" to be appended in file

# Paths for stock data in different rebalancing dates
DIFF_REBALANCING_STOCKS_DATA_ENRICHED_CSV = os.path.join(DIFF_REBALANCING_STOCKS_DIR, 'dummy1_final') # + str(number) + ".csv" to be appended in file


# Paths for sensitivity analysis
SENSITIVITY_BONDS_DATA_ENRICHED_CSV = os.path.join(SENSITIVITY_BONDS_DIR, 'dummy1_return_bonds.csv')
SENSITIVITY_BONDS_DATA_RAW_PKL = os.path.join(SENSITIVITY_BONDS_DIR, 'bonds_data.pkl')

SENSITIVITY_STOCKS_DATA_ENRICHED_CSV = os.path.join(SENSITIVITY_STOCKS_DIR, 'dummy1_final.csv')
SENSITIVITY_COMBINED_DATA_CSV = os.path.join(SENSITIVITY_COMBINED_DIR, 'dummy1_return_bonds_stocks_forPicasso.csv')

# Paths for grid search
GS_ITS=os.path.join(GS_DIR, 'its.pkl')
GS_RES=os.path.join(GS_DIR,'gs_df.csv')

GS_BONDS_DATA_ENRICHED_CSV = os.path.join(GS_BONDS_DIR, 'dummy1_return_bonds.csv')
GS_BONDS_DATA_RAW_PKL = os.path.join(GS_BONDS_DIR, 'bonds_data.pkl')
GS_STOCKS_DATA_ENRICHED_CSV = os.path.join(GS_STOCKS_DIR, 'dummy1_final.csv')


DV_QUANTILE_THRESHOLD_MAKE_YS=[0.05, 0.2, 0.5, 0.66, 0.75]
SELECTED_TOP_VOL_STOCKS_MAKE_YS=[11, 20, 27, 75, 150]#, 20, 27, 35, 40, 50, 75, 100, 125, 150, 200

# GS
DATE_GS_CUTOFF='2018-01-01'
MOMENTUM_WINDOWS_GS = [30, 63, 90, 126, 150, 200, 252, 504]#
HALF_LIVES_GS = [30, 63, 90, 126, 150, 200, 250]#
SELECTED_N_STOCK_POSITIVE_GS=[1,3,5,7,11,15,25,50,75,100]#
SELECTED_N_STOCK_CHOSE_GS=[1,3,5,7,10,16,21,25,30,50]#
EXP_WEIGHT_GS=[0.5, 0.7, 0.85, 0.95, 0.9999999999]

#  SENS
DATES_SENS=[DATE_GS_CUTOFF, '2025-01-01']
DV_QUANTILE_THRESHOLD_SENS=[0.05, 0.2, 0.33, 0.5, 0.66, 0.75, 0.9]#
SELECTED_TOP_VOL_STOCKS_SENS=[11, 20, 27, 35, 40, 50, 75, 100, 125, 150, 200]#10,  
MOMENTUM_WINDOWS_SENS=[[30], [63], [90], [126], [150], [200], [252], [504]]#
HALF_LIVES_SENS=[[30], [63], [90], [126], [150], [200], [250]]#
SELECTED_N_STOCK_POSITIVE_SENS=[7, 8, 9, 10, 11]#30, 37, 45, 50, 55, 60
SELECTED_N_STOCK_CHOSE_SENS=[1, 2, 3, 4, 5, 6, 7, 8, 9]#, 13, 15, 20, 
EXP_WEIGHT_SENS=[0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.9999999999]

SENS_PARAMETER_CONFIG = {
    'dvqt': {
        'anrets': ('Annual Returns', 'Annual Return', 'blue'),
        'sharpes': ('Sharpe Ratios', 'Sharpe Ratio', 'green'),
        'maxdraws': ('Maximum Drawdowns', 'Max Drawdown', 'red'),
        'calmars': ('Calmar Ratios', 'Calmar Ratio', 'cyan'),
        'sortinos': ('Sortino Ratios', 'Sortino Ratio', 'magenta')
    },
    'topvolstocks': {
        'anrets': ('Annual Returns', 'Annual Return', 'blue'),
        'sharpes': ('Sharpe Ratios', 'Sharpe Ratio', 'green'),
        'maxdraws': ('Maximum Drawdowns', 'Max Drawdown', 'red'),
        'calmars': ('Calmar Ratios', 'Calmar Ratio', 'cyan'),
        'sortinos': ('Sortino Ratios', 'Sortino Ratio', 'magenta')
    },
    'exp_weight': {
        'anrets': ('Annual Returns', 'Annual Return', 'blue'),
        'sharpes': ('Sharpe Ratios', 'Sharpe Ratio', 'green'),
        'maxdraws': ('Maximum Drawdowns', 'Max Drawdown', 'red'),
        'calmars': ('Calmar Ratios', 'Calmar Ratio', 'cyan'),
        'sortinos': ('Sortino Ratios', 'Sortino Ratio', 'magenta')
    },
    'n_stock_chose': {
        'anrets': ('Annual Returns', 'Annual Return', 'blue'),
        'sharpes': ('Sharpe Ratios', 'Sharpe Ratio', 'green'),
        'maxdraws': ('Maximum Drawdowns', 'Max Drawdown', 'red'),
        'calmars': ('Calmar Ratios', 'Calmar Ratio', 'cyan'),
        'sortinos': ('Sortino Ratios', 'Sortino Ratio', 'magenta')
    },
    'n_stock_positive': {
        'anrets': ('Annual Returns', 'Annual Return', 'blue'),
        'sharpes': ('Sharpe Ratios', 'Sharpe Ratio', 'green'),
        'maxdraws': ('Maximum Drawdowns', 'Max Drawdown', 'red'),
        'calmars': ('Calmar Ratios', 'Calmar Ratio', 'cyan'),
        'sortinos': ('Sortino Ratios', 'Sortino Ratio', 'magenta')
    },
    'mom_window': {
        'anrets': ('Annual Returns (Momentum Window)', 'Annual Return', 'blue'),
        'sharpes': ('Sharpe Ratios (Momentum Window)', 'Sharpe Ratio', 'green'),
        'maxdraws': ('Max Drawdowns (Momentum Window)', 'Max Drawdown', 'red'),
        'calmars': ('Calmar Ratios (Momentum Window)', 'Calmar Ratio', 'cyan'),
        'sortinos': ('Sortino Ratios (Momentum Window)', 'Sortino Ratio', 'magenta')
    },
    'half_life': {
        'anrets': ('Annual Returns (Half Life)', 'Annual Return', 'blue'),
        'sharpes': ('Sharpe Ratios (Half Life)', 'Sharpe Ratio', 'green'),
        'maxdraws': ('Max Drawdowns (Half Life)', 'Max Drawdown', 'red'),
        'calmars': ('Calmar Ratios (Half Life)', 'Calmar Ratio', 'cyan'),
        'sortinos': ('Sortino Ratios (Half Life)', 'Sortino Ratio', 'magenta')
    }
}

# Exclusions and thresholds
TICKERS_TO_EXCLUDE = BOND_TICKERS
DV_QUANTILE_THRESHOLD = 0.66
LEN_YEARS_DV_LOOKBACK = 2

# Momentum and half-life settings
MOMENTUM_WINDOWS = [252] #30, 63, 90, 126, 150, 200, 252, 504
HALF_LIVES = [126] #30, 42, 63, 90, 126, 150, 200, 250

# Selection criteria
SELECTED_TOP_VOL_STOCKS = 11
SELECTED_MOM_WINDOW = 252
SELECTED_HALF_LIFE_WINDOW = 126
SELECTED_N_STOCK_POSITIVE = 0
SELECTED_N_STOCK_CHOSE = 1
EXP_WEIGHT = 0.5