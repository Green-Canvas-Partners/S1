from datetime import datetime, timedelta
import os
import pandas as pd
# Root directories
ROOT_DIR = '/home/iyad/S1_DIR'

DATA_DIR_L = os.path.join(ROOT_DIR, 'data_S')


SINGLE_RUN_DIR_L = os.path.join(DATA_DIR_L, 'single_run')
SINGLE_RUN_BONDS_DIR_L = os.path.join(SINGLE_RUN_DIR_L, 'bonds')
SINGLE_RUN_STOCKS_DIR_L = os.path.join(SINGLE_RUN_DIR_L, 'stocks')
SINGLE_RUN_COMBINED_DIR_L = os.path.join(SINGLE_RUN_DIR_L, 'combined')


# diff rebalancing dirs
DIFF_REBALANCING_DIR_L = os.path.join(DATA_DIR_L, 'different_rebalancing_dates')
DIFF_REBALANCING_BONDS_DIR_L = os.path.join(DIFF_REBALANCING_DIR_L, 'bonds')
DIFF_REBALANCING_STOCKS_DIR_L = os.path.join(DIFF_REBALANCING_DIR_L, 'stocks')
DIFF_REBALANCING_COMBINED_DIR_L = os.path.join(DIFF_REBALANCING_DIR_L, 'combined')
DIFF_REBALANCING_PICASSO_DIR_L = os.path.join(DIFF_REBALANCING_DIR_L, 'picasso')

# sensitivity dirs
SENSITIVITY_DIR_L = os.path.join(DATA_DIR_L, 'sensitivity_analysis')
SENSITIVITY_BONDS_DIR_L = os.path.join(SENSITIVITY_DIR_L, 'bonds')
SENSITIVITY_STOCKS_DIR_L = os.path.join(SENSITIVITY_DIR_L, 'stocks')
SENSITIVITY_COMBINED_DIR_L = os.path.join(SENSITIVITY_DIR_L, 'combined')

# GS dirs
GS_DIR_L = os.path.join(DATA_DIR_L, 'grid_search')
GS_BONDS_DIR_L = os.path.join(GS_DIR_L, 'bonds')
GS_STOCKS_DIR_L = os.path.join(GS_DIR_L, 'stocks')
GS_COMBINED_DIR_L = os.path.join(GS_DIR_L, 'combined')

# Create directories if they do not exist
dirs_to_create = [
    SINGLE_RUN_DIR_L, SINGLE_RUN_BONDS_DIR_L, SINGLE_RUN_STOCKS_DIR_L, SINGLE_RUN_COMBINED_DIR_L,
    DIFF_REBALANCING_DIR_L, DIFF_REBALANCING_BONDS_DIR_L, DIFF_REBALANCING_STOCKS_DIR_L, DIFF_REBALANCING_COMBINED_DIR_L, DIFF_REBALANCING_PICASSO_DIR_L,
    SENSITIVITY_DIR_L, SENSITIVITY_BONDS_DIR_L, SENSITIVITY_STOCKS_DIR_L, SENSITIVITY_COMBINED_DIR_L,
    GS_DIR_L, GS_BONDS_DIR_L, GS_STOCKS_DIR_L, GS_COMBINED_DIR_L
]

for directory in dirs_to_create:
    if not os.path.exists(directory):
        os.makedirs(directory)

# single run
# Paths for returns and stock dicts and year stocks in single run
SINGLE_RUN_LIVE_STOCK_DICT_PKL_L = os.path.join(SINGLE_RUN_DIR_L, 'stock_dict_for_live.pkl')
SINGLE_RUN_STOCK_DICT_PKL_L = os.path.join(SINGLE_RUN_DIR_L, 'stock_dict.pkl')
SINGLE_RUN_YEARSTOCKS_PKL_L = os.path.join(SINGLE_RUN_DIR_L, 'stockstobeused1.pkl')
SINGLE_RUN_YEARSTOCKS_LIVE_PKL_L = os.path.join(SINGLE_RUN_DIR_L, 'stockstobeused1_for_live.pkl')
SINGLE_RUN_RETURNS_PKL_L = os.path.join(SINGLE_RUN_DIR_L, 'returns.pkl')
SINGLE_RUN_LIVE_RETURNS_PKL_L = os.path.join(SINGLE_RUN_DIR_L, 'returns_for_live.pkl')


# Paths for bond data in single run
SINGLE_RUN_BONDS_DATA_RAW_PKL_L = os.path.join(SINGLE_RUN_BONDS_DIR_L, 'bonds_data.pkl')
SINGLE_RUN_BONDS_DATA_RAW_LIVE_PKL_L = os.path.join(SINGLE_RUN_BONDS_DIR_L, 'bonds_data_for_live.pkl')
SINGLE_RUN_BONDS_DATA_ENRICHED_CSV_L = os.path.join(SINGLE_RUN_BONDS_DIR_L, 'dummy1_return_bonds.csv')
SINGLE_RUN_BONDS_DATA_ENRICHED_LIVE_CSV_L = os.path.join(SINGLE_RUN_BONDS_DIR_L, 'dummy1_return_bonds_for_live.csv')

# Paths for stock data in single run
SINGLE_RUN_STOCKS_DATA_ENRICHED_CSV_L = os.path.join(SINGLE_RUN_STOCKS_DIR_L, 'dummy1_final.csv')
SINGLE_RUN_STOCKS_DATA_ENRICHED_LIVE_CSV_L = os.path.join(SINGLE_RUN_STOCKS_DIR_L, 'dummy1_final_for_live.csv')

# Paths for combined data in single run
SINGLE_RUN_COMBINED_DATA_CSV_L = os.path.join(SINGLE_RUN_COMBINED_DIR_L, 'dummy1_return_bonds_stocks_forPicasso.csv')
SINGLE_RUN_COMBINED_DATA_LIVE_CSV_L = os.path.join(SINGLE_RUN_COMBINED_DIR_L, 'dummy1_return_bonds_stocks_forPicasso_for_live.csv')

# Different rebalancing dates

# Paths for returns and stock dicts and year stocks in bonds
DIFF_REBALANCING_STOCK_DICT_PKL_L = os.path.join(DIFF_REBALANCING_DIR_L, 'stock_dict') # + str(number) + ".pkl" to be appended in file 
DIFF_REBALANCING_STOCK_DICT_ALL_PKL_L = os.path.join(DIFF_REBALANCING_DIR_L, 'stock_dict.pkl')

DIFF_REBALANCING_RETURNS_PKL_L = os.path.join(DIFF_REBALANCING_DIR_L, 'returns') # + str(number) + ".pkl" to be appended in file
DIFF_REBALANCING_RETURNS_ALL_PKL_L = os.path.join(DIFF_REBALANCING_DIR_L, 'returns.pkl')

# Paths for combined data in different rebalancing dates
DIFF_REBALANCING_COMBINED_DATA_CSV_L = os.path.join(DIFF_REBALANCING_COMBINED_DIR_L, 'dummy1_return_bonds_stocks_forPicasso') # + str(number) + ".csv" to be appended in file
DIFF_REBALANCING_COMBINED_DATA_ALL_PKL_L = os.path.join(DIFF_REBALANCING_COMBINED_DIR_L, 'dummy1_return_bonds_stocks_forPicasso.pkl')

# Paths for bond data in different rebalancing dates
DIFF_REBALANCING_BONDS_DATA_RAW_PKL_L = os.path.join(DIFF_REBALANCING_BONDS_DIR_L, 'bonds_data.pkl')
DIFF_REBALANCING_BONDS_DATA_ENRICHED_CSV_L = os.path.join(DIFF_REBALANCING_BONDS_DIR_L, 'dummy1_return_bonds') # + str(number) + ".csv" to be appended in file

# Paths for stock data in different rebalancing dates
DIFF_REBALANCING_STOCKS_DATA_ENRICHED_CSV_L = os.path.join(DIFF_REBALANCING_STOCKS_DIR_L, 'dummy1_final') # + str(number) + ".csv" to be appended in file


# Paths for sensitivity analysis
SENSITIVITY_BONDS_DATA_ENRICHED_CSV_L = os.path.join(SENSITIVITY_BONDS_DIR_L, 'dummy1_return_bonds.csv')
SENSITIVITY_BONDS_DATA_RAW_PKL_L = os.path.join(SENSITIVITY_BONDS_DIR_L, 'bonds_data.pkl')

SENSITIVITY_STOCKS_DATA_ENRICHED_CSV_L = os.path.join(SENSITIVITY_STOCKS_DIR_L, 'dummy1_final.csv')
SENSITIVITY_COMBINED_DATA_CSV_L = os.path.join(SENSITIVITY_COMBINED_DIR_L, 'dummy1_return_bonds_stocks_forPicasso.csv')

# Paths for grid search
GS_ITS_L=os.path.join(GS_DIR_L, 'its.pkl')
GS_RES_L=os.path.join(GS_DIR_L,'gs_df')

GS_BONDS_DATA_ENRICHED_CSV_L = os.path.join(GS_BONDS_DIR_L, 'dummy1_return_bonds.csv')
GS_BONDS_DATA_RAW_PKL_L = os.path.join(GS_BONDS_DIR_L, 'bonds_data.pkl')
GS_STOCKS_DATA_ENRICHED_CSV_L = os.path.join(GS_STOCKS_DIR_L, 'dummy1_final.csv')


SHARE_OUTSTANDING_DATA_250_TESTING_PKL_L = os.path.join(DATA_DIR_L, 'shares_outstanding_data_250_testing.pkl')
SHARES_OUTSTANDING_DATA_250_UPDATED_PKL_L = os.path.join(DATA_DIR_L, 'updated_shares_outstanding_data_250.pkl')

EPS_DATA_TESTING_PKL_L = os.path.join(DATA_DIR_L, 'eps_data_testing.pkl')
EPS_DATA_UPDATED_PKL_L = os.path.join(DATA_DIR_L, 'updated_eps_data.pkl')

DIVIDEND_DATA_TESTING_PKL_L = os.path.join(DATA_DIR_L, 'dividend_data_testing.pkl')
DIVIDEND_DATA_UPDATED_PKL_L = os.path.join(DATA_DIR_L, 'updated_dividend_data.pkl')

EQUITY_DATA_TESTING_PKL_L = os.path.join(DATA_DIR_L, 'equity_data_testing.pkl')
EQUITY_DATA_UPDATED_PKL_L = os.path.join(DATA_DIR_L, 'updated_equity_data.pkl')


DV_QUANTILE_THRESHOLD_MAKE_YS_V=[0.05, 0.2, 0.33, 0.5, 0.66, 0.75, 0.9]
SELECTED_TOP_VOL_STOCKS_MAKE_YS_V=[11, 13, 20, 27, 35, 40, 50, 75, 100, 125, 150, 200]#

# GS
DATE_GS_CUTOFF='2018-01-01'
MOMENTUM_WINDOWS_GS_V = [63, 90, 126, 200, 252, 504]#
HALF_LIVES_GS_V = [30, 42, 63, 90, 126, 150, 200]#
SELECTED_N_STOCK_POSITIVE_GS_V=[1,3,5,10,15,20,30,50]#
SELECTED_N_STOCK_CHOSE_GS_V=[1,3,7,11,16,20,30,50]#
EXP_WEIGHT_GS_V=[0.5, 0.7, 0.85, 0.99999999999]
MULT_GS_V=[1.01, 2.0]#
WEIGHT_GS_V=[0.9]#

#  SENS
DATES_SENS=[DATE_GS_CUTOFF, '2025-01-01']
DV_QUANTILE_THRESHOLD_SENS_V=[0.05, 0.2]#, 0.33, 0.5, 0.66, 0.75, 0.9
SELECTED_TOP_VOL_STOCKS_SENS_V=[11, 20]#, 27, 35, 40, 50, 75, 100, 125, 150, 200
MOMENTUM_WINDOWS_SENS_V=[[30], [63]]#, [90], [126], [150], [200], [252], [504]
HALF_LIVES_SENS_V=[[30], [63]]#, [90], [126], [150], [200], [250]
SELECTED_N_STOCK_POSITIVE_SENS_V=[7, 8, 9, 10, 11]#30, 37, 45, 50, 55, 60
SELECTED_N_STOCK_CHOSE_SENS_V=[1, 2, 3]#, 4, 5, 6, 7, 8, 9, 13, 15, 20, 
EXP_WEIGHT_SENS_V=[0.5, 0.6, 0.7]#, 0.8, 0.85, 0.9, 0.95, 0.9999999999
MULT_SENS_V=[[1.01], [1.5], [2.0]]#, 4.0, 6.0
WEIGHT_SENS_V=[[0.1],[0.5], [0.9]]#

SENS_PARAMETER_CONFIG_V = {
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
    },
    'mult': {
        'anrets': ('Annual Returns (Multiplier)', 'Annual Return', 'blue'),
        'sharpes': ('Sharpe Ratios (Multiplier)', 'Sharpe Ratio', 'green'),
        'maxdraws': ('Max Drawdowns (Multiplier)', 'Max Drawdown', 'red'),
        'calmars': ('Calmar Ratios (Multiplier)', 'Calmar Ratio', 'cyan'),
        'sortinos': ('Sortino Ratios (Multiplier)', 'Sortino Ratio', 'magenta')
    },
    'weight': {
        'anrets': ('Annual Returns (Weight)', 'Annual Return', 'blue'),
        'sharpes': ('Sharpe Ratios (Weight)', 'Sharpe Ratio', 'green'),
        'maxdraws': ('Max Drawdowns (Weight)', 'Max Drawdown', 'red'),
        'calmars': ('Calmar Ratios (Weight)', 'Calmar Ratio', 'cyan'),
        'sortinos': ('Sortino Ratios (Weight)', 'Sortino Ratio', 'magenta')
    }
}

# Exclusions and thresholds
DV_QUANTILE_THRESHOLD_V = 0.75

# Momentum and half-life settings
MOMENTUM_WINDOWS_V = [60, 90] #30, 63, 90, 126, 150, 200, 252, 504
HALF_LIVES_V = [60] #30, 42, 63, 90, 126, 150, 200, 250
# MULT_V = [1.01, 1.5, 2.0, 4.0] #1.01, 1.5, 2.0, 4.0, 6.0
# WEIGHT_V = [0.1, 0.5, 0.9] #0.1, 0.5, 0.9
MULT_V = [1.01] #1.01, 1.5, 2.0, 4.0, 6.0
WEIGHT_V = [0.9] #0.1, 0.5, 0.9

# Selection criteria
SELECTED_TOP_VOL_STOCKS_V = 16
SELECTED_MOM_WINDOW_V = 60
SELECTED_HALF_LIFE_WINDOW_V = 60
SELECTED_N_STOCK_POSITIVE_V = 8
SELECTED_N_STOCK_CHOSE_V = 6
EXP_WEIGHT_V = 0.5
SELECTED_MULT_V = 1.01
SELECTED_WEIGHT_V = 0.9

L1_TICKERS = ['AA', 'AAL', 'AAPL', 'ABBV', 'ABC', 'ABK', 'ABNB', 'ABT', 'ABX', 'ACI', 'ACN', 'ACWI', 'ADBE', 'ADI', 'ADM', 'ADP', 'ADSK', 'AEM', 'AET', 'AFL', 'AFRM', 'AGN', 'AGNC', 'AGQ', 'AGU', 'AI', 'AIG', 'AKAM', 'ALB', 'ALGN', 'ALL', 'ALTR', 'ALXN', 'AMAT', 'AMC', 'AMD', 'AMGN', 'AMR', 'AMT', 'AMTD', 'AMX', 'AMZN', 'ANET', 'ANF', 'ANR', 'ANTM', 'AON', 'APA', 'APC', 'APD', 'APOL', 'ARKK', 'ASML', 'AT', 'ATI', 'ATVI', 'AUY', 'AVGO', 'AXP', 'AZN', 'AZO', 'BA', 'BABA', 'BAC', 'BAX', 'BBBY', 'BBD', 'BBT', 'BBY', 'BDX', 'BEN', 'BG', 'BGU', 'BHI', 'BHP', 'BIDU', 'BIIB', 'BIL', 'BILI', 'BJS', 'BK', 'BKNG', 'BLK', 'BMY', 'BNI', 'BNTX', 'BP',
               'BRCM', 'BRK.A', 'BRK.B', 'BSC', 'BSX', 'BTU', 'BUD', 'BX', 'BXP', 'C', 'CAL', 'CAT', 'CB', 'CBS', 'CCI', 'CCL', 'CCU', 'CDNS', 'CELG', 'CEPH', 'CF', 'CFC', 'CHK', 'CHTR', 'CI', 'CL', 'CLF', 'CLX', 'CMCSA', 'CMCSK', 'CME', 'CMG', 'CMI', 'CNC', 'CNQ', 'CNX', 'COF', 'COG', 'COH', 'COIN', 'COP', 'COST', 'COUP', 'COV', 'CREE', 'CRM', 'CRWD', 'CSCO', 'CSX', 'CTL', 'CTSH', 'CTX', 'CTXS', 'CVNA', 'CVS', 'CVX', 'CXO', 'D', 'DAL', 'DASH', 'DD', 'DDD', 'DDM', 'DDOG', 'DE', 'DELL', 'DFS', 'DG', 'DHR', 'DIA', 'DIG', 'DIS', 'DLTR', 'DNA', 'DO', 'DOCU', 'DOW', 'DRYS', 'DTV', 'DUG', 'DUK', 'DUST', 'DVN', 'DXCM', 'DXD', 'DXJ', 'EA', 'EBAY', 'ECA', 'EEM', 'EL', 'EMB', 'EMC',
                 'EMR', 'ENDP', 'ENPH', 'EOG', 'EP', 'EQIX', 'ERTS', 'ERX', 'ESRX', 'ESV', 'ETN', 'ETR', 'ETSY', 'EW', 'EWJ', 'EWW', 'EWY', 'EWZ', 'EXC', 'EXPE', 'EZU', 'F', 'FAS', 'FAZ', 'FB', 'FCX', 'FDC', 'FDX', 'FEYE', 'FFIV', 'FIS', 'FISV', 'FITB', 'FLR', 'FNM', 'FOXA', 'FRE', 'FSLR', 'FUTU', 'FWLT', 'FXE', 'FXI', 'GD', 'GDX', 'GDXJ', 'GE', 'GENZ', 'GG', 'GILD', 'GIS', 'GLW', 'GM', 'GMCR', 'GME', 'GOLD', 'GOOG', 'GOOGL', 'GPN', 'GPS', 'GRMN', 'GS', 'GSF', 'HAL', 'HANS', 'HCA', 'HD', 'HEDJ', 'HES', 'HIG', 'HLF', 'HON', 'HOT', 'HPE', 'HPQ', 'HTZ', 'HUM', 'IAU', 'IBB', 'IBM', 'ICE', 'IEFA', 'IEMG', 'IGV', 'IJH', 'IJR', 'ILMN', 'INCY', 'INTC', 'INTU', 'IP', 'ISRG', 'ITUB', 'ITW', 'IVV', 'IWB', 'IWD', 'IWF', 'IWM', 'IWN', 'IWO', 'IYF', 'IYR', 'JCI', 'JCP', 'JD', 'JNJ', 'JNK', 'JNPR', 'JNUG', 'JPM', 'JWN', 'KBE', 'KBH', 'KFT', 'KHC', 'KLAC', 'KMB', 'KMI', 'KO', 'KORS', 'KR', 'KRE', 'KSS', 'KWEB', 'LBTYA', 'LEN', 'LIN', 'LLTC', 'LLY', 'LM', 'LMT', 'LNG', 'LNKD', 'LOW', 'LRCX', 'LULU', 'LUV', 'LVLT', 'LVS', 'LYB', 'M', 'MA', 'MAR', 'MARA', 'MBI', 'MCD', 'MCHI', 'MCHP', 
                 'MCK', 'MCO', 'MDB', 'MDLZ', 'MDT', 'MDY', 'MEE', 'MELI', 'MER', 'MET', 'MGM', 'MHS', 'MMM', 'MO', 'MON', 'MOS', 'MOT', 'MPC', 'MRK', 'MRNA', 'MRO', 'MRVL', 'MS', 'MSFT', 'MSTR', 'MT', 'MTCH', 'MU', 'MUB', 'MXIM', 'MYL', 'NBR', 'NCC', 'NCLH', 'NE', 'NEE', 'NEM', 'NET', 'NFLX', 'NIHD', 'NIO', 'NKE', 'NLY', 'NOC', 'NOK', 'NOV', 'NOW', 'NSC', 'NSM', 'NTAP', 'NTES', 'NTRS', 'NUE', 'NUGT', 'NVAX', 'NVDA', 'NVS', 'NWSA', 'NXPI', 'NYX', 'OIH', 'OKTA', 'ON', 'ORCL', 'ORLY', 'OXY', 'P', 'PANW', 'PBR', 'PBR.A', 'PCLN', 'PCP', 'PCU', 'PD', 'PDD', 'PENN', 'PEP', 'PFE', 'PG', 'PGR', 'PHM', 'PINS', 'PLD', 'PLTR', 'PLUG', 'PM', 'PNC', 'POT', 'PRGO', 'PRU', 'PSA', 'PSQ', 'PSX', 'PTON', 'PX', 'PXD', 'PYPL', 'Q', 'QCOM', 'QID', 'QIHU', 'QLD', 'QQQ',
                   'QQQQ', 'RAI', 'RBLX', 'RCL', 'RDS.A', 'REGN', 'RF', 'RHT', 'RIG', 'RIMM', 'RIO', 'RIOT', 'RIVN', 'RKH', 'RMBS', 'RNG', 'ROKU', 'RSP', 'RSX', 'RTH', 'RTN', 'RTX', 'S', 'SBUX', 'SCHW', 'SDS', 'SE', 'SEPR', 'SGP', 'SH', 'SHLD', 'SHOP', 'SHPG', 'SHW', 'SHY', 'SII', 'SINA', 'SIRI', 'SKF', 'SLB', 'SLM', 'SLW', 'SMCI', 'SMH', 'SNAP', 'SNDK', 'SNOW', 'SNPS', 'SO', 'SOXL', 'SOXS', 'SOXX', 'SPCE', 'SPG', 'SPGI', 'SPLS', 'SPOT', 'SPXL', 'SPXS', 'SPXU', 'SPY', 'SQ', 'SQQQ', 'SRS', 'SSO', 'STI', 'STJ', 'STP', 'STT', 'STX', 'STZ', 'SU', 'SUN', 'SUNE', 'SUNW', 'SVXY', 'SWKS', 'SWN', 'SYF', 'SYK', 'SYMC', 'T', 'TBT', 'TCK', 'TDOC', 'TEAM', 'TEVA', 'TFC', 'TGT', 'TIE', 'TIP', 'TJX', 'TLRY', 'TMO', 'TMUS', 'TNA', 'TOL', 'TOT', 'TQQQ', 'TRV', 'TSLA', 'TSM', 'TSN', 'TSO', 'TTD', 'TTWO', 'TVIX', 'TWC', 'TWLO', 'TWM', 'TWTR', 'TWX', 'TXN', 'TYC', 'TZA', 'U', 'UA', 'UAL', 'UBER', 'UGAZ', 'ULTA', 'UNG', 'UNH', 'UNP', 'UPRO', 'UPS', 'UPST', 
                 'USB', 'USMV', 'USO', 'UTX', 'UVXY', 'UYG', 'V', 'VALE', 'VALE.P', 'VCIT', 'VCSH', 'VEA', 'VGK', 'VIAB', 'VIAC', 'VLO', 'VMW', 'VNO', 'VOD', 'VOO', 'VRTX', 'VRX', 'VTI', 'VTV', 'VWO', 'VXX', 'VZ', 'W', 'WAG', 'WB', 'WBA', 'WDAY', 'WDC', 'WFC', 'WFM', 'WFMI', 'WFR', 'WFT', 'WLL', 'WLP', 'WLT', 'WM', 'WMB', 'WMT', 'WY', 'WYE', 'WYNN', 'X', 'XBI', 'XHB', 'XIV', 'XLB', 'XLC', 'XLE', 'XLF', 'XLI', 'XLK', 'XLNX', 'XLP', 'XLU', 'XLV', 'XLY', 'XME', 'XOM', 'XOP', 'XRT', 'XTO', 'YELP', 'YHOO', 'YUM', 'Z', 'ZM', 'ZS', 'ZTS', 'GOLD', 'NTR', 'TFC', 'BKR', 'AVGO', 'BRK-A', 'BRK-B', 'PARA', 'CMCSA', 'CTRA', 'TPR', 'MDT', 'WOLF', 'EA', 'VAL', 'MNST', 'KHC', 'VTRS', 'ICE', 'PBR', 'BKNG', 'NTR', 'SHEL', 'BB', 'WPM', 'TECK', 'TTE', 'RTX', 'VALE', 'WBA',"META"]

API_KEY_EODHD = '672bae3e306da2.69377201'

DATA_L1_TICKERS_REFINED= {
    'Alternative/Status': [
        "ABX", "AGU", "BBT", "BHI","BRCM", "BRK.A", "BRK.B", "CBS","CMCSK", "COG", "COH", "COV", "CREE", "ERTS", "ESV","HANS","KFT", "MYL","NYX", "PBR.A", "PCLN", "POT", "RDS.A", "RIMM", "SLW","TCK",
        "TOT", "UTX", "VALE.P","WAG"],
    "Original Ticker": [
        "GOLD", "NTR", "TFC", "BKR","AVGO", "BRK-A", "BRK-B", "PARA","CMCSA", "CTRA", "TPR", "MDT", "WOLF", "EA", "VAL","MNST", "KHC","VTRS", "ICE", "PBR", "BKNG","NTR", "SHEL", "BB",
        "WPM", "TECK", "TTE", "RTX", "VALE", "WBA"]}

DATA_L1_TICKERS_REFINED = pd.DataFrame(DATA_L1_TICKERS_REFINED)

etfs_to_exclude = [
    'DIA', 'EEM', 'FAS', 'GDX', 'IWM', 'OIH', 'QQQ', 'QQQQ', 'QID', 'QLD', 
    'SDS', 'SKF', 'SMH', 'SNDK', 'SPY', 'SQQQ', 'SRS', 'SSO', 'TNA', 'TQQQ', 
    'TZA', 'VOO', 'VWO', 'VXX', 'XIV', 'XLE', 'XLF', 'XLI', 'XLK', 'XLU', 
    'XLV', 'XOM', 'BRCM', 'PCLN', 'MER', 'POT'
]