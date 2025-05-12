import pandas as pd
import multiprocessing
import pickle
import yfinance as yf
import sys
import os
import time

project_root = os.path.dirname(os.path.abspath(__file__))  # Current file directory
project_root = os.path.join(project_root, '..')  # Move one level up to the root
sys.path.append(project_root)


from definitions.constants_V import DATA_L1_TICKERS_REFINED, L1_TICKERS, API_KEY_EODHD, SHARE_OUTSTANDING_DATA_250_TESTING_PKL_L, SHARES_OUTSTANDING_DATA_250_UPDATED_PKL_L, EPS_DATA_TESTING_PKL_L, EPS_DATA_UPDATED_PKL_L, DIVIDEND_DATA_TESTING_PKL_L, DIVIDEND_DATA_UPDATED_PKL_L, EQUITY_DATA_TESTING_PKL_L, EQUITY_DATA_UPDATED_PKL_L
from utils.custom import update_data, update_dividend_data, earning_data_downloader, divident_data_downloader, shares_outstanding_data_downloader, equity_data_downloader

""" Download EPS data from eodhd.com , Now here are problems the tickers names here 
are not completely matching with the tickers names in the polygon api, we will deal that 
for now we have include all the tickers name we have including meta in it."""



"""Below are the all functions for downloading the financial data from eodhd.com, including
Earnings data, Divident, shares outstanding."""

# earning_data_downloader(L1_TICKERS,API_KEY_EODHD)
# divident_data_downloader(L1_TICKERS)
# shares_outstanding_data_downloader(L1_TICKERS,API_KEY_EODHD)
# equity_data_downloader(L1_TICKERS,API_KEY_EODHD)

df = DATA_L1_TICKERS_REFINED
# print('y')
update_data(SHARE_OUTSTANDING_DATA_250_TESTING_PKL_L, df, SHARES_OUTSTANDING_DATA_250_UPDATED_PKL_L)
update_data(EPS_DATA_TESTING_PKL_L, df, EPS_DATA_UPDATED_PKL_L)
# update_dividend_data(DIVIDEND_DATA_TESTING_PKL_L, df, DIVIDEND_DATA_UPDATED_PKL_L)
update_data(EQUITY_DATA_TESTING_PKL_L, df, EQUITY_DATA_UPDATED_PKL_L)