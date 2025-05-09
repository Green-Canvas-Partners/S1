import pandas as pd
import pickle
import sys
import os
from joblib import Parallel, delayed
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

# Append the project root to sys.path
project_root = os.path.dirname(os.path.abspath(__file__))  # Current file directory
project_root = os.path.join(project_root, '..')  # Move one level up to the root
sys.path.append(project_root)

# Import constants and custom utility functions
from definitions.constants import *
from utils.custom import (
    add_shift_columns_to_all, process_single_dataframe, 
    stock_selector, makeFinalDf, makeCorrectedDf
)

for PICA in [True, False]:

    dfsfilename=DIFF_REBALANCING_COMBINED_DATA_ALL_PKL
    dfsfilename=dfsfilename
    data=pd.read_pickle(dfsfilename)

    tickers = list(data[0]['Stock'].unique())
    start_date = '2012-02-14'
    end_date = '2024-12-25'
    data = yf.download(tickers, start=start_date, end=end_date)['Open']

    # Calculate returns
    returns = data.pct_change()
    returns=returns.shift(-1)

    returns.fillna(0,inplace=True)
    returns.to_csv(DIFF_REBALANCING_PICASSO_DIR+'returns_3.csv')

    dfsfilename=DIFF_REBALANCING_COMBINED_DATA_ALL_PKL
    dfsfilename=dfsfilename
    data=pd.read_pickle(dfsfilename)

    for day in range(18):
        # print(day)

        df = data[day]

        df = df[['Date',f'Momentum_{SELECTED_MOM_WINDOW}_{SELECTED_HALF_LIFE_WINDOW}','Stock','Returns']].sort_values('Date').reset_index(drop=True)

        dt = np.array(df[(df.Stock=='SPY')].Date)
        dict = {}
        for i in dt:
            tmp = df[df.Date == i].copy()
            sorted_tmp = tmp.sort_values(f'Momentum_{SELECTED_MOM_WINDOW}_{SELECTED_HALF_LIFE_WINDOW}', ascending=False).drop_duplicates()

            # Only include stocks with positive momentum in the top 10
            positive_momentum_stocks = sorted_tmp[sorted_tmp[f'Momentum_{SELECTED_MOM_WINDOW}_{SELECTED_HALF_LIFE_WINDOW}'] > 0]
            if len(positive_momentum_stocks) >= SELECTED_N_STOCK_POSITIVE:
                dict[i] = positive_momentum_stocks.head(SELECTED_N_STOCK_CHOSE).Stock.values
            else:
                dict[i] = np.array([])
        with open(f'{DIFF_REBALANCING_PICASSO_DIR}/momyear_3_{day}.pickle', 'wb') as file:
            # Use pickle.dump() to save your dictionary into the file
            pickle.dump(dict, file)

    for days in range(18):
    #     days = 0
        pickle_file_path = f'{DIFF_REBALANCING_PICASSO_DIR}/momyear_3_{days}.pickle'

        # Load the pickle file
        with open(pickle_file_path, 'rb') as file:
            pickle_data = pickle.load(file)

        # Function to replace 'FB' with 'META' in the arrays
        def replace_ticker(data):
            for date, tickers in data.items():
                # Use numpy.where to replace 'FB' with 'META' efficiently
                data[date] = np.where(tickers == 'FB', 'META', tickers)

        # Apply the function to modify the pickle data
        replace_ticker(pickle_data)

        # Save the modified data back to the same pickle file
        with open(pickle_file_path, 'wb') as file:
            pickle.dump(pickle_data, file)


    dynamic = pd.read_csv(DIFF_REBALANCING_PICASSO_DIR+'/dynamic_weighted_fifty_returns.csv')
    dfsfilename=DIFF_REBALANCING_COMBINED_DATA_ALL_PKL
    dfsfilename=dfsfilename
    file_path_all = dfsfilename



    # Open the file in binary mode and read the data
    with open(file_path_all, 'rb') as file:
        data = pickle.load(file)


    for days in range(18):
        # print(days)
        momentum = data[days]
        momentum = momentum[['Date',f'Momentum_{SELECTED_MOM_WINDOW}_{SELECTED_HALF_LIFE_WINDOW}','Stock','Returns']].sort_values('Date').reset_index(drop=True)
        returns = pd.read_csv(DIFF_REBALANCING_PICASSO_DIR+'returns_3.csv')

        file_path = '/dynamic_weighted_fifty_returns.csv'

        file_data = pd.read_csv(DIFF_REBALANCING_PICASSO_DIR+file_path)

        second_file_path = 'returns_3.csv'
        second_file_data = pd.read_csv(DIFF_REBALANCING_PICASSO_DIR+second_file_path)
        file_data['Date'] = pd.to_datetime(file_data['Date'])
        second_file_data['Date'] = pd.to_datetime(second_file_data['Date'])
        # Identifying the ticker columns (excluding 'Date') in both files
        ticker_columns_first_file = set(file_data.columns) - {'Date'}
        ticker_columns_second_file = set(second_file_data.columns) - {'Date'}

        # Finding common tickers to be excluded from the second file during merge
        common_tickers = ticker_columns_second_file.intersection(ticker_columns_first_file)

        # Dropping the common tickers from the second file
        second_file_data_dropped_common = file_data.drop(columns=common_tickers, errors='ignore')
        
        second_file_data_dropped_common.set_index('Date', inplace=True)
        second_file_data.set_index('Date', inplace=True)

        # Merging the first file with the modified second file
        combined_data_no_repetition = pd.merge(second_file_data_dropped_common,second_file_data, right_index=True, left_index=True)
        combined_data_no_repetition.reset_index(inplace=True)

        # Displaying the first few rows of the updated dataset
        combined_data_no_repetition.head()

        if not PICA:
            combined_data_no_repetition['regime']=1
        combined_data_no_repetition1 = combined_data_no_repetition[combined_data_no_repetition['regime']==1]
        combined_data_no_repetition1 = combined_data_no_repetition1.head(-1)
        combined_data_no_repetition0 = combined_data_no_repetition[combined_data_no_repetition['regime']!=1]
        combined_data_no_repetition0 = combined_data_no_repetition0.head(-1)
        combined_data_no_repetition1.to_csv(DIFF_REBALANCING_PICASSO_DIR+'combined_data_no_repetition1.csv')
        combined_data_no_repetition0.to_csv(DIFF_REBALANCING_PICASSO_DIR+'combined_data_no_repetition0.csv')

        pickle_file_path = f'/momyear_3_{days}.pickle'
        with open(DIFF_REBALANCING_PICASSO_DIR+pickle_file_path, 'rb') as file:
            pickle_data = pickle.load(file)
            
        # print(pickle_data)
        csv_file_path = 'combined_data_no_repetition1.csv'
        dataframe = pd.read_csv(DIFF_REBALANCING_PICASSO_DIR+csv_file_path)
        dataframe['Date'] = pd.to_datetime(dataframe['Date'])
        
        def add_correct_daily_averages_and_tickers_to_dataframe(dataframe, pickle_data):
            dataframe['Daily_Average'] = np.nan
            dataframe['Used_Tickers'] = None

            # Convert the date keys to datetime and sort them to ensure correct ordering
            sorted_dates = sorted([datetime.strptime(date_str, '%Y-%m-%d') for date_str in pickle_data.keys()])

            for i in range(len(sorted_dates) - 1):
                current_date = sorted_dates[i]
                next_date = sorted_dates[i + 1]

                # Define the date range as from the day after current_date to the day of next_date
                date_mask = (dataframe['Date'] > current_date) & (dataframe['Date'] <= next_date)

                tickers = pickle_data[current_date.strftime('%Y-%m-%d')]
                valid_tickers = [ticker for ticker in tickers if ticker in dataframe.columns]

                if valid_tickers:
                    daily_sums = dataframe.loc[date_mask, valid_tickers].sum(axis=1) / len(valid_tickers)
                    dataframe.loc[date_mask, 'Daily_Average'] = daily_sums
                    dataframe.loc[date_mask, 'Used_Tickers'] = ', '.join(valid_tickers)

            return dataframe

        
        
        updated_dataframe = add_correct_daily_averages_and_tickers_to_dataframe(dataframe, pickle_data)

        # Display the updated DataFrame
        # print(updated_dataframe)
        updated_dataframe['Used_Tickers'] = updated_dataframe['Used_Tickers'].fillna(0)
        updated_dataframe['Daily_Average'] = updated_dataframe['Daily_Average'].fillna(0)
        combined_data_no_repetition0['Daily_Average'] = 0
        combined_data_no_repetition0['Used_Tickers'] = 0
        combined_data_no_repetition0.to_csv(DIFF_REBALANCING_PICASSO_DIR+'regime_new0.csv')
        updated_dataframe.to_csv(DIFF_REBALANCING_PICASSO_DIR+'regime_new1.csv',index=False)
        df = pd.read_csv(DIFF_REBALANCING_PICASSO_DIR+'regime_new0.csv')
        df1 = pd.read_csv(DIFF_REBALANCING_PICASSO_DIR+'regime_new1.csv')
        final = pd.concat([df,df1]).sort_values(by='Date')
        final.to_csv(f'{DIFF_REBALANCING_PICASSO_DIR}PicaMom_new_3_{days}.csv',index=False)
        df = pd.read_csv(f'{DIFF_REBALANCING_PICASSO_DIR}PicaMom_new_3_{days}.csv')
        df['real_return'] = np.where(df['regime'] == 1, df['Daily_Average'], df['fifty'])
        df[['Date','regime', 'returns', 'weighted_returns', 'fifty','Daily_Average',
        'Used_Tickers','real_return']].head(40)
        df.to_csv(f'{DIFF_REBALANCING_PICASSO_DIR}Picassowithmom_new_{days}.csv')


    def exponential_weights(length, alpha=0.5):
        """
        Generate exponential weights for a given length.
        
        Args:
        length (int): Number of weights to generate.
        alpha (float): Decay factor for the weights, typically between 0 and 1.
        
        Returns:
        np.array: Array of weights, where weights exponentially decrease.
        """
        # Generate an array of indices from 0 to length-1
        indices = np.arange(length)
        # Calculate weights using the decay factor raised to the power of each index
        weights = alpha ** indices
        # Normalize weights so they sum to 1
        weights /= weights.sum()
        # Reverse the array so the highest weight is first
        return weights

    for days in range(18):
        # Load the dataframe
        file_path = f'Picassowithmom_new_{days}.csv'
        dataframe = pd.read_csv(DIFF_REBALANCING_PICASSO_DIR+file_path)

        # Function to calculate the weighted return for a given row and list of tickers
        def weighted_return(row, tickers):
            num_tickers = len(tickers)
            if num_tickers == 0:
                return 0
            # Assigning weights in reverse order (heavier weights for first tickers)
            weights = exponential_weights(SELECTED_N_STOCK_CHOSE, EXP_WEIGHT)
            # print(weights)
            # Calculating the weighted return
            return sum(row[ticker] * weight for ticker, weight in zip(tickers, weights))

        # Creating 'Ticker_List' column for the entire dataframe
        dataframe['Used_Tickers'] = dataframe['Used_Tickers'].astype(str)
        dataframe['Ticker_List'] = dataframe['Used_Tickers'].apply(lambda x: x.split(', ') if x != '0' else [])

        # Applying weighted return calculation only to rows where regime is 1
        dataframe['Weighted_Returns'] = dataframe.apply(lambda row: weighted_return(row, row['Ticker_List']) if row['regime'] == 1 else row['fifty'], axis=1)
        dataframe.to_csv(f'{DIFF_REBALANCING_PICASSO_DIR}Picassowithmom_new_weighted_{days}.csv')


    # List to store the results
    results = []
    rets=[]

    for days in range(18):
        # Load the dataframe
        df = pd.read_csv(f'{DIFF_REBALANCING_PICASSO_DIR}Picassowithmom_new_weighted_{days}.csv')

        # Assuming there is a 'Date' column to identify the date of the results
        date = df['Date'].iloc[-1]  # Get the last date in the DataFrame
        rets.append(df[['Weighted_Returns','Date']])

        # Calculate Annual Return
        annual_return = np.prod(1 + df['Weighted_Returns'].dropna())**(252/df['Weighted_Returns'].dropna().shape[0]) - 1

        # Calculate cumulative returns for maximum drawdown
        cumulative_returns = (1 + df['Weighted_Returns'].dropna()).cumprod()
        running_max = cumulative_returns.cummax()
        drawdowns = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdowns.min()
        ninetyfive_qt_draw=drawdowns.quantile(0.05)
        ninety_qt_draw=drawdowns.quantile(0.1)

        # Sortino Ratio
        MAR = 0
        downside_returns = df['Weighted_Returns'][df['Weighted_Returns'] < MAR]
        downside_deviation = downside_returns.std() * np.sqrt(252)
        average_annual_return = df['Weighted_Returns'].mean() * 252
        sortino_ratio = (average_annual_return - MAR) / downside_deviation

        # Sharpe Ratio
        risk_free_rate = 0
        standard_deviation = df['Weighted_Returns'].std() * np.sqrt(252)
        sharpe_ratio = (average_annual_return - risk_free_rate) / standard_deviation

        # Calmar Ratio
        calmar_ratio = annual_return / abs(max_drawdown)

        # Append the results for this iteration to the list
        results.append({
            'Date': date,
            'Annual Return': annual_return,
            'ninety_qt_draw':ninety_qt_draw,
            'ninetyfive_qt_draw':ninetyfive_qt_draw,
            'Maximum Drawdown': max_drawdown,
            'Sortino Ratio': sortino_ratio,
            'Sharpe Ratio': sharpe_ratio,
            'Calmar Ratio': calmar_ratio
        })

    # Create a DataFrame from the results list
    results_df = pd.DataFrame(results)

    # Saving the results to CSV
    results_df.to_csv(DIFF_REBALANCING_PICASSO_DIR+'simulation_results.csv', index=False)

    for i in range(18):
        rets[i].set_index('Date', inplace=True)

    result = pd.concat(rets, axis=1)

    result['finalrets']=result.mean(axis=1)
    results=[]
    annual_return = np.prod(1 + result['finalrets'].dropna())**(252/result['finalrets'].dropna().shape[0]) - 1

    # Calculate cumulative returns for maximum drawdown
    cumulative_returns = (1 + result['finalrets'].dropna()).cumprod()
    running_max = cumulative_returns.cummax()
    drawdowns = (cumulative_returns - running_max) / running_max
    max_drawdown = drawdowns.min()
    ninetyfive_qt_draw=drawdowns.quantile(0.05)
    ninety_qt_draw=drawdowns.quantile(0.1)

    # Sortino Ratio
    MAR = 0
    downside_returns = result['finalrets'][result['finalrets'] < MAR]
    downside_deviation = downside_returns.std() * np.sqrt(252)
    average_annual_return = result['finalrets'].mean() * 252
    sortino_ratio = (average_annual_return - MAR) / downside_deviation

    # Sharpe Ratio
    risk_free_rate = 0
    standard_deviation = result['finalrets'].std() * np.sqrt(252)
    sharpe_ratio = (average_annual_return - risk_free_rate) / standard_deviation

    # Calmar Ratio
    calmar_ratio = annual_return / abs(max_drawdown)

    # Append the results for this iteration to the list
    results.append({
        'Date': date,
        'Annual Return': annual_return,
        'ninety_qt_draw':ninety_qt_draw,
        'ninetyfive_qt_draw':ninetyfive_qt_draw,
        'Maximum Drawdown': max_drawdown,
        'Sortino Ratio': sortino_ratio,
        'Sharpe Ratio': sharpe_ratio,
        'Calmar Ratio': calmar_ratio
    })

    # Create a DataFrame from the results list
    results_result = pd.DataFrame(results)

    # print(results_result)

    tickers = ['SPY']
    data = yf.download(tickers, start=start_date, end=end_date)['Open']

    # Calculate returns
    returns = data.pct_change()
    returns=returns.shift(-1)
    returns.fillna(0,inplace=True)
    returns=pd.DataFrame(returns)
    returns.index=pd.to_datetime(returns.index)
    returns.rename(columns={'Open':'SPY'}, inplace=True)
    result.index=pd.to_datetime(result.index)
    finaldf=pd.merge(result,returns,left_index=True, right_index=True)

    strategy_params = f"{DV_QUANTILE_THRESHOLD}_{SELECTED_TOP_VOL_STOCKS}_{SELECTED_MOM_WINDOW}_{SELECTED_HALF_LIFE_WINDOW}_{SELECTED_N_STOCK_POSITIVE}_{SELECTED_N_STOCK_CHOSE}_{EXP_WEIGHT}"

    # Save the final returns to a CSV file with the strategy parameters in the filename
    if PICA:
        finaldf[['finalrets', 'SPY']].to_csv(f"{DIFF_REBALANCING_PICASSO_DIR}/finaldf_{strategy_params}_PICA.csv")
    else:
        finaldf[['finalrets', 'SPY']].to_csv(f"{DIFF_REBALANCING_PICASSO_DIR}/finaldf_{strategy_params}_NONPICA.csv")

    pdf_filename = "log_cumprod_plot.pdf"
    with PdfPages(pdf_filename) as pdf:
        # Create the plot
        plt.figure(figsize=(10, 6))
        plt.plot(np.log((finaldf['finalrets'] + 1).cumprod()), label='finalrets')
        plt.plot(np.log((finaldf['SPY'] + 1).cumprod()), label='SPY')
        
        # Add title and labels
        plt.title('Log of Cumulative Product of Returns')
        plt.xlabel('Index')
        plt.ylabel('Log(Cumulative Returns)')
        
        # Add legend
        plt.legend()
        
        # Save the figure to the PDF
        pdf.savefig()
        plt.close()

    pdf_filename = "cumprod_plot.pdf"
    with PdfPages(pdf_filename) as pdf:
        # Create the plot
        plt.figure(figsize=(10, 6))
        plt.plot((finaldf['finalrets'] + 1).cumprod(), label='finalrets')
        plt.plot((finaldf['SPY'] + 1).cumprod(), label='SPY')
        
        # Add title and labels
        plt.title('Log of Cumulative Product of Returns')
        plt.xlabel('Index')
        plt.ylabel('Log(Cumulative Returns)')
        
        # Add legend
        plt.legend()
        
        # Save the figure to the PDF
        pdf.savefig()
        plt.close()

    print(f"Plot saved to {pdf_filename}")

    

    