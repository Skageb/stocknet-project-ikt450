import os
import pandas as pd

def get_all_stock_trading_dates(stock_name, price_dir) -> list:
    price_df = pd.read_csv(os.path.join(price_dir, f'{stock_name}.csv'))    #Read stock price file
    price_df['Date'] = pd.to_datetime(price_df['Date']) 
    all_stock_trading_dates = sorted(price_df['Date'].unique())     #Get all unique trading dates and return them sorted.
    all_stock_trading_dates_str = [date.strftime('%Y-%m-%d') for date in all_stock_trading_dates]
    return all_stock_trading_dates_str


price_d = '/home/skage/projects/ikt450_deep-neural-networks/stocknet-project-ikt450/dataset/price/preprocessed/csv'
Stoc = 'SRE'

print(get_all_stock_trading_dates(Stoc, price_d))