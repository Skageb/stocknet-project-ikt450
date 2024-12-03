import pandas as pd
import os
from skage.config import cfg
from sklearn.preprocessing import LabelEncoder
import numpy as np

def get_stock_data(path, start_date, end_date) -> pd.DataFrame:
    
    df = pd.read_csv(path)

    stock_name = path.split('.')[0]

    if '/' in stock_name:
        stock_name = stock_name.split('/')[-1]
    
    df['Name'] = stock_name

    df['Label'] = df.apply(lambda row: generate_label(row['Open'], row['Close']), axis=1)

    df = df[~(df['Label'] == -1)]

    df['Date'] = pd.to_datetime(df['Date'])

    # Define start and end dates
    start_date = '2014-01-01'
    end_date = '2015-08-02'

    # Filter the DataFrame for dates between start_date and end_date
    filtered_df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]

    return filtered_df


def generate_label(open_price, close_price) -> int:
    global missing_entries
    movement_percent = (close_price - open_price) / open_price * 100
    if movement_percent <= -0.5:
        return 0
    elif movement_percent >= 0.55:
        return 1
    else:
        return -1
    

def get_valid_dataframe(split) -> pd.DataFrame:
    
    if split == 'train':
        start_date = cfg.train_start_date
        end_date = cfg.train_end_date

    elif split == 'eval':
        start_date = cfg.eval_start_date
        end_date = cfg.eval_end_date

    root = 'dataset/price/raw'
    for idx, stock_path in enumerate(os.listdir(root)):
        df_stock = get_stock_data(os.path.join(root, stock_path), start_date, end_date)
        if idx == 0:
            df_valid = df_stock
        else:
            df_valid = pd.concat([df_valid, df_stock])
        
    return df_valid
    


def create_rnn_input_old(df):
    # Ensure 'Date' column is in datetime format
    df['Date'] = pd.to_datetime(df['Date'])

    # Sort the DataFrame by 'Name' and 'Date'
    df = df.sort_values(['Name', 'Date']).reset_index(drop=True)

    # List to store the new rows
    data_rows = []

    name_encoder = LabelEncoder()
    df['Name'] = name_encoder.fit_transform(df['Name'])

    # Group the DataFrame by 'Name' (stock)
    grouped = df.groupby('Name')

    # For each stock group
    for name, group in grouped:
        group = group.sort_values('Date').reset_index(drop=True)

        # Skip if the group has less than 6 rows (need at least 5 days of history)
        if len(group) < 6:
            continue

        # Iterate over the group starting from the 5th index
        for idx in range(5, len(group)):
            # Get the previous 5 trading days
            prev_data = group.iloc[idx-5:idx]

            # Prepare the row data
            row_dict = {'Date': group.loc[idx, 'Date'], 'Name': name}

            # Add Open and Close prices from previous 5 trading days
            for i in range(5):
                row_dict[f'Open_t-{5-i}'] = prev_data.iloc[i]['Open']
                row_dict[f'Close_t-{5-i}'] = prev_data.iloc[i]['Close']

            # Add the Label for the target date
            row_dict['Label'] = group.loc[idx, 'Label']

            # Append the row to data_rows
            data_rows.append(row_dict)

    # Create a new DataFrame from data_rows
    new_df = pd.DataFrame(data_rows)

    x = new_df.drop(['Label', 'Date'], axis=1)
    y = new_df['Label']
    x = x.to_numpy()
    y = y.to_numpy()

    return x, y


def normalize_per_stock(df):
    """
    Normalize 'Open' and 'Close' prices stock-wise (per stock group).
    """
    df_normalized = df.copy()

    # Group by 'Name' (stock) and normalize within each group
    for feature in ['Open', 'Close']:
        df_normalized[feature] = df.groupby('Name')[feature].transform(
            lambda x: (x - x.mean()) / x.std()
        )

    return df_normalized

def create_rnn_input(df):
    # Ensure 'Date' column is in datetime format
    df['Date'] = pd.to_datetime(df['Date'])

    # Sort the DataFrame by 'Name' and 'Date'
    df = df.sort_values(['Name', 'Date']).reset_index(drop=True)

    # Initialize lists for inputs (x) and labels (y)
    x_data = []
    y_data = []

    # Encode the 'Name' column (stock names)
    name_encoder = LabelEncoder()
    df['Name'] = name_encoder.fit_transform(df['Name'])

    # Group the DataFrame by 'Name' (stock)
    grouped = df.groupby('Name')

    # For each stock group
    for name, group in grouped:
        group = group.sort_values('Date').reset_index(drop=True)

        # Skip if the group has less than 6 rows (need at least 5 days of history)
        if len(group) < 6:
            continue

        # Iterate over the group starting from the 5th index
        for idx in range(5, len(group)):
            # Get the previous 5 trading days
            prev_data = group.iloc[idx-5:idx]

            # Extract Open and Close prices from the previous 5 days as features
            features = prev_data[['Open', 'Close']].to_numpy()  # Shape: (5, 2)

            # Append to x_data
            x_data.append(features)

            # Append the target label to y_data
            y_data.append(group.loc[idx, 'Label'])

    # Convert lists to numpy arrays
    x_data = np.array(x_data)  # Shape: (num_samples, seq_length, input_size)
    y_data = np.array(y_data)  # Shape: (num_samples,)

    return x_data, y_data