import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import pandas as pd
import json
import os
from transformers import BertTokenizer, AutoTokenizer, AutoModelForSequenceClassification
import numpy as np


def get_n_prior_dates(date, n):
    date_format = "%Y-%m-%d"

    # Convert the string to a datetime object
    given_date_obj = datetime.strptime(date, date_format)

    # Calculate the 3 dates leading up to the given date
    days_leading_up = [given_date_obj - timedelta(days=i) for i in range(n, 0, -1)]

    # Convert the dates back to strings in the desired format
    dates_as_strings = [date.strftime(date_format) for date in days_leading_up]

    return dates_as_strings

def get_date_range(start_date, end_date):
    """
    Generate a list of dates from start_date to end_date (inclusive).

    Args:
        start_date (str): Start date in "YYYY-MM-DD" format.
        end_date (str): End date in "YYYY-MM-DD" format.

    Returns:
        list: List of dates as strings in "YYYY-MM-DD" format.
    """
    # Convert the start and end dates to datetime objects
    start_date_obj = datetime.strptime(start_date, "%Y-%m-%d")
    end_date_obj = datetime.strptime(end_date, "%Y-%m-%d")

    # Generate all dates in the range
    date_list = []
    current_date = start_date_obj
    while current_date <= end_date_obj:
        date_list.append(current_date.strftime("%Y-%m-%d"))
        current_date += timedelta(days=1)
    
    return date_list


def get_date_from_difference(date, offset_days:int):
    date_format = "%Y-%m-%d"
    given_date= datetime.strptime(date, date_format)
    
    offset_date = given_date + timedelta(days=offset_days)
    return offset_date.strftime(date_format)

class TweetXPriceY(Dataset):
    def __init__(self, start_date, end_date ,day_lag:int, tweets_per_day:int, words_per_tweet:int, twitter_root:str, price_root:str) -> None:
        self.tweet_dir = twitter_root
        self.price_dir = price_root
        self.day_lag = day_lag
        self.start_date = get_date_from_difference(start_date, 5)
        self.end_date = end_date

        self.tweets_per_day = tweets_per_day
        self.words_per_tweet = words_per_tweet

        self.data_samples = self.__prepare_dataset__()

        #Tokenizer to encode words
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    def __prepare_dataset__(self):
        """Prepares a list of (stock_name, target_date) pairs for indexing."""
        data_samples = []
        for stock_name in os.listdir(self.tweet_dir):
            price_df = pd.read_csv(os.path.join(self.price_dir, f'{stock_name}.csv'))
            price_df['Date'] = pd.to_datetime(price_df['Date'])

            for date in get_date_range(self.start_date, self.end_date):
                date_obj = pd.to_datetime(date)
                if date_obj in price_df['Date'].values:
                    movement_percent = price_df.loc[price_df['Date'] == date_obj, 'Movement_Percent'].iloc[0]
                    if movement_percent > 0.0055 or movement_percent <= -0.005:
                        data_samples.append((stock_name, date))
        return data_samples

    def encode_tweet(self, words):
        return self.tokenizer.convert_tokens_to_ids(words)

    def __getitem__(self, idx):
        stock_name, target_date = self.data_samples[idx]

        price_df = pd.read_csv(os.path.join(self.price_dir, f'{stock_name}.csv'))
        price_df['Date'] = pd.to_datetime(price_df['Date'])
        target_date_obj = pd.to_datetime(target_date)

        price_movement = price_df.loc[price_df['Date'] == target_date_obj, 'Movement_Percent'].values[0]
        y = 1 if price_movement > 0.0055 else 0
            
        #elif price_movement <= -0.005:
        #    y = 0

        x = []
        for prev_date in get_n_prior_dates(target_date, self.day_lag):
            tweet_file = os.path.join(self.tweet_dir, stock_name, f"{prev_date}.json")
            daily_tweets = []

            if os.path.exists(tweet_file):
                # Load tweets
                with open(tweet_file, 'r') as f:
                    tweets = json.load(f)

                # Process tweets_per_day tweets
                for tweet in tweets[:self.tweets_per_day]:
                    words = tweet['text'][:self.words_per_tweet]  # Truncate to words_per_tweet
                    words += ["[PAD]"] * (self.words_per_tweet - len(words))  # Pad words
                    encoded_tweet = self.encode_tweet(words)
                    daily_tweets.append(encoded_tweet)

            # Pad missing tweets for the day
            while len(daily_tweets) < self.tweets_per_day:
                daily_tweets.append(self.encode_tweet(["[PAD]"] * self.words_per_tweet))

            x.append(daily_tweets)


        # Convert to tensor
        x_tensor = torch.tensor(x, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.long)
        return x_tensor, y_tensor


    def __len__(self):
        return len(self.data_samples)
    


class SentimentPriceXPriceY(Dataset):
    '''Dataset loader. The data used is twitter sentiment and price movements over the last five days'''
    def __init__(self, start_date, end_date, day_lag:int, tweets_per_day:int, words_per_tweet:int, twitter_root:str, price_root:str, **kwargs) -> None:
        self.tweet_dir = twitter_root
        self.price_dir = price_root
        self.day_lag = day_lag
        self.start_date = get_date_from_difference(start_date, 5)
        self.end_date = end_date

        self.words_per_tweet = words_per_tweet

        self.data_samples = self.__prepare_dataset__()


    def __prepare_dataset__(self):
        """Prepares a list of (stock_name, target_date) pairs for indexing."""
        data_samples = []
        for stock_name in os.listdir(self.tweet_dir):
            price_df = pd.read_csv(os.path.join(self.price_dir, f'{stock_name}.csv'))
            price_df['Date'] = pd.to_datetime(price_df['Date'])
            included_trades = 0
            for date in get_date_range(self.start_date, self.end_date):
                date_obj = pd.to_datetime(date)
                if date_obj in price_df['Date'].values:
                    movement_percent = price_df.loc[price_df['Date'] == date_obj, 'Movement_Percent'].iloc[0]
                    if movement_percent > 0.0055 or movement_percent <= -0.005:
                        if included_trades == self.day_lag:
                            data_samples.append((stock_name, date))
                        else:
                            included_trades += 1
        return data_samples #Drop the first d days to ensure all dates in the dataset has d days worth of lag data.

    
    def __getitem__(self, idx):
        stock_name, target_date = self.data_samples[idx]
        target_date_obj = pd.to_datetime(target_date)
        
        # Load price data and extract trading dates
        price_df = pd.read_csv(os.path.join(self.price_dir, f'{stock_name}.csv'))
        price_df['Date'] = pd.to_datetime(price_df['Date'])
        trading_dates = sorted(price_df['Date'].unique())

        # Get previous T trading days
        T = self.day_lag
        target_idx = trading_dates.index(target_date_obj)
        prior_trading_dates = trading_dates[target_idx - T: target_idx]

        sentiment_sequence = []
        price_movement_sequence = []

        for idx, date in enumerate(prior_trading_dates):
            date_obj = pd.to_datetime(date)

            price_movement = price_df.loc[price_df['Date'] == date, 'Movement_Percent'].values[0]
            
            price_movement_sequence.append(price_movement)

            # Get tweets from the previous calendar day
            tweet_end_date = (date_obj - timedelta(days=1)).strftime("%Y-%m-%d")
            tweet_start_date = trading_dates[target_idx - T + idx - 1].strftime("%Y-%m-%d")     #target_idx - T is first trading day in prior trading days
            tweet_dates = get_date_range(tweet_start_date, tweet_end_date)

            sentiment_relevant_for_trading_day = []
            average_sentiment = []      #Average twitter sentiment from tweets in the date range [prior_trading_date, current_trading_date)

            for tweet_date in tweet_dates:      #Get all tweets in the date range [prior_trading_date, current_trading_date)
                tweet_file = os.path.join(self.tweet_dir, stock_name, f"{tweet_date}.json")
                #print(tweet_file)
                if os.path.exists(tweet_file):
                    # Load tweets
                    with open(tweet_file, 'r') as f:
                        tweets = json.load(f)
                        #input(tweets)

                    # Process tweets from the day
                    for tweet in tweets:
                        sentiment = tweet['sentiment']
                        sentiment_relevant_for_trading_day.append(sentiment)

            #Calculate average twitter sentiment of stock tweets
            if sentiment_relevant_for_trading_day:
                average_sentiment = np.mean(sentiment_relevant_for_trading_day)

            #If no tweets use the prior trading days sentiment or neutral(3) if this is the first day of the lag
            else:
                if sentiment_sequence:
                    average_sentiment = sentiment_sequence[-1]
                else:
                    average_sentiment = 3    
            sentiment_sequence.append(average_sentiment)
        
        #Apply neutral sentiment padding in cases of missing data.
        if len(sentiment_sequence) < self.day_lag:
            padding_length = self.day_lag - len(sentiment_sequence)
            sentiment_sequence = [3.0] * padding_length + sentiment_sequence

        # Convert sequences to tensors
        sentiment_sequence = torch.tensor(sentiment_sequence, dtype=torch.float32)
        price_movement_sequence = torch.tensor(price_movement_sequence, dtype=torch.float32)

        if len(sentiment_sequence) < 5 or len(price_movement_sequence) < 5:
            with open('debug.json', 'w') as f:
                debug_object = {
                    'sentiment_sequence': sentiment_sequence.numpy().tolist(),
                    'price_movement_sequence': price_movement_sequence.numpy().tolist(),
                    'Prior dates': prior_trading_dates,
                    'Stock': stock_name,
                    'target_date': target_date
                }
                json.dump(debug_object, f)

        x = (sentiment_sequence, price_movement_sequence)

        # Get target y for the target trading day
        y_movement = price_df.loc[price_df['Date'] == target_date_obj, 'Movement_Percent'].values[0]
        y = 1 if y_movement > 0.0055 else 0  # Define your threshold

        y_tensor = torch.tensor(y, dtype=torch.long)

        return x, y_tensor


    def __len__(self):
        return len(self.data_samples)
                        

class NormSentimentAllPriceXPriceY(Dataset):
    '''Dataset loader. The data used is twitter sentiment normalized and price movements with all available stock info over the last five days'''
    def __init__(self, start_date, end_date, day_lag:int, tweets_per_day:int, words_per_tweet:int, twitter_root:str, price_root:str, **kwargs) -> None:
        self.tweet_dir = twitter_root
        self.price_dir = price_root
        self.day_lag = day_lag
        self.start_date = get_date_from_difference(start_date, 5)
        self.end_date = end_date


        self.words_per_tweet = words_per_tweet

        # Initialize lists to collect all values
        self.all_movement_percents = []
        self.all_volumes = []
        self.all_sentiments = []

        self.data_samples = self.__prepare_dataset__()

        # Get sentiment values for normalization
        for stock_name in os.listdir(self.tweet_dir):
            stock_tweet_dir = os.path.join(self.tweet_dir, stock_name)
            for tweet_file in os.listdir(stock_tweet_dir):
                if tweet_file.endswith('.json'):
                    with open(os.path.join(stock_tweet_dir, tweet_file), 'r') as f:
                        tweets = json.load(f)
                        sentiments = [tweet['sentiment'] for tweet in tweets]
                        self.all_sentiments.extend(sentiments)



        # Compute mean and std after collecting all values
        self.movement_mean = np.mean(self.all_movement_percents)
        self.movement_std = np.std(self.all_movement_percents)
        self.volume_mean = np.mean(self.all_volumes)
        self.volume_std = np.std(self.all_volumes)
        self.sentiment_mean = np.mean(self.all_sentiments)
        self.sentiment_std = np.std(self.all_sentiments)

        self.dataset_cols = ['Sentiment', 'Movement_Percent' ,'Open', 'High' ,'Low', "Close" , 'Volume' ]


    def __prepare_dataset__(self):
        """Prepares a list of (stock_name, target_date) pairs for indexing."""
        data_samples = []
        for stock_name in os.listdir(self.tweet_dir):
            price_df = pd.read_csv(os.path.join(self.price_dir, f'{stock_name}.csv'))
            price_df['Date'] = pd.to_datetime(price_df['Date'])

            # Collect movement percents and volumes for normalization
            self.all_movement_percents.extend(price_df['Movement_Percent'].values)
            self.all_volumes.extend(price_df['Volume'].values)

            included_trades = 0
            for date in get_date_range(self.start_date, self.end_date):
                date_obj = pd.to_datetime(date)
                if date_obj in price_df['Date'].values:
                    movement_percent = price_df.loc[price_df['Date'] == date_obj, 'Movement_Percent'].iloc[0]
                    if movement_percent > 0.0055 or movement_percent <= -0.005:
                        if included_trades == self.day_lag:
                            data_samples.append((stock_name, date))
                        else:
                            included_trades += 1
        return data_samples #Drop the first d days to ensure all dates in the dataset has d days worth of lag data.

    
    def __getitem__(self, idx):
        stock_name, target_date = self.data_samples[idx]
        target_date_obj = pd.to_datetime(target_date)
        
        # Load price data and extract trading dates
        price_df = pd.read_csv(os.path.join(self.price_dir, f'{stock_name}.csv'))
        price_df['Date'] = pd.to_datetime(price_df['Date'])
        trading_dates = sorted(price_df['Date'].unique())

        # Get previous T trading days
        T = self.day_lag
        target_idx = trading_dates.index(target_date_obj)
        prior_trading_dates = trading_dates[target_idx - T: target_idx]

        sentiment_sequence = []
        price_sequence = []

        for idx, date in enumerate(prior_trading_dates):
            date_obj = pd.to_datetime(date)

            row = price_df.loc[price_df['Date'] == date]
            
            

            if not row.empty:
                # Extract the required features
                movement_percent = row['Movement_Percent'].values[0]
                open_price = row['Open'].values[0]
                high_price = row['High'].values[0]
                low_price = row['Low'].values[0]
                close_price = row['Close'].values[0]
                volume = row['Volume'].values[0]

                movement_percent = (movement_percent - self.movement_mean) / self.movement_std
                volume = (volume - self.volume_mean) / self.volume_std

                price_features = [movement_percent, open_price, high_price, low_price, close_price, volume]
            else:
                # Handle missing data by inserting zeros or previous valid data
                price_features = [0, 0, 0, 0, 0]
            price_sequence.append(price_features)
            # Get tweets from the previous calendar day
            tweet_end_date = (date_obj - timedelta(days=1)).strftime("%Y-%m-%d")
            tweet_start_date = trading_dates[target_idx - T + idx - 1].strftime("%Y-%m-%d")     #target_idx - T is first trading day in prior trading days
            tweet_dates = get_date_range(tweet_start_date, tweet_end_date)

            sentiment_relevant_for_trading_day = []
            average_sentiment = []      #Average twitter sentiment from tweets in the date range [prior_trading_date, current_trading_date)

            for tweet_date in tweet_dates:      #Get all tweets in the date range [prior_trading_date, current_trading_date)
                tweet_file = os.path.join(self.tweet_dir, stock_name, f"{tweet_date}.json")
                #print(tweet_file)
                if os.path.exists(tweet_file):
                    # Load tweets
                    with open(tweet_file, 'r') as f:
                        tweets = json.load(f)
                        #input(tweets)

                    # Process tweets from the day
                    for tweet in tweets:
                        sentiment = tweet['sentiment']
                        sentiment_relevant_for_trading_day.append(sentiment)

            #Calculate average twitter sentiment of stock tweets
            if sentiment_relevant_for_trading_day:
                average_sentiment = np.mean(sentiment_relevant_for_trading_day)

            #If no tweets use the prior trading days sentiment or neutral(3) if this is the first day of the lag
            else:
                if sentiment_sequence:
                    average_sentiment = sentiment_sequence[-1]
                else:
                    average_sentiment = 3    
            # Normalize sentiment
            average_sentiment = (average_sentiment - self.sentiment_mean) / self.sentiment_std
            sentiment_sequence.append(average_sentiment)
        
        #Apply neutral sentiment padding in cases of missing data.
        if len(sentiment_sequence) < self.day_lag:
            padding_length = self.day_lag - len(sentiment_sequence)
            sentiment_sequence = [3.0] * padding_length + sentiment_sequence

        # Convert sequences to tensors
        sentiment_sequence = torch.tensor(sentiment_sequence, dtype=torch.float32)
        price_sequence = torch.tensor(price_sequence, dtype=torch.float32)

        if len(sentiment_sequence) < 5 or len(price_sequence) < 5:
            with open('debug.json', 'w') as f:
                debug_object = {
                    'sentiment_sequence': sentiment_sequence.numpy().tolist(),
                    'price_movement_sequence': price_sequence.numpy().tolist(),
                    'Prior dates': prior_trading_dates,
                    'Stock': stock_name,
                    'target_date': target_date
                }
                json.dump(debug_object, f)

        x = (sentiment_sequence, price_sequence)

        # Get target y for the target trading day
        y_movement = price_df.loc[price_df['Date'] == target_date_obj, 'Movement_Percent'].values[0]
        y = 1 if y_movement > 0.0055 else 0  # Define your threshold

        y_tensor = torch.tensor(y, dtype=torch.long)
        #input(x[1].size(), x[0].size())
        return x, y_tensor

    def get_input_size(self):
        return (1, 5)

    def __len__(self):
        return len(self.data_samples)


class NormPriceXPriceY(Dataset):
    '''Dataset loader. The data used is twitter sentiment normalized and price movements with all available stock info over the last five days'''
    def __init__(self, start_date, end_date, day_lag:int, tweets_per_day:int, words_per_tweet:int, twitter_root:str, price_root:str, **kwargs) -> None:
        self.tweet_dir = twitter_root
        self.price_dir = price_root
        self.day_lag = day_lag
        self.start_date = get_date_from_difference(start_date, 5)
        self.end_date = end_date


        self.words_per_tweet = words_per_tweet

        # Initialize lists to collect all values
        self.all_movement_percents = []
        self.all_volumes = []
        self.all_sentiments = []

        self.data_samples = self.__prepare_dataset__()

        # Get sentiment values for normalization
        for stock_name in os.listdir(self.tweet_dir):
            stock_tweet_dir = os.path.join(self.tweet_dir, stock_name)
            for tweet_file in os.listdir(stock_tweet_dir):
                if tweet_file.endswith('.json'):
                    with open(os.path.join(stock_tweet_dir, tweet_file), 'r') as f:
                        tweets = json.load(f)
                        sentiments = [tweet['sentiment'] for tweet in tweets]
                        self.all_sentiments.extend(sentiments)



        # Compute mean and std after collecting all values
        self.movement_mean = np.mean(self.all_movement_percents)
        self.movement_std = np.std(self.all_movement_percents)
        self.volume_mean = np.mean(self.all_volumes)
        self.volume_std = np.std(self.all_volumes)
        self.sentiment_mean = np.mean(self.all_sentiments)
        self.sentiment_std = np.std(self.all_sentiments)

        self.dataset_cols = ['Sentiment', 'Movement_Percent' ,'Open', 'High' ,'Low', "Close" , 'Volume' ]


    def __prepare_dataset__(self):
        """Prepares a list of (stock_name, target_date) pairs for indexing."""
        data_samples = []
        for stock_name in os.listdir(self.tweet_dir):
            price_df = pd.read_csv(os.path.join(self.price_dir, f'{stock_name}.csv'))
            price_df['Date'] = pd.to_datetime(price_df['Date'])

            # Collect movement percents and volumes for normalization
            self.all_movement_percents.extend(price_df['Movement_Percent'].values)
            self.all_volumes.extend(price_df['Volume'].values)

            included_trades = 0
            for date in get_date_range(self.start_date, self.end_date):
                date_obj = pd.to_datetime(date)
                if date_obj in price_df['Date'].values:
                    movement_percent = price_df.loc[price_df['Date'] == date_obj, 'Movement_Percent'].iloc[0]
                    if movement_percent > 0.0055 or movement_percent <= -0.005:
                        if included_trades == self.day_lag:
                            data_samples.append((stock_name, date))
                        else:
                            included_trades += 1
        return data_samples #Drop the first d days to ensure all dates in the dataset has d days worth of lag data.

    
    def __getitem__(self, idx):
        stock_name, target_date = self.data_samples[idx]
        target_date_obj = pd.to_datetime(target_date)
        
        # Load price data and extract trading dates
        price_df = pd.read_csv(os.path.join(self.price_dir, f'{stock_name}.csv'))
        price_df['Date'] = pd.to_datetime(price_df['Date'])
        trading_dates = sorted(price_df['Date'].unique())

        # Get previous T trading days
        T = self.day_lag
        target_idx = trading_dates.index(target_date_obj)
        prior_trading_dates = trading_dates[target_idx - T: target_idx]

        sentiment_sequence = []
        price_sequence = []

        for idx, date in enumerate(prior_trading_dates):
            date_obj = pd.to_datetime(date)

            row = price_df.loc[price_df['Date'] == date]
            
            

            if not row.empty:
                # Extract the required features
                movement_percent = row['Movement_Percent'].values[0]
                open_price = row['Open'].values[0]
                high_price = row['High'].values[0]
                low_price = row['Low'].values[0]
                close_price = row['Close'].values[0]
                volume = row['Volume'].values[0]

                movement_percent = (movement_percent - self.movement_mean) / self.movement_std
                volume = (volume - self.volume_mean) / self.volume_std

                price_features = [movement_percent, open_price, high_price, low_price, close_price, volume]
            else:
                # Handle missing data by inserting zeros or previous valid data
                price_features = [0, 0, 0, 0, 0]
            price_sequence.append(price_features)
            # Get tweets from the previous calendar day
            tweet_end_date = (date_obj - timedelta(days=1)).strftime("%Y-%m-%d")
            tweet_start_date = trading_dates[target_idx - T + idx - 1].strftime("%Y-%m-%d")     #target_idx - T is first trading day in prior trading days
            tweet_dates = get_date_range(tweet_start_date, tweet_end_date)

            sentiment_relevant_for_trading_day = []
            average_sentiment = []      #Average twitter sentiment from tweets in the date range [prior_trading_date, current_trading_date)

            for tweet_date in tweet_dates:      #Get all tweets in the date range [prior_trading_date, current_trading_date)
                tweet_file = os.path.join(self.tweet_dir, stock_name, f"{tweet_date}.json")
                #print(tweet_file)
                if os.path.exists(tweet_file):
                    # Load tweets
                    with open(tweet_file, 'r') as f:
                        tweets = json.load(f)
                        #input(tweets)

                    # Process tweets from the day
                    for tweet in tweets:
                        sentiment = tweet['sentiment']
                        sentiment_relevant_for_trading_day.append(sentiment)

            #Calculate average twitter sentiment of stock tweets
            if sentiment_relevant_for_trading_day:
                average_sentiment = np.mean(sentiment_relevant_for_trading_day)

            #If no tweets use the prior trading days sentiment or neutral(3) if this is the first day of the lag
            else:
                if sentiment_sequence:
                    average_sentiment = sentiment_sequence[-1]
                else:
                    average_sentiment = 3    
            # Normalize sentiment
            average_sentiment = (average_sentiment - self.sentiment_mean) / self.sentiment_std
            sentiment_sequence.append(average_sentiment)
        
        #Apply neutral sentiment padding in cases of missing data.
        if len(sentiment_sequence) < self.day_lag:
            padding_length = self.day_lag - len(sentiment_sequence)
            sentiment_sequence = [3.0] * padding_length + sentiment_sequence

        # Convert sequences to tensors
        sentiment_sequence = torch.tensor(sentiment_sequence, dtype=torch.float32)
        price_sequence = torch.tensor(price_sequence, dtype=torch.float32)

        if len(sentiment_sequence) < 5 or len(price_sequence) < 5:
            with open('debug.json', 'w') as f:
                debug_object = {
                    'sentiment_sequence': sentiment_sequence.numpy().tolist(),
                    'price_movement_sequence': price_sequence.numpy().tolist(),
                    'Prior dates': prior_trading_dates,
                    'Stock': stock_name,
                    'target_date': target_date
                }
                json.dump(debug_object, f)

        x = price_sequence

        # Get target y for the target trading day
        y_movement = price_df.loc[price_df['Date'] == target_date_obj, 'Movement_Percent'].values[0]
        y = 1 if y_movement > 0.0055 else 0  # Define your threshold

        y_tensor = torch.tensor(y, dtype=torch.long)
        #input(x[1].size(), x[0].size())
        return x, y_tensor

    def get_input_size(self):
        return 6

    def __len__(self):
        return len(self.data_samples)



def debug_dataset(dataset, num_samples=5):
    """
    Debug function to present an overview of the dataset.
    
    Args:
        dataset: Dataset object to debug.
    
    Prints:
        Overview of the dataset with mean, max, and min of specified features,
        as well as the shape of inputs and outputs for a few samples.
    """
    from collections import defaultdict
    import numpy as np
    
    # Accumulators for statistics
    feature_stats = defaultdict(list)
    
    print("### Dataset Debug Information ###")
    print(f"Dataset Length: {len(dataset)}")
    print(f"Features: {dataset.dataset_cols}")
    
    # Collect statistics over all data samples
    for i in range(len(dataset)):
        x, y = dataset[i]
        
        sentiment_sequence, price_sequence = x
        twitter_features = sentiment_sequence.numpy()
        price_features = price_sequence.numpy()  # Convert to NumPy array for stats
        
        # Collect stats for each column in price features
        for col_idx, col_name in enumerate(dataset.dataset_cols[:1]):  # Only sentiment
            column_values = twitter_features
            feature_stats[col_name].append(column_values)
        
        for col_idx, col_name in enumerate(dataset.dataset_cols[1:]):  # Exclude sentiment
            column_values = price_features[:, col_idx]
            feature_stats[col_name].append(column_values)
        
        if i < num_samples:
            print(f"\nSample {i + 1}:")
            print(f"  Sentiment sequence shape: {sentiment_sequence.shape}")
            print(f"  Price sequence shape: {price_sequence.shape}")
            print(f"  Target (y) shape: {y.shape}")
            print(f"  Target (y): {y}")
        
        # Avoid processing the entire dataset for large datasets
        if i > 1000:
            print("\nProcessing only the first 1000 samples for statistics.")
            break
    
    # Calculate mean, max, min for each feature
    for col_name, values in feature_stats.items():
        all_values = np.concatenate(values)  # Flatten list of arrays
        print(f"\nFeature: {col_name}")
        print(f"  Mean: {np.mean(all_values):.4f}")
        print(f"  Std: {np.std(all_values):.4f}")
        print(f"  Max: {np.max(all_values):.4f}")
        print(f"  Min: {np.min(all_values):.4f}")
    
    print("\n### End of Dataset Debug Information ###")
                    

if __name__ == '__main__':
    dataset_class = NormSentimentAllPriceXPriceY

    #From config
    train_start_date = '2014-01-01'
    train_end_date = '2015-08-01'
    eval_start_date = '2015-08-01'
    eval_end_date = '2015-10-01'
    test_start_date = '2015-10-01'
    test_end_date = '2016-01-01'

    dataset_loader_args = {
            "twitter_root" : "/home/skage/projects/ikt450_deep-neural-networks/stocknet-project-ikt450/dataset/tweet/preprocessed-sentiment-json",
            "price_root" : "/home/skage/projects/ikt450_deep-neural-networks/stocknet-project-ikt450/dataset/price/preprocessed/csv",
            "day_lag" : 5,
            "tweets_per_day" : 2,
            "words_per_tweet" : 30,
            "sentiment_model": 'nlptown/bert-base-multilingual-uncased-sentiment'
        }

    dataset = dataset_class(
    start_date=train_start_date,
    end_date=train_end_date,
    **dataset_loader_args
    )

    debug_dataset(dataset)


                

if 0:
    import os
    import json
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from collections import defaultdict
    from datetime import datetime, timedelta

    twitter_root = '/root/projects/ikt450_deep-neural-networks/stocknet-project-ikt450/dataset/tweet/preprocessed-json'
    price_root = '/root/projects/ikt450_deep-neural-networks/stocknet-project-ikt450/dataset/price/preprocessed/csv'

    start_date = '2014-01-01'
    end_date = '2016-01-01'

    day_lag = 5  # Number of previous days considered
    tweets_per_day = 2
    words_per_tweet = 30

    tweet_lengths = []         # List to store lengths of individual tweets
    tweets_per_day_list = []   # List to store number of tweets per day
    words_per_tweet_list = []  # List to store number of words per tweet
    tweets_per_stock = defaultdict(int)  # Number of tweets per stock

    from tqdm import tqdm

    # Get the date range
    date_range = get_date_range(start_date, end_date)

    # Iterate over each stock
    for stock_name in tqdm(os.listdir(twitter_root), desc='Processing stocks'):
        stock_tweet_counts = []
        stock_tweet_lengths = []
        stock_words_per_tweet = []

        stock_dir = os.path.join(twitter_root, stock_name)
        if not os.path.isdir(stock_dir):
            continue  # Skip if not a directory

        # Iterate over each date
        for date in date_range:
            tweet_file = os.path.join(stock_dir, f"{date}.json")
            if os.path.exists(tweet_file):
                # Load tweets
                with open(tweet_file, 'r') as f:
                    tweets = json.load(f)
                
                tweets_on_date = len(tweets)
                stock_tweet_counts.append(tweets_on_date)
                tweets_per_day_list.append(tweets_on_date)

                for tweet in tweets:
                    words = tweet['text']
                    num_words = len(words)
                    tweet_lengths.append(num_words)
                    stock_tweet_lengths.append(num_words)

                    words_per_tweet_list.append(num_words)
            else:
                tweets_per_day_list.append(0)
                stock_tweet_counts.append(0)

        # Total tweets for this stock
        total_tweets_stock = sum(stock_tweet_counts)
        tweets_per_stock[stock_name] = total_tweets_stock

        import numpy as np

        # Average tweet length
        average_tweet_length = np.mean(tweet_lengths)
        median_tweet_length = np.median(tweet_lengths)

        # Average number of tweets per day
        average_tweets_per_day = np.mean(tweets_per_day_list)
        median_tweets_per_day = np.median(tweets_per_day_list)

        # Words per tweet statistics
        average_words_per_tweet = np.mean(words_per_tweet_list)
        median_words_per_tweet = np.median(words_per_tweet_list)

        print(f"Total number of stocks: {len(tweets_per_stock)}")
        print(f"Total number of tweets: {sum(tweets_per_stock.values())}")

        print(f"\nAverage tweet length (in words): {average_tweet_length:.2f}")
        print(f"Median tweet length (in words): {median_tweet_length}")

        print(f"\nAverage number of tweets per day: {average_tweets_per_day:.2f}")
        print(f"Median number of tweets per day: {median_tweets_per_day}")

        print(f"\nAverage words per tweet: {average_words_per_tweet:.2f}")
        print(f"Median words per tweet: {median_words_per_tweet}")