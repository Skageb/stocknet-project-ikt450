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
from sklearn.preprocessing import RobustScaler, StandardScaler


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

def get_all_stock_trading_dates(stock_name: str, price_dir: str) -> list:
    """
    Reads the stock price CSV file and returns a sorted list of unique trading dates 
    in 'YYYY-MM-DD' string format.

    Args:
        stock_name (str): The name of the stock.
        price_dir (str): The directory where the stock price CSV files are stored.

    Returns:
        list: A sorted list of trading dates as strings in 'YYYY-MM-DD' format.
    """
    price_file = os.path.join(price_dir, f'{stock_name}.csv')
    
    # Read only the 'Date' column and parse dates during read to save memory and time
    price_df = pd.read_csv(price_file, usecols=['Date'], parse_dates=['Date'])

    price_df['Date'] = pd.to_datetime(price_df['Date'])
    
    # Extract unique dates and sort them using pandas' optimized functions
    unique_dates_sorted = price_df['Date'].drop_duplicates().sort_values()

    return unique_dates_sorted.to_list()


def get_date_from_difference(date, offset_days:int):
    date_format = "%Y-%m-%d"
    given_date= datetime.strptime(date, date_format)
    
    offset_date = given_date + timedelta(days=offset_days)
    return offset_date.strftime(date_format)


def collect_stock_date_pairs(start_date, end_date, day_lag, price_data_dir, exclude_t_high= 0.0055, exclude_t_low = -0.005):
    '''Collect all stock date pairs where price movement for the date is outside of threshhold and minimum of 5 prior days exists.'''
    data_samples = []
    for stock_file in os.listdir(price_data_dir):
        stock_name = stock_file.split('.')[0]
        price_df = pd.read_csv(os.path.join(price_data_dir, stock_file))
        price_df['Date'] = pd.to_datetime(price_df['Date'])
        included_trades = 0
        for date in get_date_range(start_date, end_date):
            date_obj = pd.to_datetime(date)
            if date_obj in price_df['Date'].values:
                movement_percent = price_df.loc[price_df['Date'] == date_obj, 'Movement_Percent'].iloc[0]
                if movement_percent > exclude_t_high or movement_percent <= exclude_t_low:
                    if included_trades == day_lag:
                        data_samples.append((stock_name, date))
                    else:
                        included_trades += 1
    return data_samples


def get_all_values_tweet(keys:list, twitter_dir) -> tuple:
    '''Fetch all values for the given keys from the twitter files and return them as (key1_all_values, key2_all_values, ...)
    
    Typically used with keys = [sentiment]'''
    all_values_tuple = [[] for key in keys]
    for stock_name in os.listdir(twitter_dir):
        stock_tweet_dir = os.path.join(twitter_dir, stock_name)
        for tweet_file in os.listdir(stock_tweet_dir):
            if tweet_file.endswith('.json'):
                with open(os.path.join(stock_tweet_dir, tweet_file), 'r') as f:
                    tweets = json.load(f)
                    for idx, key in enumerate(keys):
                        all_values_tuple[idx].extend([tweet[key] for tweet in tweets])
    if len(all_values_tuple) == 1:
        return all_values_tuple[0]
    return all_values_tuple





def get_all_values_price(keys:list, stock_date_pairs:list, price_data_dir:str) -> tuple:
    '''Fetch all values for the given keys from the price files and return them as (key1_all_values, key2_all_values, ...)


    Example usage: 

    all_Movement_Percents, all_Volumes = get_all_values_price([Movement_Percent, Volume], data_samples, price_dir)'''
    all_values = [[] for key in keys]
    for stock_name, date in stock_date_pairs:
        stock_file = os.path.join(price_data_dir, f'{stock_name}.csv')
        price_df = pd.read_csv(stock_file)
        row = price_df.loc[price_df['Date'] == date]
        for idx, key in enumerate(keys):
            value = row[key].iloc[0]
            all_values[idx].append(float(value))
    return tuple(all_values)




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


def twitter_dates_from_trading_date(trading_date, all_stock_trading_dates:list):
    '''Generate dates in the range [prior_trading_date, trading_date)'''
    tweet_end_date = (trading_date - timedelta(days=1)).strftime("%Y-%m-%d")
    tweet_start_date = all_stock_trading_dates[all_stock_trading_dates.index(trading_date)-1].strftime("%Y-%m-%d")     #Get the prior valid trading date
    tweet_dates = get_date_range(tweet_start_date, tweet_end_date)
    return tweet_dates




def get_twitter_data(tweet_dates, tweet_dir, data_extractor):
    data = []
    for tweet_date in tweet_dates:      #Get all tweets in the date range [prior_trading_date, current_trading_date)
        tweet_file = os.path.join(tweet_dir, f"{tweet_date}.json")
        #print(tweet_file)
        if os.path.exists(tweet_file):
            # Load tweets
            with open(tweet_file, 'r') as f:
                tweets = json.load(f)
                #input(tweets)

            # Process tweets from the day
            data.extend([float(data_extractor(tweet)) for tweet in tweets])
    return data
        

def get_sentiment_from_tweet_object(tweet):
    sentiment = tweet['sentiment']
    return sentiment


class NormSentimentNormPriceXPriceY(Dataset):
    '''Dataset loader. The data used is twitter sentiment normalized and price movements with all available stock except close over the last five days'''
    def __init__(self, start_date, end_date, day_lag:int, tweets_per_day:int, words_per_tweet:int, twitter_root:str, price_root:str, **kwargs) -> None:
        self.tweet_dir = twitter_root
        self.price_dir = price_root
        self.day_lag = day_lag
        self.start_date = get_date_from_difference(start_date, 5)
        self.end_date = end_date

        self.words_per_tweet = words_per_tweet

        # Initialize lists to collect all values for the features to be normalized
        self.all_movement_percents = []
        self.all_volumes = []
        self.all_sentiments = []

        self.scalers = {
            'Sentiment': RobustScaler(),
            'Movement_Percent': RobustScaler(),
            'Open': RobustScaler(),
            'High': RobustScaler(),
            'Low': RobustScaler(),
            'Close': RobustScaler(),
            'Volume': RobustScaler(),
        }

        self.dataset_cols = ['Sentiment', 'Movement_Percent' ,'Open', 'High' ,'Low' , 'Volume' ]

        #Prepare dataset.
        self.__prepare_dataset__()

        self.scalers['Sentiment'].fit(np.array(self.all_sentiments).reshape(-1, 1))  #Fit Robust scaler for sentiment normalization

        for idx, feature in enumerate(self.dataset_cols[1:]):   #Fit Robust scalers for remaining features
            self.scalers[feature].fit(np.array(self.all_price_values[idx]).reshape(-1, 1))
        #print(len(self.all_movement_percents), len(self.all_volumes), len(self.all_sentiments))

        

        

        



    def __prepare_dataset__(self):
        """Prepares a list of (stock_name, target_date) pairs for indexing and get all feature values for features that are normalized."""
        self.data_samples = collect_stock_date_pairs(self.start_date, self.end_date, self.day_lag, self.price_dir)

        #Extract all feature values for normilization
        self.all_price_values = get_all_values_price(self.dataset_cols[self.get_input_size()[0]:], self.data_samples, self.price_dir)

        self.all_sentiments = get_all_values_tweet(keys=['sentiment'], twitter_dir=self.tweet_dir)

        #Create dict of all available trading dates for each stock
        self.all_stocks = list({stock_name for stock_name, _ in self.data_samples})
        self.trading_dates_stock_dict = {stock: get_all_stock_trading_dates(stock, self.price_dir) for stock in self.all_stocks}
    
    def __getitem__(self, idx):
        stock_name, target_date = self.data_samples[idx]
        target_date_obj = pd.to_datetime(target_date)

        stock_price_df = pd.read_csv(os.path.join(self.price_dir, f'{stock_name}.csv'))
        stock_price_df['Date'] = pd.to_datetime(stock_price_df['Date'])
        
        available_trading_days = self.trading_dates_stock_dict[stock_name]

        target_idx = available_trading_days.index(target_date_obj)        

        sentiment_sequence = []     #Feature 1
        price_sequence = []         #Feature 2

        dates_in_lag = available_trading_days[target_idx - self.day_lag: target_idx]

        for idx, prior_date in enumerate(dates_in_lag):
            prior_trade_date_obj = pd.to_datetime(prior_date)

            row = stock_price_df.loc[stock_price_df['Date'] == prior_trade_date_obj]
            
            if not row.empty:
                # Extract the required features
                movement_percent = row['Movement_Percent'].values[0]
                open_price = row['Open'].values[0]
                high_price = row['High'].values[0]
                low_price = row['Low'].values[0]
                #close_price = row['Close'].values[0]
                volume = row['Volume'].values[0]


                price_features = [movement_percent, open_price, high_price, low_price, volume]
                #Normalize
                for idx, feature in enumerate(self.dataset_cols[1:]):   
                    price_features[idx] = self.scalers[feature].transform(np.array(price_features[idx]).reshape(1, -1))
            else:
                # Handle missing data by inserting zeros or previous valid data
                price_features = [0, 0, 0, 0, 0, 0]

            price_sequence.append(price_features)

            
            
            
            twitter_dates = twitter_dates_from_trading_date(trading_date=prior_trade_date_obj, all_stock_trading_dates=self.trading_dates_stock_dict[stock_name])

            stock_tweet_dir = os.path.join(self.tweet_dir, stock_name)
            twitter_data = get_twitter_data(tweet_dates=twitter_dates, tweet_dir=stock_tweet_dir, data_extractor=get_sentiment_from_tweet_object)

            #average_sentiment = []      #Average twitter sentiment from tweets in the date range [prior_trading_date, current_trading_date)

            #Calculate average twitter sentiment of stock tweets from prior trading day to current trading day
            if twitter_data:
                average_sentiment = np.mean(twitter_data)

            #If no tweets use the prior trading days sentiment or neutral(3) if this is the first day of the lag
            else:
                if sentiment_sequence:
                    average_sentiment = sentiment_sequence[-1]
                else:
                    average_sentiment = 3    
            # Normalize sentiment
            
            sentiment_sequence.append(average_sentiment)
        
        #Apply neutral sentiment padding in cases of missing data.
        if len(sentiment_sequence) < self.day_lag:
            padding_length = self.day_lag - len(sentiment_sequence)
            sentiment_sequence = [3.0] * padding_length + sentiment_sequence


        #Normalize sentiment
        sentiment_sequence = self.scalers['Sentiment'].transform(np.array(sentiment_sequence).reshape(-1, 1))   
        # Convert sequences to tensors
        sentiment_sequence = torch.tensor(sentiment_sequence, dtype=torch.float32).squeeze()
        price_sequence = np.array(price_sequence) 
        price_sequence = torch.tensor(price_sequence, dtype=torch.float32).squeeze()

        x = (sentiment_sequence, price_sequence)

        # Get target y for the target trading day
        
        y_movement = stock_price_df.loc[stock_price_df['Date'] == target_date_obj, 'Movement_Percent'].values[0]
        #print(y_movement)
        y = 1 if y_movement > 0.0055 else 0  # Define your threshold

        y_tensor = torch.tensor(y, dtype=torch.long)

        #print(sentiment_sequence.shape, price_sequence.shape, y_tensor.shape)

        return x, y_tensor
    
    def get_input_size(self):
        return (1, 5)


    def __len__(self):
        return len(self.data_samples)
    

class TwitterSentimentVolumePriceXPriceY(Dataset):
    '''Dataset loader. The data used is twitter sentiment normalized and price movements with all available stock except close over the last five days'''
    def __init__(self, start_date, end_date, day_lag:int, tweets_per_day:int, words_per_tweet:int, twitter_root:str, price_root:str, **kwargs) -> None:
        self.tweet_dir = twitter_root
        self.price_dir = price_root
        self.day_lag = day_lag
        self.start_date = get_date_from_difference(start_date, day_lag)
        self.end_date = end_date

        self.n_twitter_features = 2
        self.n_price_features = 5

        self.words_per_tweet = words_per_tweet


        self.scalers = {
            'Sentiment': RobustScaler(),
            'Twitter_Volume': StandardScaler(),
            'Movement_Percent': RobustScaler(),
            'Open': RobustScaler(),
            'High': RobustScaler(),
            'Low': RobustScaler(),
            'Close': RobustScaler(),
            'Volume': RobustScaler(),
        }

        self.dataset_cols = ['Sentiment', 'Twitter_Volume', 'Movement_Percent' ,'Open', 'High' ,'Low' , 'Volume']

        #Prepare dataset.
        self.__prepare_dataset__()


        for idx, feature in enumerate(self.dataset_cols[:self.n_twitter_features]):   #Fit Robust scalers for Twitter features
            self.scalers[feature].fit(np.array(self.all_twitter_values[idx]).reshape(-1, 1))
            print(f'Normalization fitting done for feature {feature}')

        for idx, feature in enumerate(self.dataset_cols[self.n_twitter_features:]):   #Fit Robust scalers for remaining features
            self.scalers[feature].fit(np.array(self.all_price_values[idx]).reshape(-1, 1))
            print(f'Normalization fitting done for feature {feature}')
        #print(len(self.all_movement_percents), len(self.all_volumes), len(self.all_sentiments))

        

        
    def get_all_values_trading_date_interval_tweet(self) -> tuple:
        '''Helper function to extract all values from tweets following the interval [prior_trading_date, current_trading_date)
        
        Typically used with keys = [sentiment]'''
        all_values_tuple = [[],[]]  #[[sentiments], [twitter_vols]]
        for stock_name in os.listdir(self.tweet_dir):    #For all stocks
            stock_tweet_dir = os.path.join(self.tweet_dir, stock_name)
            for idx, trading_date in enumerate(self.trading_dates_stock_dict[stock_name][1:]):      #For all trading dates with prior trading dates
                tweet_dates_for_trading_date = twitter_dates_from_trading_date(trading_date, self.trading_dates_stock_dict[stock_name])

                sentiments = []
                tweet_vol = 0
                for tweet_date in tweet_dates_for_trading_date:     #For all dates since last trading day
                    if f'{tweet_date}.json' in os.listdir(stock_tweet_dir):
                        with open(os.path.join(stock_tweet_dir, f'{tweet_date}.json'), 'r') as f:
                            tweets = json.load(f)
                            sentiments += [tweet['sentiment'] for tweet in tweets]
                            tweet_vol += len(tweets)
                    else:
                        sentiments += [3]
                sentiment = np.mean(sentiments)
                all_values_tuple[0].append(sentiment)
                all_values_tuple[1].append(min(tweet_vol, 20))
        return all_values_tuple

    def get_twitter_data(self, tweet_dates_for_trading_date, stock_name):
        stock_tweet_dir = os.path.join(self.tweet_dir, stock_name)
        vol = 0
        sentiments = []
        for tweet_date in tweet_dates_for_trading_date:  
            if f'{tweet_date}.json' in os.listdir(stock_tweet_dir):
                with open(os.path.join(stock_tweet_dir, f'{tweet_date}.json'), 'r') as f:
                        tweets = json.load(f)
                        sentiments += [tweet['sentiment'] for tweet in tweets]
                        vol += len(tweets)
        average_sentiment = np.mean(sentiments) if len(sentiments) > 0 else None
        return [average_sentiment, min(vol, 20)]
        
           
        



    def __prepare_dataset__(self):
        """Prepares a list of (stock_name, target_date) pairs for indexing and get all feature values for features that are normalized."""
        self.data_samples = collect_stock_date_pairs(self.start_date, self.end_date, self.day_lag, self.price_dir)

        #Create dict of all available trading dates for each stock
        self.all_stocks = list({stock_name for stock_name, _ in self.data_samples})
        self.trading_dates_stock_dict = {stock: get_all_stock_trading_dates(stock, self.price_dir) for stock in self.all_stocks}

        #Extract all feature values for normilization
        self.all_price_values = get_all_values_price(self.dataset_cols[self.n_twitter_features:], self.data_samples, self.price_dir)

        self.all_twitter_values = self.get_all_values_trading_date_interval_tweet()

        
    
    def __getitem__(self, idx):
        stock_name, target_date = self.data_samples[idx]
        target_date_obj = pd.to_datetime(target_date)

        stock_price_df = pd.read_csv(os.path.join(self.price_dir, f'{stock_name}.csv'))
        stock_price_df['Date'] = pd.to_datetime(stock_price_df['Date'])
        
        available_trading_days = self.trading_dates_stock_dict[stock_name]

        target_idx = available_trading_days.index(target_date_obj)        

        twitter_sequence = []     #Feature 1
        price_sequence = []         #Feature 2

        dates_in_lag = available_trading_days[target_idx - self.day_lag: target_idx]

        for idx, prior_date in enumerate(dates_in_lag):
            prior_trade_date_obj = pd.to_datetime(prior_date)

            row = stock_price_df.loc[stock_price_df['Date'] == prior_trade_date_obj]
            
            if not row.empty:
                # Extract the required features
                movement_percent = row['Movement_Percent'].values[0]
                open_price = row['Open'].values[0]
                high_price = row['High'].values[0]
                low_price = row['Low'].values[0]
                #close_price = row['Close'].values[0]
                volume = row['Volume'].values[0]


                price_features = [movement_percent, open_price, high_price, low_price, volume]
                #Normalize
                for idx, feature in enumerate(self.dataset_cols[2:]):   
                    price_features[idx] = self.scalers[feature].transform(np.array(price_features[idx]).reshape(-1, 1))
            else:
                # Handle missing data by inserting zeros or previous valid data
                price_features = [0, 0, 0, 0, 0, 0]

            price_sequence.append(price_features)

            
            
            
            twitter_dates = twitter_dates_from_trading_date(trading_date=prior_trade_date_obj, all_stock_trading_dates=self.trading_dates_stock_dict[stock_name])

            stock_tweet_dir = os.path.join(self.tweet_dir, stock_name)
            twitter_data = self.get_twitter_data(tweet_dates_for_trading_date=twitter_dates, stock_name=stock_name)

            #input(twitter_data)

            #average_sentiment = []      #Average twitter sentiment from tweets in the date range [prior_trading_date, current_trading_date)

            #Calculate average twitter sentiment of stock tweets from prior trading day to current trading day
            if twitter_data[0] is None:
                #if twitter_sequence:
                #    twitter_data[0] = twitter_sequence[-1][0]
                #else:
                twitter_data[0] = 3.0

            #If no tweets use the prior trading days sentiment or neutral(3) if this is the first day of the lag
        
            #print(f'Twitter data prior to normalization: {twitter_data}')
            # Normalize sentiment
            for idx, feature in enumerate(self.dataset_cols[:self.n_twitter_features]):   
                twitter_data[idx] = self.scalers[feature].transform(np.array(twitter_data[idx]).reshape(-1, 1))

            #input(f'Twitter data post to normalization: {twitter_data}')
            
            twitter_sequence.append(twitter_data)
        
        #Apply neutral sentiment padding in cases of missing data.
        assert len(twitter_sequence) == self.day_lag


       
        # Convert sequences to tensors
        twitter_sequence = np.array(twitter_sequence) 
        twitter_sequence = torch.tensor(twitter_sequence, dtype=torch.float32).squeeze()
        price_sequence = np.array(price_sequence) 
        price_sequence = torch.tensor(price_sequence, dtype=torch.float32).squeeze()

        x = (twitter_sequence, price_sequence)

        # Get target y for the target trading day
        
        y_movement = stock_price_df.loc[stock_price_df['Date'] == target_date_obj, 'Movement_Percent'].values[0]
        #print(y_movement)
        y = 1 if y_movement > 0.0055 else 0  # Define your threshold

        y_tensor = torch.tensor(y, dtype=torch.long)

        #print(sentiment_sequence.shape, price_sequence.shape, y_tensor.shape)

        return x, y_tensor
    
    def get_input_size(self):
        return (self.n_twitter_features, self.n_price_features)


    def __len__(self):
        return len(self.data_samples)
    

class TwitterSentimentVolumePriceXPriceYRegression(Dataset):
    '''Dataset loader. The data used is twitter sentiment normalized and price movements with all available stock except close over the last five days'''
    def __init__(self, start_date, end_date, day_lag:int, tweets_per_day:int, words_per_tweet:int, twitter_root:str, price_root:str, **kwargs) -> None:
        self.tweet_dir = twitter_root
        self.price_dir = price_root
        self.day_lag = day_lag
        self.start_date = get_date_from_difference(start_date, day_lag)
        self.end_date = end_date

        self.n_twitter_features = 2
        self.n_price_features = 5

        self.words_per_tweet = words_per_tweet


        self.scalers = {
            'Sentiment': RobustScaler(),
            'Twitter_Volume': StandardScaler(),
            'Movement_Percent': RobustScaler(),
            'Open': RobustScaler(),
            'High': RobustScaler(),
            'Low': RobustScaler(),
            'Close': RobustScaler(),
            'Volume': RobustScaler(),
        }

        self.dataset_cols = ['Sentiment', 'Twitter_Volume', 'Movement_Percent' ,'Open', 'High' ,'Low' , 'Volume']

        #Prepare dataset.
        self.__prepare_dataset__()


        for idx, feature in enumerate(self.dataset_cols[:self.n_twitter_features]):   #Fit Robust scalers for Twitter features
            self.scalers[feature].fit(np.array(self.all_twitter_values[idx]).reshape(-1, 1))
            print(f'Normalization fitting done for feature {feature}')

        for idx, feature in enumerate(self.dataset_cols[self.n_twitter_features:]):   #Fit Robust scalers for remaining features
            self.scalers[feature].fit(np.array(self.all_price_values[idx]).reshape(-1, 1))
            print(f'Normalization fitting done for feature {feature}')
        #print(len(self.all_movement_percents), len(self.all_volumes), len(self.all_sentiments))

        

        
    def get_all_values_trading_date_interval_tweet(self) -> tuple:
        '''Helper function to extract all values from tweets following the interval [prior_trading_date, current_trading_date)
        
        Typically used with keys = [sentiment]'''
        all_values_tuple = [[],[]]  #[[sentiments], [twitter_vols]]
        for stock_name in os.listdir(self.tweet_dir):    #For all stocks
            stock_tweet_dir = os.path.join(self.tweet_dir, stock_name)
            for idx, trading_date in enumerate(self.trading_dates_stock_dict[stock_name][1:]):      #For all trading dates with prior trading dates
                tweet_dates_for_trading_date = twitter_dates_from_trading_date(trading_date, self.trading_dates_stock_dict[stock_name])

                sentiments = []
                tweet_vol = 0
                for tweet_date in tweet_dates_for_trading_date:     #For all dates since last trading day
                    if f'{tweet_date}.json' in os.listdir(stock_tweet_dir):
                        with open(os.path.join(stock_tweet_dir, f'{tweet_date}.json'), 'r') as f:
                            tweets = json.load(f)
                            sentiments += [tweet['sentiment'] for tweet in tweets]
                            tweet_vol += len(tweets)
                    else:
                        sentiments += [3]
                sentiment = np.mean(sentiments)
                all_values_tuple[0].append(sentiment)
                all_values_tuple[1].append(min(tweet_vol, 20))
        return all_values_tuple

    def get_twitter_data(self, tweet_dates_for_trading_date, stock_name):
        stock_tweet_dir = os.path.join(self.tweet_dir, stock_name)
        vol = 0
        sentiments = []
        for tweet_date in tweet_dates_for_trading_date:  
            if f'{tweet_date}.json' in os.listdir(stock_tweet_dir):
                with open(os.path.join(stock_tweet_dir, f'{tweet_date}.json'), 'r') as f:
                        tweets = json.load(f)
                        sentiments += [tweet['sentiment'] for tweet in tweets]
                        vol += len(tweets)
        average_sentiment = np.mean(sentiments) if len(sentiments) > 0 else None
        return [average_sentiment, min(vol, 20)]
        
           
        



    def __prepare_dataset__(self):
        """Prepares a list of (stock_name, target_date) pairs for indexing and get all feature values for features that are normalized."""
        self.data_samples = collect_stock_date_pairs(self.start_date, self.end_date, self.day_lag, self.price_dir)

        #Create dict of all available trading dates for each stock
        self.all_stocks = list({stock_name for stock_name, _ in self.data_samples})
        self.trading_dates_stock_dict = {stock: get_all_stock_trading_dates(stock, self.price_dir) for stock in self.all_stocks}

        #Extract all feature values for normilization
        self.all_price_values = get_all_values_price(self.dataset_cols[self.n_twitter_features:], self.data_samples, self.price_dir)

        self.all_twitter_values = self.get_all_values_trading_date_interval_tweet()

        
    
    def __getitem__(self, idx):
        stock_name, target_date = self.data_samples[idx]
        target_date_obj = pd.to_datetime(target_date)

        stock_price_df = pd.read_csv(os.path.join(self.price_dir, f'{stock_name}.csv'))
        stock_price_df['Date'] = pd.to_datetime(stock_price_df['Date'])
        
        available_trading_days = self.trading_dates_stock_dict[stock_name]
        target_idx = available_trading_days.index(target_date_obj)        

        twitter_sequence = []
        price_sequence = []

        dates_in_lag = available_trading_days[target_idx - self.day_lag: target_idx]

        for prior_date in dates_in_lag:
            prior_trade_date_obj = pd.to_datetime(prior_date)
            row = stock_price_df.loc[stock_price_df['Date'] == prior_trade_date_obj]
            if not row.empty:
                price_features = [
                    row['Movement_Percent'].values[0],
                    row['Open'].values[0],
                    row['High'].values[0],
                    row['Low'].values[0],
                    row['Volume'].values[0]
                ]
                for idx, feature in enumerate(self.dataset_cols[2:]):   
                    price_features[idx] = self.scalers[feature].transform(np.array(price_features[idx]).reshape(-1, 1))
            else:
                price_features = [0, 0, 0, 0, 0]

            price_sequence.append(price_features)
            
            twitter_dates = twitter_dates_from_trading_date(trading_date=prior_trade_date_obj, all_stock_trading_dates=self.trading_dates_stock_dict[stock_name])
            twitter_data = self.get_twitter_data(tweet_dates_for_trading_date=twitter_dates, stock_name=stock_name)

            if twitter_data[0] is None:
                twitter_data[0] = 3.0

            for idx, feature in enumerate(self.dataset_cols[:self.n_twitter_features]):   
                twitter_data[idx] = self.scalers[feature].transform(np.array(twitter_data[idx]).reshape(-1, 1))
            
            twitter_sequence.append(twitter_data)

        assert len(twitter_sequence) == self.day_lag

        twitter_sequence = torch.tensor(np.array(twitter_sequence), dtype=torch.float32).squeeze()
        price_sequence = torch.tensor(np.array(price_sequence), dtype=torch.float32).squeeze()

        x = (twitter_sequence, price_sequence)

        # Ensure target y is a float tensor
        y_movement = stock_price_df.loc[stock_price_df['Date'] == target_date_obj, 'Movement_Percent'].values[0]
        y = torch.tensor(y_movement, dtype=torch.float32)  # Change dtype to torch.float32

        return x, y

    
    def get_input_size(self):
        return (self.n_twitter_features, self.n_price_features)


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
        for col_idx, col_name in enumerate(dataset.dataset_cols[:2]):  # Only sentiment
            column_values = twitter_features[:, col_idx]
            feature_stats[col_name].append(column_values)
        
        for col_idx, col_name in enumerate(dataset.dataset_cols[2:]):  # Exclude sentiment
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
    dataset_class = TwitterSentimentVolumePriceXPriceY

    #From config
    train_start_date = '2014-01-01'
    train_end_date = '2015-08-01'
    eval_start_date = '2015-08-01'
    eval_end_date = '2015-10-01'
    test_start_date = '2015-10-01'
    test_end_date = '2016-01-01'

    dataset_loader_args = {
            "twitter_root" : "C:/Users/andre/Documents/Skole/2024H/IKT450-Prosjekt/dataset/tweet/preprocessed-sentiment-json",
            "price_root" : "dataset/price/preprocessed/csv",
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


                
