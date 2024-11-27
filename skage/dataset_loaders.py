import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import pandas as pd
import json
import os
from config import cfg
from transformers import BertTokenizer


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
        y = 1 if price_movement > 0 else 0

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
                        

                    




                

if __name__ == "__main__":
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