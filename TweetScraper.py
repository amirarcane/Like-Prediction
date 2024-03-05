from datetime import date, timedelta
from time import sleep

import pandas as pd
import snscrape.modules.twitter as sntwitter
from tqdm.auto import tqdm


class TweetScraper:
    """
    A class for scraping tweets from Twitter for a given month and year, with a specified number of tweets per day.
    """

    def __init__(self, year, num_tweets_per_day):
        """
        Initializes the TweetScraper with the year and the number of tweets to scrape per day.

        Parameters:
        - year (int): The year for which to scrape tweets.
        - num_tweets_per_day (int): The number of tweets to scrape per day.
        """
        self.year = year
        self.num_tweets_per_day = num_tweets_per_day

    def grab_tweets(self, month):
        """
        Scrapes tweets for a specified month and saves them to a CSV file.

        Parameters:
        - month (int): The month for which to scrape tweets.
        """
        days_in_month = self.calculate_days_in_month(month)
        total_tweets = days_in_month * self.num_tweets_per_day
        tweets_list = []
        pbar = tqdm(total=total_tweets)

        since = date(self.year, month, 1)
        for _ in range(days_in_month):
            until = since + timedelta(days=1)
            tweets_list.extend(self.scrape_tweets_for_day(since, until, pbar))
            since = until

        pbar.close()
        self.save_to_csv(tweets_list, month)

    def calculate_days_in_month(self, month):
        """
        Calculates the number of days in a given month.

        Parameters:
        - month (int): The month number (1-12).

        Returns:
        - int: The number of days in the month.
        """
        if month == 2:
            return 29 if (self.year % 4 == 0 and self.year % 100 != 0) or (self.year % 400 == 0) else 28
        elif month in [4, 6, 9, 11]:
            return 30
        else:
            return 31

    def scrape_tweets_for_day(self, since, until, pbar):
        """
        Scrapes tweets for a single day.

        Parameters:
        - since (date): The start date.
        - until (date): The end date (exclusive).
        - pbar (tqdm): The progress bar object.

        Returns:
        - list: A list of tweets scraped for the day.
        """
        tweets_list = []
        query = f'since:{since.isoformat()} until:{until.isoformat()} lang:en'
        for tweet in sntwitter.TwitterSearchScraper(query).get_items():
            if tweet.inReplyToTweetId is not None or tweet.inReplyToUser is not None or len(
                    tweets_list) >= self.num_tweets_per_day:
                continue
            tweets_list.append(self.extract_tweet_data(tweet))
            sleep(0.01)
            pbar.update(1)
        return tweets_list

    @staticmethod
    def extract_tweet_data(tweet):
        """
        Extracts relevant data from a tweet object.

        Parameters:
        - tweet (Tweet): The tweet object.

        Returns:
        - list: A list containing relevant data from the tweet.
        """
        return [tweet.id, tweet.url, tweet.date, tweet.content,
                tweet.likeCount, tweet.replyCount, tweet.retweetCount,
                tweet.quoteCount, tweet.sourceLabel, tweet.links, tweet.media,
                tweet.quotedTweet, tweet.mentionedUsers, tweet.coordinates, tweet.place,
                tweet.hashtags, tweet.cashtags, tweet.card, tweet.vibe,
                tweet.user.username, tweet.user.description, tweet.user.favouritesCount,
                tweet.user.followersCount, tweet.user.friendsCount, tweet.user.location,
                tweet.user.verified, tweet.user.protected, tweet.user.mediaCount]

    def save_to_csv(self, tweets_list, month):
        """
        Saves the scraped tweets to a CSV file.

        Parameters:
        - tweets_list (list): The list of tweets to save.
        - month (int): The month number for naming the file.
        """
        df = pd.DataFrame(tweets_list, columns=['id', 'url', 'date', 'content',
                                                'likeCount', 'replyCount', 'retweetCount',
                                                'quoteCount', 'sourceLabel', 'links', 'media',
                                                'quotedTweet', 'mentionedUsers', 'coordinates', 'place',
                                                'hashtags', 'cashtags', 'card', 'vibe', 'username',
                                                'UserDescription', 'UserFavouritesCount', 'followersCount',
                                                'friendsCount', 'location', 'verified', 'protected', 'mediaCount'])
        df.to_csv(f'{month}-{self.year}.csv', encoding='utf-8', index=False)


# Example usage:
year = 2022
num_tweets_per_day = 3000
scraper = TweetScraper(year, num_tweets_per_day)
for month in range(1, 13):
    scraper.grab_tweets(month)
