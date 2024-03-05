import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.special import softmax
from sklearn.preprocessing import MinMaxScaler
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
from wordcloud import WordCloud


class TweetDataProcessor:
    """
    A class to process tweet data for sentiment analysis, feature scaling, and visualization.
    """

    def __init__(self, data_path, model_name):
        """
        Initializes the TweetDataProcessor with the dataset and model to use.

        Parameters:
        - data_path (str): The path to the CSV file containing the tweet data.
        - model_name (str): The identifier for the pretrained model to use for sentiment analysis.
        """
        self.data_path = data_path
        self.model_name = model_name
        self.data = self.load_data()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.config = AutoConfig.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)

    def load_data(self):
        """
        Loads tweet data from a specified CSV file and selects relevant columns.

        Returns:
        - DataFrame: The loaded and preprocessed DataFrame.
        """
        selected_columns = ['content', 'likeCount', 'replyCount', 'retweetCount', 'viewCount',
                            'quoteCount', 'hashtags', 'UserFavouritesCount', 'followersCount',
                            'friendsCount', 'verified', 'mediaCount']
        data = pd.read_csv(self.data_path, usecols=selected_columns)
        data = data.dropna(subset=['content'])
        return data

    def preprocess_text(self, text):
        """
        Preprocesses tweet text by replacing usernames and URLs with placeholders.

        Parameters:
        - text (str): The text of the tweet to preprocess.

        Returns:
        - str: The preprocessed tweet text.
        """
        new_text = []
        for t in text.split(" "):
            t = '@user' if t.startswith('@') and len(t) > 1 else t
            t = 'http' if t.startswith('http') else t
            new_text.append(t)
        return " ".join(new_text)

    def process_row(self, tweet):
        """
        Processes a single tweet row for sentiment analysis using a pretrained model.

        Parameters:
        - tweet (str): The tweet text to analyze.

        Returns:
        - int: An integer representing the sentiment classification (0: negative, 1: neutral, 2: positive).
        """
        try:
            text = self.preprocess_text(tweet)
            encoded_input = self.tokenizer(text, return_tensors='pt')
            output = self.model(**encoded_input)
            scores = output[0][0].detach().numpy()
            scores = softmax(scores)
            ranking = np.argsort(scores)[::-1]
            result = self.config.id2label[ranking[0]]
            return {"negative": 0, "neutral": 1, "positive": 2}.get(result, 1)
        except Exception as e:
            print(e)
            return 1

    def add_sentiment_column(self):
        """
        Applies sentiment analysis to each row of the tweet content and adds a sentiment column to the DataFrame.
        """
        self.data['sentiment'] = self.data['content'].apply(self.process_row)

    def visualize_wordcloud(self):
        """
        Generates and displays a word cloud from the hashtags in the dataset.
        """
        hashtags = self.data['hashtags'].explode().dropna()
        text = " ".join(review for review in hashtags)
        wordcloud = WordCloud(background_color="white").generate(text)
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.show()

    def scale_and_save(self, output_path):
        """
        Performs MinMax scaling on numeric features and saves the processed DataFrame to a CSV file.

        Parameters:
        - output_path (str): The path where the scaled and cleaned data should be saved.
        """
        features_to_scale = ['likeCount', 'replyCount', 'retweetCount', 'viewCount', 'quoteCount',
                             'UserFavouritesCount', 'followersCount', 'friendsCount', 'mediaCount']
        scaler = MinMaxScaler()
        self.data[features_to_scale] = scaler.fit_transform(self.data[features_to_scale])
        self.data.to_csv(output_path, index=False)

    def generate_correlation_heatmap(self):
        """
        Generates and displays a heatmap of the correlations between features in the dataset.
        """
        correlations = self.data.corr()
        plt.figure(figsize=(10, 10))
        sns.heatmap(correlations, vmax=1.0, center=0, fmt='.2f', square=True, linewidths=.5, annot=True,
                    cbar_kws={"shrink": .70})
        plt.show()
