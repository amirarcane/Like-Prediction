# Twitter Like Prediction

## Overview
This project is an exploration into predicting the number of likes a tweet may receive based on its content and various numerical features. By leveraging a combination of BERT for text-based features and CNN for numerical features, this project aims to provide insights into what factors may influence the popularity of a tweet. This README outlines the process from dataset generation through feature selection, pre-processing, model architecture, training, and final evaluation.

## Dataset Generation
The dataset was meticulously generated to avoid biases related to specific events, dates, or topics, encompassing a wide range of tweets from January 1st, 2022, to December 30th, 2022. Given the limitations of the Twitter API for free subscriptions, the Snscrape library was utilized for data scraping, resulting in a comprehensive dataset of 547,484 tweets. This dataset, including features of tweets and user details, is available on Kaggle for public access and further research.

The dataset generated for this project is available on [Kaggle](https://www.kaggle.com/datasets/amirhosseinnaghshzan/twitter-2022), providing a valuable resource for researchers and enthusiasts interested in Twitter data analysis and like prediction.

![2022 hashtags](https://github.com/amirarcane/English-Tweets/assets/15520230/7f38ce90-68bf-49ba-bd2c-0ef498493862)


## Model Architecture and Training
The architecture leverages the strengths of BERT for textual data and CNN for numerical data. The models were separately trained and then concatenated to form a comprehensive model for like prediction. This approach, combining textual and numerical analysis, was optimized through extensive hyperparameter tuning and validated on a split dataset.

## Model Evaluation and Performance
Our dataset comprises a diverse range of tweets, with the number of likes varying significantly from 0 to several thousand. A critical aspect of our dataset is that the average number of likes per tweet is 44. This average is pivotal in setting benchmarks for model performance and expectations for predictive accuracy.

The final model, a sophisticated blend of BERT for textual analysis and CNN for numerical data interpretation achieved a Mean Absolute Error (MAE) of 9.07 likes. Given the average like count of 44, this MAE represents approximately 20.6% of the average like count.

The results demonstrate the potential of our approach to provide insights into tweet popularity, making it a valuable tool for researchers and practitioners interested in social media analytics.
