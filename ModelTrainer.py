import re

import nltk
import numpy as np
import pandas as pd
from keras.layers import Dropout, Flatten, Dense, Input, concatenate, Conv1D, GlobalMaxPooling1D, MaxPooling1D
from keras.models import Sequential, Model
from keras.optimizers import Adam, Adamax, RMSprop
from keras.regularizers import l2
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from scipy.stats import randint
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from transformers import BertTokenizer, TFBertModel

nltk.download('stopwords')
nltk.download('wordnet')


class TweetModelTrainer:
    """
    A class to train models for tweet sentiment analysis using both text and numerical features.
    """

    def __init__(self, data_path):
        """
        Initializes the TweetModelTrainer with the path to the dataset.

        Parameters:
        - data_path (str): The path to the CSV file containing the tweet data.
        """
        self.data_path = data_path
        self.data = None
        self.text_train = None
        self.text_val = None
        self.text_test = None
        self.num_train = None
        self.num_val = None
        self.num_test = None
        self.labels_train = None
        self.labels_val = None
        self.labels_test = None
        self.max_length = None
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def load_and_preprocess_data(self):
        """
        Loads and preprocesses the tweet data from a CSV file.
        """
        df = pd.read_csv(self.data_path, nrows=1000)
        df['content'] = df['content'].astype(str)
        # Preprocess text
        stop_words = set(stopwords.words('english'))
        lemmatizer = WordNetLemmatizer()
        preprocess_steps = [
            (r'http\S+', ''),  # Remove URLs
            (r'@[^\s]+', ''),  # Remove mentions
            (r'#([^\s]+)', r'\1'),  # Remove hashtags
            (r'[^\w\s]', ''),  # Remove punctuation
        ]
        for pattern, repl in preprocess_steps:
            df['content'] = df['content'].apply(lambda x: re.sub(pattern, repl, x))
        df['content'] = df['content'].apply(
            lambda x: ' '.join([lemmatizer.lemmatize(word) for word in x.lower().split() if word not in stop_words]))

        # Remove rows with NaN values in specific columns
        df = df.dropna(subset=['content', 'likeCount', 'replyCount', 'retweetCount', 'viewCount', 'quoteCount'])

        # Tokenize and pad text data
        text_data = df['content'].apply(lambda x: self.tokenizer.encode(x, add_special_tokens=True))
        self.max_length = max(len(text) for text in text_data)
        text_data = pad_sequences(text_data, maxlen=self.max_length, dtype='int32', padding='post', truncating='post')

        # Split dataset
        numerical_data = df[['replyCount', 'retweetCount', 'viewCount', 'quoteCount']]
        labels = df['likeCount']
        self.split_dataset(text_data, numerical_data, labels)

    def split_dataset(self, text_data, numerical_data, labels):
        """
        Splits the dataset into training, validation, and test sets.

        Parameters:
        - text_data (np.array): The tokenized and padded text data.
        - numerical_data (DataFrame): The numerical features of the dataset.
        - labels (Series): The target variable (likeCount).
        """
        self.text_train, self.text_val, self.num_train, self.num_val, self.labels_train, self.labels_val = train_test_split(
            text_data, numerical_data, labels, test_size=0.2, random_state=42)
        self.text_val, self.text_test, self.num_val, self.num_test, self.labels_val, self.labels_test = train_test_split(
            self.text_val, self.num_val, self.labels_val, test_size=0.5, random_state=42)

    def train_bert_model(self):
        """
        Defines and trains the BERT model for text-based features.
        """
        text_input = Input(shape=(None,), dtype='int32')
        bert_model = TFBertModel.from_pretrained('bert-base-uncased')
        text_embedding = bert_model(text_input)[0]
        text_embedding = GlobalMaxPooling1D()(text_embedding)
        text_model = Model(inputs=text_input, outputs=text_embedding)
        text_model.compile(optimizer=Adamax(learning_rate=1e-4), loss='mean_squared_error', metrics=['mae'])
        text_model.fit(np.vstack(self.text_train), np.array(self.labels_train), batch_size=16, epochs=10,
                       validation_data=(np.vstack(self.text_val), np.array(self.labels_val)))
        # Evaluate the model
        score = text_model.evaluate(self.text_test, self.labels_test, batch_size=32)
        print("BERT Model - Test loss:", score[0], "Test MAE:", score[1])
        return text_model

    @staticmethod
    def create_cnn_model(lr=0.002, filters1=32, filters2=64, dropout1=0.2, dropout2=0.2, kernel_regularizer=0.01):
        """
        Defines the CNN model for numerical features.

        Parameters:
        - lr (float): Learning rate for the optimizer.
        - filters1 (int): Number of filters in the first Conv1D layer.
        - filters2 (int): Number of filters in the second Conv1D layer.
        - dropout1 (float): Dropout rate for the first Dropout layer.
        - dropout2 (float): Dropout rate for the second Dropout layer.
        - kernel_regularizer (float): Regularization factor for the Dense layer.

        Returns:
        - cnn_model (Model): The compiled CNN model.
        """
        cnn_model = Sequential()
        cnn_model.add(Conv1D(filters1, 3, activation='relu', input_shape=(4, 1), padding='same'))
        cnn_model.add(MaxPooling1D(2))
        cnn_model.add(Conv1D(filters2, 3, activation='relu', padding='same'))
        cnn_model.add(MaxPooling1D(2))
        cnn_model.add(Flatten())
        cnn_model.add(Dropout(dropout1))
        cnn_model.add(Dense(128, activation='relu', kernel_regularizer=l2(kernel_regularizer)))
        cnn_model.add(Dropout(dropout2))
        cnn_model.add(Dense(1))
        cnn_model.compile(optimizer=RMSprop(learning_rate=lr), loss='mean_squared_error', metrics=['mae'])
        return cnn_model

    def train_cnn_model(self):
        """
        Trains the CNN model with the numerical features of the dataset.
        """
        # Since we're using KerasRegressor which expects a function, create_cnn_model should not be an instance method.
        regressor = KerasRegressor(build_fn=self.create_cnn_model, verbose=0)
        param_dist = {
            'lr': [0.001, 0.002, 0.003],
            'filters1': randint(16, 65),
            'filters2': randint(32, 129),
            'dropout1': [0.1, 0.2, 0.3],
            'dropout2': [0.1, 0.2, 0.3],
            'kernel_regularizer': [0.001, 0.01, 0.1]
        }
        early_stop = EarlyStopping(monitor='val_loss', patience=20)
        random_search = RandomizedSearchCV(estimator=regressor, param_distributions=param_dist, n_iter=10, cv=3)
        random_search.fit(self.num_train, self.labels_train, batch_size=32, epochs=100,
                          validation_data=(self.num_val, self.labels_val), callbacks=[early_stop])
        best_params = random_search.best_params_
        print("Best Hyperparameters: ", best_params)

        # Train the model with the best hyperparameters
        cnn_model = self.create_cnn_model(**best_params)
        history = cnn_model.fit(self.num_train, self.labels_train, batch_size=32, epochs=10,
                                validation_data=(self.num_val, self.labels_val), callbacks=[early_stop])
        # Evaluate the model
        score = cnn_model.evaluate(self.num_test, self.labels_test, batch_size=32)
        print("CNN Model - Test loss:", score[0], "Test MAE:", score[1])
        return cnn_model

    def train_combined_model(self, text_model, cnn_model):
        """
        Trains a combined model using both text and numerical features.

        Parameters:
        - text_model (Model): The trained BERT model for text features.
        - cnn_model (Model): The trained CNN model for numerical features.
        """
        combined_input = concatenate([text_model.output, cnn_model.output])
        combined = Dense(64, activation='relu')(combined_input)
        output = Dense(1, activation='linear')(combined)
        model = Model(inputs=[text_model.input, cnn_model.input], outputs=output)
        model.compile(optimizer=Adam(learning_rate=1e-4), loss='mean_squared_error', metrics=['mae'])

        # Train the combined model
        history = model.fit([np.array(self.text_train), np.array(self.num_train)], np.array(self.labels_train),
                            batch_size=32, epochs=10,
                            validation_data=(
                                [np.array(self.text_val), np.array(self.num_val)], np.array(self.labels_val)))
        # Evaluate the model
        eval_metrics = model.evaluate([np.array(self.text_test), np.array(self.num_test)], np.array(self.labels_test),
                                      batch_size=32)
        print("Combined Model - Loss: {:.4f}, MAE: {:.4f}".format(eval_metrics[0], eval_metrics[1]))
