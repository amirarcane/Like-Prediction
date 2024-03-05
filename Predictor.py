import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV


class Predictor:
    """
    A class for training and evaluating a RandomForestRegressor model, including
    preprocessing datasets and predicting missing values.
    """

    def __init__(self, values_path, nan_path):
        """
        Initializes the RandomForestModel with paths to the datasets.

        Parameters:
        - values_path (str): The path to the CSV file containing the dataset with values.
        - nan_path (str): The path to the CSV file containing the dataset with missing values.
        """
        self.values_path = values_path
        self.nan_path = nan_path
        self.model = None

    def load_and_preprocess_data(self):
        """
        Loads the datasets and preprocesses them by dropping rows with missing target
        values and splitting into features and targets.
        """
        df_with_values = pd.read_csv(self.values_path)
        df_with_values.dropna(subset=['viewCount', 'likeCount'], inplace=True)

        X = df_with_values[['likeCount']]
        y = df_with_values['viewCount']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test

    def hyperparameter_tuning(self, X_train, y_train):
        """
        Performs hyperparameter tuning using GridSearchCV on a RandomForestRegressor.

        Parameters:
        - X_train (DataFrame): Training features.
        - y_train (Series): Training target variable.
        """
        param_grid = {
            'n_estimators': [50, 100, 150],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        rf_model = RandomForestRegressor(random_state=42)
        grid_search = GridSearchCV(rf_model, param_grid, cv=5)
        grid_search.fit(X_train, y_train)
        self.model = RandomForestRegressor(
            n_estimators=grid_search.best_params_['n_estimators'],
            max_depth=grid_search.best_params_['max_depth'],
            min_samples_split=grid_search.best_params_['min_samples_split'],
            min_samples_leaf=grid_search.best_params_['min_samples_leaf'],
            random_state=42
        )
        self.model.fit(X_train, y_train)
        print("Best parameters: ", grid_search.best_params_)

    def evaluate_model(self, X_test, y_test):
        """
        Evaluates the RandomForestRegressor model on the test set.

        Parameters:
        - X_test (DataFrame): Testing features.
        - y_test (Series): Testing target variable.
        """
        y_pred = self.model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print("Mean Squared Error (MSE):", mse)
        print("Root Mean Squared Error (RMSE):", rmse)
        print("Mean Absolute Error (MAE):", mae)
        print("R-squared:", r2)

    def predict_and_fill_missing_values(self):
        """
        Predicts missing values in the dataset with NaN values and fills them.
        """
        df_with_nan = pd.read_csv(self.nan_path)
        df_with_nan['likeCount'] = pd.to_numeric(df_with_nan['likeCount'], errors='coerce')
        df_with_nan.dropna(subset=['likeCount'], inplace=True)

        X_pred = df_with_nan[['likeCount']]
        viewCount_pred = self.model.predict(X_pred)

        df_with_nan['viewCount'] = viewCount_pred
        df_with_nan.to_csv('dataset_with_nan_filled.csv', index=False)
        print("Filled missing values and saved to 'dataset_with_nan_filled.csv'.")
