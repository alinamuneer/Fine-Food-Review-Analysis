import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error

import sys
sys.path.append('.')

from settings import PROCESSED_DATA_DIR, REVIEW_FILE_PATH

import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def get_reviews_per_day(df: pd.DataFrame) -> pd.DataFrame:
    reviews_per_day = df.resample('D').size()
    return reviews_per_day

def create_dataset(reviews_per_day: pd.Series, n_days: int = 20) -> tuple[pd.DataFrame, pd.Series]:
    """
    Create a dataset for time series forecasting.

    Parameters:
        reviews_per_day (pd.Series): Series containing daily reviews.
        n_days (int): Number of past days to consider as features. Default is 20.

    Returns:
        tuple[pd.DataFrame, pd.Series]: Feature matrix X and target vector y.
    """
    X = pd.DataFrame()
    for i in range(n_days):
        X[f'day_{i+1}'] = reviews_per_day.shift(i)

    y = reviews_per_day.shift(-1)  # Target variable is the next day's reviews

    return X.iloc[:-1], y.dropna()

def split_train_test(X_train: pd.DataFrame, y_train: 
                     pd.Series) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split the data into training and testing sets.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]: Split datasets (X_train, X_test, y_train, y_test).
    """
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def train_evaluate_model(X_train: pd.DataFrame, y_train: pd.Series,
                            X_test: pd.DataFrame, y_test: pd.Series) -> RandomForestRegressor:
    """
    Train and evaluate a Random Forest model.

    Returns:
        RandomForestRegressor: Trained Random Forest model.
    """
    # Initialize and train the model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred_train = model.predict(X_train)
    train_rmse = mean_squared_error(y_train, y_pred_train, squared=False)

    y_pred_test = model.predict(X_test)
    test_rmse = mean_squared_error(y_test, y_pred_test, squared=False)
    
    print(f"RMSE - Train set: {train_rmse}")
    print(f"RMSE - Test set: {test_rmse}")

    return model

def make_predictions(model: RandomForestRegressor, X_test: pd.DataFrame) -> pd.Series:
    """
    Make predictions for X_test using the trained Random Forest model.

    Parameters:
        model (RandomForestRegressor): Trained Random Forest model.
        X_test (pd.DataFrame): Feature matrix for testing.

    Returns:
        pd.Series: Predicted values for X_test.
    """
    y_pred = model.predict(X_test)
    return pd.Series(y_pred, index=X_test.index)


def plot_predicted_vs_actual(actual: pd.Series, predicted: pd.Series, title: str = "Predicted vs Actual") -> None:
    """
    Plot a time series of predicted vs actual values.

    Parameters:
        actual (pd.Series): Series containing the actual values.
        predicted (pd.Series): Series containing the predicted values.
        title (str): Title of the plot. Default is "Predicted vs Actual".

    Returns:
        None
    """
    plt.figure(figsize=(10, 6))
    plt.plot(actual.index, actual, 'o', label='Actual', color='blue')
    plt.plot(predicted.index, predicted, 'o', label='Predicted', color='red')
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.show()



def main():
    # Load your DataFrame here
    data_name = str(REVIEW_FILE_PATH).split('\\')[-1].split('.')[0]
    data_path = PROCESSED_DATA_DIR / f'{data_name}_processed.csv'

    df = pd.read_csv(data_path)

    # convert to datetime
    df['Time'] = pd.to_datetime(df['Time'], unit='s')
    df.set_index('Time', inplace=True)

    reviews_per_day = get_reviews_per_day(df)
    X, y = create_dataset(reviews_per_day=reviews_per_day)

    X_train, X_test, y_train, y_test = split_train_test(X, y)
    model = train_evaluate_model(X_train, y_train, X_test, y_test)
    predicted_reviews = make_predictions(model, X_test)
    print(f"Predicted number of reviews for the next day: {predicted_reviews}")

    plot_predicted_vs_actual(y_test, predicted_reviews, title="Predicted vs Actual number of reviews")



if __name__ == "__main__":
    main()