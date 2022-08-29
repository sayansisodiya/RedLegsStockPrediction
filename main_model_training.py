from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
import joblib
from data_utils import create_stock_dataframe, create_train_data
import sys
from rnn_model import load_rnn_model
from sklearn.model_selection import GridSearchCV
import numpy as np

    
def train_stock_models(stocks, rnn_model):
    """
    Summary: Trains XGBoost models for prediction and saves them as .dat models

    Inputs: Stocks-list of stock symbols to train model on

    Return value: None

    """
    for stock in stocks:
        training_df = create_stock_dataframe(stock, rnn_model, oneday=False)
        model = train_model(training_df, stock)
        joblib.dump(model, "models/" + stock + ".dat")
        

def train_model(training_df, stock):
    """
    Summary: Trains XGBoost model on stock prices

    Inputs: stock_df - Pandas DataFrame containing data about stock price, date, and daily tweet sentiment regarding that stock
            stock - String representing stock symbol to be used in training

    Return value: Trained XGBoost model

    """
    print("Beginning training model for ", stock)
    X_train, X_test, y_train, y_test = create_train_data(training_df)
    print("Created data")
    xgb = XGBRegressor(objective="reg:squarederror", random_state=42)
    parameters = {
        'n_estimators': [100, 200, 300, 400],
        'learning_rate': [0.001, 0.005, 0.01, 0.05],
        'max_depth': [8, 10, 12, 15],
        'gamma': [0.001, 0.005, 0.01, 0.02],
    }

    print("Performing Grid Search")
    gs = GridSearchCV(xgb, parameters)
    gs.fit(X_train, y_train, verbose=2)
    print("Grid Search Done")
    model = XGBRegressor(**gs.best_params_, objective="reg:squarederror")
    model.fit(X_train, y_train)
    print("Model fit")
    y_pred = model.predict(X_test)
    print(stock)
    print(f'y_true = {np.array(y_test)[:5]}')
    print(f'y_pred = {y_pred[:5]}')
    print(f'mean_squared_error = {mean_squared_error(y_test, y_pred)}')
    print("----------------")

    return model

if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise ValueError('Usage: python3 main_model_training.py stock1 stock2 etc..')
    else:
        rnn_model = load_rnn_model()
        stocks = sys.argv[1:]
        train_stock_models(stocks, rnn_model)
    
