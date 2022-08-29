import os
import sys
from xgb_utils import load_model
from data_utils import get_days_data
from rnn_model import load_rnn_model


def make_prediction(stock, rnn_model):
    """
    Summary: Predicts the next day's price of a stock

    Inputs: stock - String representing stock symbol to be analyzed 
            model - RNN model used to perform sentiment analysis

    Return value: String representing the next day's predicted stock price
    """
    print("Running prediction for " + stock)
    stocks = []
    for model in os.listdir("models"):
        name = model.split(".")[0]
        stocks.append(name)
        if name == stock:
            model = load_model(stock)
            prediction_data = get_days_data(stock, rnn_model)
            return str(model.predict(prediction_data)[0])
    raise ValueError("Stock currently not supported, current stocks: " + str(stocks))



if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise ValueError('you can only pass one argument, which should be a stock symbol')
    stock = sys.argv[1]
    model = load_rnn_model()
    print(make_prediction(stock, model))