import joblib

def load_model(stock):
    """
    Summary: Loads trained XGBoost model into memory

    Input: stock - stock model used for prediction

    Return value: Trained XGBoost model
    """
    return joblib.load("models/" + stock + ".dat")