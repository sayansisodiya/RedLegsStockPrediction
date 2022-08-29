# RedlegsStockPrediction
Made by Sumay Thakurdesai, Xinyi He, and Sayan Sisodiya. 

## How it Works
Prediction utilizes two models. The model that predicts the stock price is trained on data consisting of a stock's daily closing price, a nine day moving average of that stock's closing price, and the average sentiment of tweets regarding that stock on each day represented in our data. The average sentiment is calculated through a 2-layer Bidirectional Recurrent Neural Network. Stock prices are obtained through AlphaVantage's free API, and Twitter data from Twitter's free API

## Limitations on Functionality
Our application "out of the box" can only predict stock prices for the companies Apple, Tesla, Meta (formerly Facebook; the stock symbol is still FB), and Amazon. This is due to time and space limitations for training and storing the machine learning models associated with each stock. More stocks can be added by running ```python3 main_model_training.py stock1 stock2 etc...``` with 1+ stocks. Note training does take a while, about 45 minutes per stock on Sumay's machine. Also, the models that are provided here do not take full advantage of the tweet sentiment field because we did not have access to more than 7 days of Twitter data with our free API key.

## Getting Started
You must have Python3 installed to use our application. To build our application, clone the GitHub repository into your local environment. Then, from the ```final-project-redlegs``` directory in the terminal, run the command ```pip install -r requirements.txt``` to install all of our application's dependencies.

To run the application once this is complete, run ```python3 predict_prices.py``` file from the terminal with a stock symbol as the sole argument. For instance, if you wanted to get the prediction for Apple's stock, you would run the command ```python3 predict_prices.py AAPL```. 
