import requests
import rnn_model
import pandas as pd
import json
from sklearn.model_selection import train_test_split


def create_stock_dataframe(stock, model, oneday):
    """
    Summary: Creates Pandas DataFrame consisting of dates, stock prices, average sentiment, and a 9 day moving average
    Inputs: stock - String representing stock symbol to be analyzed, 
            model - RNN model used to perform sentiment analysis
            oneday - Boolean value specifying if only the most recent day is needed, as in prediction

    Return value: Pandas DataFrame with columns Date, Price, MovingAvg, Sentiment
    """

    config = None
    with open("config.json") as config_file:
        config = json.load(config_file)
    stock_key = config["AlphaVantage"]
    twitter_headers = config["Twitter"]
    stock_dict = {}
    mode = "full"
    if oneday:
        mode = "compact"
    stock_response = GetStockPriceJson(stock, mode, stock_key)
    for date in stock_response['Time Series (Daily)']:
        stock_dict[date] = float(stock_response['Time Series (Daily)'][date]['5. adjusted close'])
    sentiment_dict = create_sentiment_dict(stock, model, twitter_headers, stock_key, oneday)
    sentiment_df = pd.DataFrame(sentiment_dict.items(), columns=["Date", "Sentiment"])
    stock_df = pd.DataFrame(stock_dict.items(), columns=["Date", "Price"])
    stock_df["MovingAvg"] = stock_df["Price"].ewm(9).mean().shift()
    prediction_df = pd.merge(stock_df, sentiment_df, how="left", on="Date")
    prediction_df.to_csv("stock_df.csv",index=False)
    return prediction_df


def get_days_data(stock, model):
    """
    Summary: Gets data from the most recent day recorded in stock/twitter APIs for prediction

    Inputs: stock - String representing stock symbol to be analyzed, 
            model - RNN model used to perform sentiment analysis

    Return value: Numpy Array representing the MovingAvg and Sentiment values of the most recent day recorded
    """
    print("Grabbing day's data")
    prediction_df = create_stock_dataframe(stock, model, oneday=True)
    last_date = prediction_df.Date.max()
    split_date = last_date.split("-")
    day = int(split_date[2])
    day += 1
    if day < 10:
        split_date[2] = "0" + str(day)
    else:
        split_date[2] = str(day)
    print("Prediction for " + "-".join(split_date))
    return prediction_df.loc[prediction_df["Date"] == last_date][["MovingAvg", "Sentiment"]].to_numpy()

def create_train_data(stock_df, verbose=True):
    """
    Summary: Creates Training and Testing Split of stock/sentiment data for training of XGBoost models

    Inputs: stock_df - Pandas DataFrame containing data about stock price, date, and daily tweet sentiment regarding that stock

    Return value: Numpy Arrays representing training and testing data
    """
    if verbose:
        print("Creating Training Data")
    stock_df["Date"] = pd.to_datetime(stock_df["Date"], infer_datetime_format=True)
    stock_df["MovingAvg"] = stock_df["Price"].ewm(9).mean().shift()
    stock_df['Price'] = stock_df['Price'].shift(-1)
    stock_df = stock_df[:-1]
    X = stock_df[["MovingAvg", "Sentiment"]]
    y = stock_df["Price"]
    print("Number of days used in training: " + str(X.shape[0]))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
    if verbose:
        print("Done Creating Training Data")
    return X_train, X_test, y_train, y_test


def create_sentiment_dict(stock, model, twitter_headers, stock_key, oneday=False, verbose=True):
    """
    Summary: Creates dictionary of date : average tweet sentiment pairs

    Inputs: stock - String representing stock symbol to be analyzed, 
            model - RNN model used to perform sentiment analysis
            oneday - Boolean specifying whether only most recent day's data is needed

    Return Value: Dictionary where each key is a string representation of a date and each value is 
    the average sentiment of the tweets on that day represented as a float between 0 and 1
    """
    if verbose:
        print("Creating Sentiment Dict")
    twitter_response = GetTweetJson(stock, twitter_headers, stock_key)
    month_map = {"Jan" : "01", "Feb": "02", "Mar" : "03", "Apr" : "04", "May": "05", "Jun" :"06", "Jul" : "07", "Aug" : "08", "Sep" : "09", "Oct":"10", "Nov":"11", "Dec":"12"}
    sum_sentiment_dict = {}
    count_dict = {}
    counter = 0 # counts number of requests made
    prev_date = ""
    broken = False
    while twitter_response['statuses'] and counter < 50: # cap of 5,000 tweets returned if num_results = 100
        if broken:
            break
        counter += 1
        try:
            for x in twitter_response['statuses']:
                line = x['full_text']
                if (line[0:4] == 'RT @'): # pulls full text from retweets
                    line = x.get('retweeted_status')
                    if line != None:
                        line = line['full_text']
                if line != None:
                    line = line.replace('\n','')
                    line = line.replace('\t','')
                    given_date = x['created_at'].split(" ")
                    month = month_map[given_date[1]]
                    day = given_date[2]
                    year = given_date[5]
                    tweet = line
                    date = month + "/" + day + "/" + year
                    if oneday:
                        if counter > 1 and date != prev_date:
                            broken = True
                            break
                    sentiment = rnn_model.sentiment_analysis(tweet, model) # Function to analyze sentiment of a tweet via rnn
                    if date in sum_sentiment_dict:
                        sum_sentiment_dict[date] += sentiment
                        count_dict[date] += 1
                    else:
                        sum_sentiment_dict[date] = sentiment
                        count_dict[date] = 1

                    prev_date = date
            next_url = 'https://api.twitter.com/1.1/search/tweets.json' + twitter_response['search_metadata']['next_results'] + '&tweet_mode=extended'
            response = requests.get(next_url, headers=twitter_headers)
            twitter_response = response.json()
            if twitter_response.get('statuses') == None:
                break
        except:
            raise Exception("Something went wrong in reading the file")
    sentiment_dict = {}
    for date in sum_sentiment_dict.keys():
        sentiment_dict[date] = sum_sentiment_dict[date]/count_dict[date]
    if verbose:
        print("Done with sentiment dict")
        print(str(counter) + " tweets used")
    return sentiment_dict
    

def GetStockPriceJson(symbol, size, stock_key):
    """
    Summary: Gets JSON response with stock data from AlphaVantage API

    Inputs: symbol - string representing stock's ticker symbol
            size - string denoting size of response, "compact" means 100 days' data will be returned, "full" means the full range
            stock_key - string representing AlphaVantage API Key

    Return Value: JSON object storing AlphaVantage API response
    """
    url = 'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol={}&outputsize={}&apikey={}'.format(symbol, size, stock_key)
    response = requests.get(url)
    response_json = response.json()
    return response_json


def GetTweetJson(symbol, headers, stock_key):
    """
    Summary: Gets JSON response with tweet data from Twitter API

    Inputs: symbol - string representing stock's ticker symbol
            headers - dictionary storing Twitter API headers
            stock_key - string representing AlphaVantage API Key

    Return Value: JSON object storing Twitter API response
    """
    num_results = 100
    if (len(symbol) > 5): # this request can't recognize if the passed symbol is a real stock or not, so the other request should be run first because it can
        print('\'{}\' is not a valid stock symbol!'.format(symbol))
    # raise ValueError('\'{}\' is not a valid stock symbol!'.format(symbol))
    company_name = GetCompanyName(symbol, stock_key)
    if company_name != None:
        q = company_name + ' OR ' + symbol
    else:
        q = symbol
        url = 'https://api.twitter.com/1.1/search/tweets.json?q={}&result_type=recent&lang=en&count={}&tweet_mode=extended'.format(q, num_results) # removed query param that omitted tweets with cashtags to increase volume of results
        response = requests.get(url, headers=headers)
        response_json = response.json()
    try:
        test = response_json['statuses'] # rate limit exceeded if exception is thrown here
    except:
        print('something went wrong while opening the file, or rate limit has been exceeded!')
        return "ERROR"
    return response_json


def GetCompanyName(symbol, stock_key):
    """
    Summary: Gets JSON response with tweet data from Twitter API

    Inputs: symbol - string representing a company's ticker symbol
            stock_key - string representing AlphaVantage API Key

    Return Value: string representing name of company
    """
    url = 'https://www.alphavantage.co/query?function=OVERVIEW&symbol={}&apikey={}'.format(symbol, stock_key)
    response = requests.get(url)
    response_json = response.json()
    company_name = response_json.get('Name')
    return company_name




