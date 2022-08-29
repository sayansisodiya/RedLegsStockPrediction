from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import pandas as pd
import unicodedata
import re
from tqdm import tqdm

END = '<END>'
UNK = '<UNK>'
PAD = '<PAD>'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def preprocess(tweet):
    """
    Summary: Removes symbols and punctuation from tweet

    Input: tweet - string representing tweet to analyze

    Return value: string storing preprocessed tweet
    """

    tweet = ''.join(w for w in unicodedata.normalize('NFD', tweet.lower().strip()) if unicodedata.category(w) != 'Mn')
    tweet = re.sub(r'[^a-zA-Z]+', ' ', tweet)
    res = []
    s = tweet.split(' ')
    for item in s:
      if item == '':
        continue
      else:

        res.append(item)
    return res


class TextDataset():

    def __init__(self, text_data, split, threshold, max_len, idx_word=None, word_idx=None):

        self.text_data = text_data
        assert split in {'train', 'test'}
        self.split = split
        self.threshold = threshold
        self.max_len = max_len

        self.idx_word = idx_word
        self.word_idx = word_idx
        if split == 'train':
            self.build_dictionary()
        self.vocab_size = len(self.word_idx)
        
        self.text_id = []
        self.encode_text()

    
    def build_dictionary(self): 
        assert self.split == 'train'

        self.idx_word = {0:PAD, 1:END, 2: UNK}
        self.word_idx = {PAD:0, END:1, UNK: 2}

        word_freq_dic = defaultdict()
        idx = 3
        for sample in self.text_data:
          for item in sample[1]:
            word = item.lower()
            word_freq_dic[word] = word_freq_dic.get(word, 0) + 1
        for word, word_freq in word_freq_dic.items():
          if word_freq >= self.threshold:
            if word not in self.word_idx:
              self.idx_word[idx] = word
              self.word_idx[word] = idx
              idx += 1


    def encode_text(self):

        for sample in self.text_data:
          id_list = []
          for word in sample[1]:
            if word in self.word_idx:
              id_list.append(self.word_idx[word])
            else:
              id_list.append(self.word_idx[UNK])
          id_list.append(self.word_idx[END])
          self.text_id.append(id_list)

    def get_text(self, idx):

        tweets_ids = self.text_id[idx]
        if len(tweets_ids) < self.max_len:
          tweets_ids += [self.word_idx[PAD]] * (self.max_len - len(tweets_ids))
        elif len(tweets_ids) > self.max_len:
          tweets_ids = tweets_ids[:self.max_len]
        return torch.tensor(tweets_ids)
    
    def get_label(self, idx):

        label = self.text_data[idx][0]
        if label == 1:
          return torch.tensor(1)
        else:
          return torch.tensor(0)

    def __len__(self):

        return len(self.text_data)
    
    def __getitem__(self, idx):

        return self.get_text(idx), self.get_label(idx)


class RNN(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, bidirectional, dropout, num_classes, pad_idx):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_dirs = 1 if bidirectional == False else 2

        self.embedding = nn.Embedding(vocab_size, embed_size, pad_idx)
        self.GRU = nn.GRU(input_size = embed_size ,hidden_size=self.hidden_size, num_layers = self.num_layers, dropout = dropout, bidirectional = bidirectional, batch_first = True)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(self.num_dirs*self.hidden_size, num_classes)
        self.softmax = nn.Softmax()


    def forward(self, texts):

        x = self.embedding(texts) 
        output, hidden = self.GRU(x)
        if self.num_dirs == 2:
          hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1) 
        x = self.dropout(hidden)
        x = self.linear(x)
        
        return x

def train_model(model, num_epochs, data_loader, optimizer, loss_func):
  """
  Summary: Trains RNN model

  Inputs: model - RNN model to be trained
          num_epochs - integer representing number of epochs model is to be trained for
          data_loader - DataLoader object used for training data
          optimizer - Optimizer for RNN model
          loss_func - loss function model is evaluated on

  Return value: None
  """
  print('Training RNN Model...')
  model.train()
  for epoch in range(num_epochs):
      e_loss = 0
      e_acc = 0
      for texts, labels in data_loader:
          texts = texts.to(device) 
          labels = labels.to(device)

          optimizer.zero_grad()

          output = model(texts)
          acc = accuracy(output, labels)
          
          loss = loss_func(output, labels)
          loss.backward()
          optimizer.step()

          e_loss += loss.item()
          e_acc += acc.item()
      print('[TRAIN]\t Epoch: {:2d}\t Loss: {:.4f}\t Train Accuracy: {:.2f}%'.format(epoch+1, e_loss/len(data_loader), 100*e_acc/len(data_loader)))
  print('Model Trained')


def evaluate(model, data_loader, criterion):
    """
    Summary: Evaluates RNN model performance on test dataset

    Inputs: model - RNN model to be evaluated
            data_loader - DataLoader object used to pass in testing data
            criterion - 
    """
    print('Evaluating model performance on the test dataset')
    model.eval()
    epoch_loss = 0
    epoch_acc = 0
    all_predictions = []
    for texts, labels in tqdm(data_loader):
        texts = texts.to(device)
        labels = labels.to(device)
        
        output = model(texts)
        acc = accuracy(output, labels)
        pred = output.argmax(dim=1)
        all_predictions.append(pred)
        
        loss = criterion(output, labels)
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()
    full_acc = 100*epoch_acc/len(data_loader)
    full_loss = epoch_loss/len(data_loader)
    print('[TEST]\t Loss: {:.4f}\t Accuracy: {:.2f}%'.format(full_loss, full_acc))
    predictions = torch.cat(all_predictions)
    return full_acc, full_acc, predictions

def accuracy(output, labels):
  """
  Summary: Calculates Accuracy of RNN model

  Input: output - Tensor representing sentiment of tweets predicted by RNN model
         labels - Tensor representing true sentiment of tweets

  Return value: Float with RNN model's accuracy score
  """
  pred = output.argmax(dim=1) 
  correct = (pred == labels).sum().float() 
  acc = correct / len(labels)
  return acc

def create_train_data():
  """
  Summary: Preprocesses data used for RNN model training

  Input: None

  Return value: Numpy array storing processed training data
  """
  
  data = pd.read_csv("stock_tweet_data.csv")
  data["Sentiment"] = data["Sentiment"].replace(-1,0)
  train_data = data.to_numpy()
  train_data = [(x[1], preprocess(x[0])) for x in train_data]
  return train_data


def load_rnn_model():
  """
  Summary: Loads RNN Model from memory using saved state_dict

  Inputs: None

  Return Value: Pytorch RNN model
  """

  print("loading RNN model")
  model = RNN(vocab_size = train_Ds.vocab_size,
                embed_size = 256, 
                hidden_size = 256, 
                num_layers = 2,
                bidirectional = True,
                dropout = 0.5,
                num_classes = 2,
                pad_idx = train_Ds.word_idx[PAD]) 
  model.load_state_dict( torch.load('models/rnn_state.pt', map_location=torch.device('cpu')))
  model.eval()
  print("loaded model")
  return model


def sentiment_analysis(text, model):
  """
  Summary: Analyzes sentiment of tweet using RNN model

  Inputs: text - string representing a tweet
          model - RNN model used in sentiment analysis

  Return Value: Float between 0 and 1 representing sentiment of the tweet, with 1 being fully positive and 0 fully negative
  """
  texts = [[1, preprocess(text)]]
  test_Ds = TextDataset(texts, 'test', 5, 100, train_Ds.idx_word, train_Ds.word_idx)
  test_loader = torch.utils.data.DataLoader(test_Ds, batch_size=1, shuffle=False, num_workers=1, drop_last=False)
  model.eval()
  confidence_score = 0
  m = nn.Softmax(dim = 1)
  for t, labels in test_loader:
    output = model(t)
    output = m(output)
    confidence_score = output[0, 1].item()
  return confidence_score
  

if __name__=='__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    data = pd.read_csv("stock_data.csv")
    data["Sentiment"] = data["Sentiment"].replace(-1,0)
    train_data = data.to_numpy()
    train_data = [(x[1], preprocess(x[0])) for x in train_data]

    print('Num. Train text_data:', len(train_data))
    print("data sample:")
    for x in random.sample(train_data, 2):
        print('Sample text:', x[1])
        print('Sample label:', x[0], '\n')
        

    THRESHOLD = 5 
    MAX_LEN = 100 
    BATCH_SIZE = 32

    train_Ds = TextDataset(train_data, 'train', THRESHOLD, MAX_LEN)
    train_loader = torch.utils.data.DataLoader(train_Ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, drop_last=True)

    rnn_model = RNN(vocab_size = train_Ds.vocab_size,
                embed_size = 256, 
                hidden_size = 256, 
                num_layers = 2,
                bidirectional = True,
                dropout = 0.5,
                num_classes = 2,
                pad_idx = train_Ds.word_idx[PAD]) 

    rnn_model = rnn_model.to(device)
    learning_rate = 5e-4 
    loss_func = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(rnn_model.parameters(), lr=learning_rate)

    N_EPOCHS = 15
    train_model(rnn_model, N_EPOCHS, train_loader, optimizer, loss_func)

    print("Saving RNN model....") 
    torch.save(rnn_model, "sa_rnn.pt")
    print("Saved!")

    text = [[1, 'aapl weekly options gamblers lose'], [1, 'aapl always looking improve'], [0, 'aapl worst trading hour summary last 20 days']]
    text = [(x[0], preprocess(x[1])) for x in text]
    test_Ds = TextDataset(text, 'test', THRESHOLD, MAX_LEN, train_Ds.idx_word, train_Ds.word_idx)
    test_loader = torch.utils.data.DataLoader(test_Ds, batch_size=1, shuffle=False, num_workers=1, drop_last=False)


    for text, labels in test_loader:
        text = text.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        output = rnn_model(text)
        pred = output.argmax(dim=1)
        print(pred, labels)