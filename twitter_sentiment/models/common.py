from tensorflow.keras import layers, Sequential
import pickle
import json

with open(r'artifacts/config.json') as json_file:
    CONFIG = json.load(json_file)


def simple_lstm(num_words=None, max_len=None):
    if num_words is None:
        with open(CONFIG["tokenizer_path"], "rb") as input_file:
            tokenizer = pickle.load(input_file)
            num_words = len(tokenizer.index_word.keys())+1

    if max_len is None:
        max_len = CONFIG['MAXLEN']

    model = Sequential()
    model.add(layers.Embedding(num_words, 16, input_length=max_len))
    model.add(layers.LSTM(32))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1, activation='sigmoid'))
    return model
