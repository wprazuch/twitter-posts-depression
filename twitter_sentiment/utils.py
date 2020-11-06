import pickle

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import models
import json
import pandas as pd

with open(r'artifacts/config.json') as json_file:
    CONFIG = json.load(json_file)


def preprocess_sequence_data(data):
    with open(CONFIG["tokenizer_path"], "rb") as input_file:
        tokenizer = pickle.load(input_file)

    data_tokenized = tokenizer.texts_to_sequences(data)
    data_padded = pad_sequences(data_tokenized, maxlen=CONFIG['MAXLEN'])
    return data_padded


def load_data():
    dataset_path = CONFIG["data_path"]
    df = pd.read_csv(dataset_path, engine='python')
    DATASET_COLUMNS = ["target", "ids", "date", "flag", "user", "TweetText"]
    df.columns = DATASET_COLUMNS

    data, target = df['TweetText'].values, df['target'].values
    target[target == 4] = 1

    return data, target


def load_model(model_type='simple_lstm'):
    if model_type == 'simple_lstm':
        model = models.load_model(CONFIG['simple_lstm_path'])

    return model


def evaluate_prediction(y_pred):
    with open(CONFIG["class_mapping_path"], "rb") as input_file:
        class_mapping = pickle.load(input_file)

    if y_pred < 0.5:
        return class_mapping[0]
    elif y_pred >= 0.5:
        return class_mapping[1]
