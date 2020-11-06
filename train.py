import numpy as np
import pandas as pd
from twitter_sentiment import utils
from twitter_sentiment.models import common

import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer

from tensorflow.keras.optimizers import Adam


def main():
    data, target = utils.load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        data, target, test_size=0.2, stratify=target, random_state=13)

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(X_train)

    X_train_ready, X_test_ready = utils.preprocess_sequence_data(
        X_train), utils.preprocess_sequence_data(X_test)

    model = common.simple_lstm()

    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss='binary_crossentropy', metrics=['accuracy'])

    history = model.fit(X_train_ready, y_train, epochs=4, batch_size=512, validation_split=0.2)

    model.evaluate(X_test_ready, y_test)

    model.save(r'models/simple_lstm.h5')


if __name__ == '__main__':

    main()
