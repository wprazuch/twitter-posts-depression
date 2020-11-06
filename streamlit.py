import numpy as np
import pandas as pd

import streamlit as st
from twitter_sentiment import utils


def load_model_wrapper(model_type='simple_lstm'):
    model = utils.load_model(model_type)
    return model


model = load_model_wrapper()

user_input = st.text_area("Enter Tweet to check:", "")


if user_input != "":
    user_input = np.array([user_input])
    input_processed = utils.preprocess_sequence_data(user_input)
    y_pred = model.predict(input_processed)
    result = utils.evaluate_prediction(y_pred)
    st.text(result)
