#! /usr/bin/env python

from flask import Flask, request, jsonify, make_response 
from flask_cors import CORS

import pickle
import numpy as np
import pandas as pd
import sys
import os

import logging


# this is for logging with fluend

from fluent import sender
from fluent import event



logger = logging.getLogger(__name__)


# for dockerised version
sys.path.append('webservice')
module_path = os.path.abspath(os.path.join('.'))
if module_path not in sys.path:
    sys.path.append(module_path)


from webservice.process_data import pre_process_data


dir_name = os.path.dirname(__file__)
model_path = os.path.abspath(os.path.join(dir_name, 'static/credit_model.pkl'))
dict_vectoriser_path = os.path.abspath(os.path.join(dir_name, 'static/dict_vectorizer.pkl'))
data_scaler_path = os.path.abspath(os.path.join(dir_name, 'static/data_scaler.pkl'))

if 'credit_model' not in globals():
    with open(model_path, 'rb') as stream:
        credit_model = pickle.load(stream)
    # load the DictVectoriser
    with open(dict_vectoriser_path, 'rb') as stream:
        dict_vectoriser = pickle.load(stream)
    # load the data scaler
    with open(data_scaler_path, 'rb') as stream:
        data_scaler = pickle.load(stream)


app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.DEBUG)

# Configure the sender to send logs to fluend
sender.setup('webservice', host='fluentd', port=24224)


def get_dummy_features(data):
    """
    Create dummy features from data received from the request
    """
    tmp_data = {}
    for key in data.keys():
        tmp_data[key] = [int(data[key])] # Some browsers send strings instead of a number
    tmp_data = pd.DataFrame(tmp_data)
    tmp_data = pre_process_data(tmp_data)
    dummy_features = dict_vectoriser.transform(tmp_data.to_dict('records'))
    cols = dict_vectoriser.get_feature_names()
    # Convert the data back to data frame
    dummy_features = pd.DataFrame(dummy_features, index=tmp_data.index, columns=cols)
    new_cols = {}
    for key in cols:
        new_cols[key] = key.replace('=', '_')
    return data_scaler.transform(dummy_features.rename(columns=new_cols))


def get_predictions(data):
    """
    predict from data
    """
    features = get_dummy_features(data)
    prediction = credit_model.predict(features)
    probability = credit_model.predict_proba(features)
    # prepare the response
    result = {}
    result['credit-rating'] = 'Bad' if prediction[0] == 0 else 'Good'
    result['probabilities'] = {'Bad': np.round(probability[0][0], 3), 'Good': np.round(probability[0][1], 3)}
    return result


def prepare_response(result):
    """
    Create a response to send to the client
    """
    response = jsonify(result)
    response.status_code = 200
    response = make_response(response)
    response.headers['Access-Control-Allow-Origin'] = "*"
    response.headers['content-type'] = "application/json"
    return response


@app.route("/index")
def index():
    app.logger.info('access to the /index page!!!')
    return "Hello world"


@app.route("/another-route")
def another_route():
    # logging using JSON
    return "Yeeee, it works!!"


@app.route("/api/v1/credit-rating", methods=["POST"])
def get_credit_rating():
    if request.method == 'POST':
        # Get data posted as json
        app.logger.debug(request)
        data = request.get_json()
        result = get_predictions(data)
        event.Event('prediction', {'input': data, 'output': result})
        # Convert the response to a valid json content
        response = prepare_response(result)
        return response


if __name__ == "__main__":

    app.run(host='0.0.0.0', port='5000')
