#! /usr/bin/env python

from flask import Flask, request

import pickle
import numpy as np
import pandas as pd
import sys
import os


# for dockerised version
sys.path.append('webservice')
module_path = os.path.abspath(os.path.join('.'))
if module_path not in sys.path:
    sys.path.append(module_path)



dir_name = os.path.dirname(__file__)
model_path = os.path.abspath(os.path.join(dir_name, 'static/credit_model.pkl'))


if 'credit_model' not in globals():
    with open(model_path, 'rb') as stream:
        credit_model = pickle.load(stream)


app = Flask(__name__)



def get_dummy_features(features):
    """
    Create dummy features from a feature dictionary
    """
    features_dictionary = {'Account Balance': ['NoAccount', 'NoBalance', 'SomeBalance'],
                       'Payment Status': ['NoProblem', 'SomeProblems'],
                       'Savings/Stock Value':['AboveThousand', 'BellowHundred', 'NoSavings','Other'],
                       'Employment Length': ['AboveSevent', 'BellowOneYear', 'FourToSevent', 'OneToFour'],
                       'Sex & Marital Status':['Female', 'MaleMarried', 'MaleSingle'],
                       'NumberCredits': ['One', 'OnePlus'],
                       'Guarantors': ['No','Yes'],
                       'Concurrent Credits': ['NoCredit', 'OtherBanks'],
                       'Purpose':['HouseRelated', 'NewCar', 'Other', 'UsedCar'],
                       'AgeGroups':['MidAgeAdult', 'OldAdult', 'Senior', 'Young']
                      }

    dummy_features = {}
    for key in features.keys():
        for cat in features_dictionary[key]:
            if cat == features[key]:
                dummy_features['{}_{}'.format(key, cat)] = [1]
            else:
                dummy_features['{}_{}'.format(key, cat)] = [0]

    return dummy_features



@app.route("/index")
def index():
    return "Hello world"



@app.route("/another-route")
def another_route():
    return "Yeeee, it works!!"



@app.route("/api/v1/credit-rating", methods=["POST"])
def get_credit_rating():
    if request.method == 'POST':
        # Get data posted as json
        data = request.get_json()
        print(data)
        dummy_features = get_dummy_features(data)
        prediction = credit_model.predict(pd.DataFrame(dummy_features))
        return 'Bad' if prediction[0] == 0 else 'Good'


if __name__ == "__main__":

    app.run(host='0.0.0.0', port='5000')
