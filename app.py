# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 18:14:18 2020

@author: MOHAN
"""


from flask import Flask, render_template, url_for, request
import pandas as pd

import pickle

load_model = pickle.load(open('random_forest_regression_model.pkl', 'rb'))
app = Flask(__name__)

@app.route('/')
def home():

    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    df = pd.read_csv(r'E:\Data Scientist\Projects\Machine Learning Projects\Air_Quality_Index\Working\Deployment\real_2018.csv')
    my_prediction=load_model.predict(df.iloc[:,:-1].values)
    my_prediction=my_prediction.tolist()
    return render_template('result.html', prediction=my_prediction)

if __name__ == '__main__':
    app.run(debug=True)
