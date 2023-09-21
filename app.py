# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 00:20:34 2023

@author: DELL
"""

import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle as pkl

app = Flask(__name__)

model = pkl.load(open("model.pkl","rb"))

@app.route("/")
def Home():
    return render_template("index.html")

@app.route("/predict",methods=["POST"])
def predict():
    features = [np.array()]
    prediction = model.predict(features)
    
    return render_template("index.html", prediction_text = "The mushroom is {}".format(prediction))

if __name__ == "__main__":
    app.run(debug=True)