from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split
#import matplotlib.pyplot as plt


app = Flask(__name__, template_folder = 'template')

@app.route("/", methods=["GET", "POST"])
def index():
    return render_template('index.html')

@app.route("/search", methods=["GET", "POST"])
def getPredictedHosuePrice():
    area = request.form.get('area')
    room = request.form.get('room')
    houre_price = predictHousePrice(area, room)
    
    return render_template('result.html', name=houre_price)
    # return houre_price

def predictHousePrice(area, room):
    df = pd.read_csv('data/homeprices.csv')
    
    df.bedrooms.median()
    df.bedrooms = df.bedrooms.fillna(df.bedrooms.median())

    X_train, X_test, y_train, y_test = train_test_split(df.drop('price',axis='columns'), df.price, test_size=0.2, random_state=10)

    reg = linear_model.LinearRegression()
    reg.fit(X_train,y_train)

    price = reg.predict([[area, room]])
    rounded = np.round(price[0], 2)
    #score_test = reg.score(X_test, y_test)
    
    return rounded



if __name__ == "__main__":
    app.run(debug=True)