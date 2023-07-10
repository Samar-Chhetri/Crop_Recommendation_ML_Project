from flask import Flask, render_template, request

import pandas as pd
import numpy as np

from src.exception import CustomException
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application

#Route for home page
@app.route('/')
def index():
    return render_template('index.html')

# Route for prediction
@app.route('/predictcrop', methods= ['GET', 'POST'])
def predict_crop():
    if request.method =='GET':
        return render_template('home.html')
    else:
        data = CustomData(
            N = int(request.form.get('N')),
            P = int(request.form.get('P')),
            K = int(request.form.get('K')),
            temperature = float(request.form.get('temperature')),
            humidity = float(request.form.get('humidity')),
            ph = float(request.form.get('ph')),
            rainfall = float(request.form.get('rainfall'))
        )

        pred_df = data.get_data_as_dataframe()
        print(pred_df)

        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)
        result = results[0]

        if result ==0.0:
            result = 'Mungbean'
        elif result ==1.0:
            result = 'Muskmelon 🍈🍈'
        elif result ==2.0:
            result = 'Watermelon 🍉🍉'
        elif result ==3.0:
            result = 'Lentil'
        elif result ==4.0:
            result = 'Jute'
        elif result ==5.0:
            result = 'Grapes 🍇🍇'
        elif result ==6.0:
            result = 'Mothbeans'
        elif result ==7.0:
            result = 'Banana 🍌🍌'
        elif result ==8.0:
            result = 'Rice 🌾🌾'
        elif result ==9.0:
            result = 'Kidneybeans 🫘🫘'
        elif result ==10.0:
            result = 'Orange 🍊🍊'
        elif result ==11.0:
            result = 'Coffee ☕☕'
        elif result ==12.0:
            result = 'Pigeonpeas'
        elif result ==13.0:
            result = 'Cotton'
        elif result ==14.0:
            result = 'Pomegranate'
        elif result ==15.0:
            result = 'Coconut  🌴🌴'
        elif result ==16.0:
            result = 'Mango  🥭🥭'
        elif result ==17.0:
            result = 'Maize  🌽🌽'
        elif result ==18.0:
            result = 'Blackgram'
        elif result ==19.0:
            result = 'Chickpea'
        elif result ==20.0:
            result = 'Apple  🍎🍎 ' 

        else:
            result = 'Papaya'


        return render_template('home.html', results=result)
    

if __name__ =="__main__":
    app.run(host='0.0.0.0')