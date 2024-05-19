from flask import Flask, render_template, request
import pickle as pk
import numpy as np

app = Flask(__name__)
model = pk.load(open('Price_Predictor.pkl', 'rb'))
scaler = pk.load(open('Scaler.pkl', 'rb'))

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/predict/', methods=['POST'])
def predict():
    form_data = request.form
    brand = form_data['brand']
    battery = float(form_data['battery']) / 1000
    camera = float(form_data['camera']) / 10

    brand_dict = {'Samsung': [1, 0, 0, 0, 0],
                  'Apple': [0, 1, 0, 0, 0],
                  'Xiaomi': [0, 0, 1, 0, 0],
                  'Oppo': [0, 0, 0, 1, 0],
                  'Vivo': [0, 0, 0, 0, 1]}
    
    brand_one_hot = brand_dict[brand]
    
    test_set = np.array([[battery, camera] + brand_one_hot])
    test_set = scaler.transform(test_set)
    
    prediction = model.predict(test_set)
    
    return render_template('predict.html', prediction_text=f"Predicted price of the phone is Rs.{prediction[0]:.2f}")

if __name__ == '__main__':
    app.run(debug=True)
