import numpy as np
from flask import Flask, request, jsonify, render_template
import joblib

app = Flask(__name__)

model = joblib.load("randomForestRegressor.pkl")

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = [float(x) for x in request.form.values()]
    final_features = np.array([features])
    
    prediction = model.predict(final_features)[0]
    
    return render_template(
        'home.html',
        prediction_text=f"AQI for Jaipur: {prediction:.2f}"
    )

@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])[0]
    return jsonify({'aqi': prediction})

if __name__ == "__main__":
    app.run(debug=True)