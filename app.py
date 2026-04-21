from flask import Flask, render_template, request
import pickle
import numpy as np

application = Flask(__name__)   # AWS EB looks for `application`
app = application               # alias

# Load model
model = pickle.load(open("randomForestRegressor.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        features = [float(x) for x in request.form.values()]
        final_features = np.array(features).reshape(1, -1)
        prediction = model.predict(final_features)

        return render_template(
            "index.html",
            prediction_text=f"Predicted Value: {round(prediction[0], 2)}"
        )
    except Exception as e:
        return render_template(
            "index.html",
            prediction_text=f"Error: {str(e)}"
        )
if __name__ == "__main__":
    app.run(debug=True)
