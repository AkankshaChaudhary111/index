from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the model and the scaler
model = joblib.load('model/fraud_detection_model.pkl')
scaler = joblib.load('model/scaler.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the form data
        form_data = request.form.to_dict() 
        features = [float(form_data[key]) for key in form_data]
        
        # Preprocess the input
        features_scaled = scaler.transform([features])
        
        # Predict using the model
        prediction = model.predict(features_scaled)[0]
        
        # Render the result page
        return render_template('result.html', prediction=prediction)
    except Exception as e:
        return str(e)

if __name__ == '__main__':
    app.run(debug=True)
    