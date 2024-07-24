from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the model and label encoders
model = joblib.load('insurance_model.pkl')
label_encoders = joblib.load('label_encoders.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        age = data['age']
        sex = label_encoders['sex'].transform([data['sex']])[0]
        bmi = data['bmi']
        children = data['children']
        smoker = label_encoders['smoker'].transform([data['smoker']])[0]
        region = label_encoders['region'].transform([data['region']])[0]

        features = np.array([[age, sex, bmi, children, smoker, region]])
        prediction = model.predict(features)

        return jsonify({'charges': prediction[0]})

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': 'An error occurred. Please try again.'}), 500

if __name__ == '__main__':
    app.run(debug=True)
