from flask import Flask, request, render_template
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the trained model
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Retrieve form values
        age = request.form['age']
        gender = request.form['gender']
        bmi = request.form['bmi']
        children = request.form['children']
        smoker = request.form['smoker']
        region = request.form['region']

        print(f"Received inputs - Age: {age}, Gender: {gender}, BMI: {bmi}, Children: {children}, Smoker: {smoker}, Region: {region}")

        # Validate numeric fields
        if not age.isdigit() or not bmi.replace('.', '', 1).isdigit() or not children.isdigit():
            return render_template('index.html', error_message="Please enter valid numeric values for age, BMI, and children.")

        # Convert to appropriate data types
        age = int(age)
        bmi = float(bmi)
        children = int(children)

        print(f"Validated and converted inputs - Age: {age}, BMI: {bmi}, Children: {children}")

        # Validate categorical fields
        if gender not in ['male', 'female']:
            return render_template('index.html', error_message="Please select a valid gender.")
        if smoker not in ['yes', 'no']:
            return render_template('index.html', error_message="Please select a valid smoking status.")
        if region not in ['northeast', 'northwest', 'southeast', 'southwest']:
            return render_template('index.html', error_message="Please select a valid region.")

        # One-hot encode categorical features
        data = {
            'age': [age],
            'bmi': [bmi],
            'children': [children],
            'gender_male': [1 if gender == 'male' else 0],
            'smoker_yes': [1 if smoker == 'yes' else 0],
            'region_northeast': [1 if region == 'northeast' else 0],
            'region_northwest': [1 if region == 'northwest' else 0],
            'region_southeast': [1 if region == 'southeast' else 0],
            'region_southwest': [1 if region == 'southwest' else 0]
        }

        # Create DataFrame for prediction
        df = pd.DataFrame(data)
        
        print(f"DataFrame for prediction: \n{df}")

        # Predict using the model
        prediction = model.predict(df)
        output = round(prediction[0], 2)

        print(f"Prediction output: {output}")

        return render_template('index.html', prediction_text='Estimated Insurance Charges: ${}'.format(output))

    except ValueError:
        return render_template('index.html', error_message="Please enter valid numeric values for age, BMI, and children.")
    except Exception as e:
        print(f"Exception: {e}")
        return render_template('index.html', error_message=str(e))

if __name__ == "__main__":
    app.run(debug=True)
