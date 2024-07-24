# model_evaluation.py
import joblib
from sklearn.metrics import mean_squared_error
from data_preparation import load_data, preprocess_data

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    return mse

if __name__ == "__main__":
    file_path = 'F:/NIBM/HD/ML/CW4/Medical_Insuarance/insurance.csv'  # Corrected file path
    data = load_data(file_path)
    X_train, X_test, y_train, y_test = preprocess_data(data)
    
    model = joblib.load('F:/NIBM/HD/ML/CW4/Medical_Insuarance/insurance_model.pkl')
    mse = evaluate_model(model, X_test, y_test)
    print(f"Mean Squared Error: {mse}")
