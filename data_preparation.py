# data_preparation.py
import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

def preprocess_data(data):
    # Encode categorical features
    data = pd.get_dummies(data, columns=['sex', 'smoker', 'region'], drop_first=True)
    X = data.drop('charges', axis=1)
    y = data['charges']
    return train_test_split(X, y, test_size=0.2, random_state=42)

if __name__ == "__main__":
    file_path = 'F:/NIBM/HD/ML/CW4/Medical_Insuarance/insurance.csv'  # Corrected file path
    data = load_data(file_path)
    X_train, X_test, y_train, y_test = preprocess_data(data)
    print("Data loaded and preprocessed.")
