import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

def standardize_data(X_train, X_test):
    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)
    X_test_std = scaler.transform(X_test)
    return X_train_std, X_test_std

def load_model(model_path):
    model = joblib.load(model_path)
    return model