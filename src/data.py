import os
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


def load_data(train_path='data/train.csv', test_path='data/test.csv'):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    X = train.drop('SalePrice', axis=1)
    y = train['SalePrice']

    return X, y, test


def preprocess_data(X, y, test):
    # Splitting the data
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Identifying column types
    numeric_cols = X_train.select_dtypes(include=['int64', 'float64']).columns
    non_numeric_cols = X_train.select_dtypes(exclude=['int64', 'float64']).columns

    # Imputing numeric
    imputer = KNNImputer()
    X_train[numeric_cols] = imputer.fit_transform(X_train[numeric_cols])
    X_val[numeric_cols] = imputer.transform(X_val[numeric_cols])
    test[numeric_cols] = imputer.transform(test[numeric_cols])

    # Imputing non-numeric with mode
    for col in non_numeric_cols:
        mode_train = X_train[col].mode(dropna=True)[0]
        X_train[col] = X_train[col].fillna(mode_train)
        X_val[col] = X_val[col].fillna(mode_train)
        test[col] = test[col].fillna(mode_train)

    # One-hot encoding
    ohe = OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=True)
    X_train_enc = ohe.fit_transform(X_train)
    X_val_enc = ohe.transform(X_val)
    test_enc = ohe.transform(test)

    return X_train_enc, X_val_enc, y_train, y_val, test_enc


X_raw, y_raw, test_raw = load_data()
X_train, X_val, y_train, y_val, test = preprocess_data(X_raw, y_raw, test_raw)
