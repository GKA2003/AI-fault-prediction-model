import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.utils import to_categorical


def load_and_preprocess_data(filepath):
    df = pd.read_excel(filepath, header=0, skiprows=[1])  # Skip the first row of data

    X = df.drop('Y', axis=1).values
    y = df['Y'].values

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    y_train_cat = to_categorical(y_train)
    y_val_cat = to_categorical(y_val)

    return X_train, X_val, y_train_cat, y_val_cat
