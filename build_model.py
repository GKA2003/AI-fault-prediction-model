from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input


def build_model(input_shape, num_classes):
    model = Sequential([
        Input(shape=(input_shape,)),  # Define input shape explicitly
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    return model
