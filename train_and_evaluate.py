import matplotlib.pyplot as plt
from keras.src.metrics import F1Score
from tensorflow.keras.optimizers import Adam
from data_preprocessing import load_and_preprocess_data
from build_model import build_model
from tensorflow.keras.callbacks import Callback
from sklearn.metrics import f1_score
import numpy as np


def train_and_evaluate(X_train, y_train_cat, X_val, y_val_cat):
    model = build_model(X_train.shape[1], y_train_cat.shape[1])
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

    # Pass both training and validation data to the callback
    f1_callback = F1ScoreCallback(train_data=(X_train, y_train_cat), val_data=(X_val, y_val_cat))
    history = model.fit(X_train, y_train_cat, validation_data=(X_val, y_val_cat), epochs=50, batch_size=32, callbacks=[f1_callback])

    # Plotting all metrics in one figure
    plt.figure(figsize=(15, 5))

    # Accuracy
    plt.subplot(1, 3, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Loss
    plt.subplot(1, 3, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # F1 Score
    plt.subplot(1, 3, 3)
    plt.plot(range(1, len(f1_callback.train_f1s) + 1), f1_callback.train_f1s, label='Train F1 Score')
    plt.plot(range(1, len(f1_callback.val_f1s) + 1), f1_callback.val_f1s, label='Validation F1 Score')
    plt.title('F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.legend()

    plt.tight_layout()
    plt.show()


class F1ScoreCallback(Callback):
    def __init__(self, train_data=(), val_data=()):
        super(F1ScoreCallback, self).__init__()
        self.train_data = train_data
        self.val_data = val_data
        self.train_f1s = []
        self.val_f1s = []

    def on_epoch_end(self, epoch, logs=None):
        # Calculate F1 for training data
        train_predictions = np.argmax(self.model.predict(self.train_data[0]), axis=1)
        train_true = np.argmax(self.train_data[1], axis=1)
        train_f1 = f1_score(train_true, train_predictions, average='macro')
        self.train_f1s.append(train_f1)

        # Calculate F1 for validation data
        val_predictions = np.argmax(self.model.predict(self.val_data[0]), axis=1)
        val_true = np.argmax(self.val_data[1], axis=1)
        val_f1 = f1_score(val_true, val_predictions, average='macro')
        self.val_f1s.append(val_f1)

        print(f'Epoch {epoch+1} - train_f1: {train_f1:.4f} - val_f1: {val_f1:.4f}')


if __name__ == "__main__":
    filepath = 'CCD.xls'
    X_train, X_val, y_train_cat, y_val_cat = load_and_preprocess_data(filepath)
    train_and_evaluate(X_train, y_train_cat, X_val, y_val_cat)
