import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from data_loader import load_datasets
from config import DATA_DIR, MODEL_PATH

def evaluate():

    _, _, test_ds = load_datasets(DATA_DIR)
    model = tf.keras.models.load_model(MODEL_PATH)

    y_true = []
    y_pred = []

    for images, labels in test_ds:
        preds = model.predict(images)
        y_true.extend(labels.numpy())
        y_pred.extend((preds > 0.5).astype(int).flatten())

    print("Classification Report:")
    print(classification_report(y_true, y_pred))

    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))


if __name__ == "__main__":
    evaluate()