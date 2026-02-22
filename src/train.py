import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from data_loader import load_datasets
from model_builder import build_model
from config import DATA_DIR, EPOCHS, MODEL_PATH

def train():

    train_ds, val_ds, _ = load_datasets(DATA_DIR)

    model = build_model()

    callbacks = [
        EarlyStopping(patience=5, restore_best_weights=True),
        ModelCheckpoint(MODEL_PATH, save_best_only=True)
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks
    )

    return history


if __name__ == "__main__":
    train()