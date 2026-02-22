import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from config import IMG_SIZE, BATCH_SIZE

def load_datasets(data_dir):

    train_ds = image_dataset_from_directory(
        f"{data_dir}/train",
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )

    val_ds = image_dataset_from_directory(
        f"{data_dir}/val",
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )

    test_ds = image_dataset_from_directory(
        f"{data_dir}/test",
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = train_ds.prefetch(AUTOTUNE)
    val_ds = val_ds.prefetch(AUTOTUNE)
    test_ds = test_ds.prefetch(AUTOTUNE)

    return train_ds, val_ds, test_ds