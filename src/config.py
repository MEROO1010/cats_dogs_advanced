import os

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")

# Image & Training Config
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-4
FINE_TUNE_LR = 1e-5

# Model
MODEL_NAME = "EfficientNetB0"
MODEL_PATH = os.path.join(MODEL_DIR, "best_model.h5")