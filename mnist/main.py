from training import train_model
from testing import test_model
import os

if __name__ == '__main__':
    train_model(epochs=10)
    model_path = os.listdir('models')[0]
    print(model_path)
    test_model(f"models/{model_path}")