from trainer import train_baseline
import torch
import random
import numpy as np

# Set random seeds
torch.manual_seed(1120)
random.seed(1120)
np.random.seed(1120)


if __name__ == "__main__":
    # Train sinkhorn with different weights
    train_baseline({
        "data": "data/imagewoof",
        "lr": 10e-6,
        "save_dir": "result/imagewoof",
        "save_name": "baseline.npy",
        "output_model_prefix": "weights/imagewoof/baseline.pth",
        "epochs": 20,
        "hidden_size": 256,
        "k": 1
    })
