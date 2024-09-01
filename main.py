from trainer import train_last_layer_sinkhorn
import torch
import random
import numpy as np

# Set random seeds
torch.manual_seed(1120)
random.seed(1120)
np.random.seed(1120)


if __name__ == "__main__":
    # Train sinkhorn with different weights
    train_last_layer_sinkhorn({
        "data": "data/cub200",
        "lr": 10e-6,
        "save_dir": "result/cub200",
        "save_name": "last_layer_sinkhorn_5.npy",
        "output_model_prefix": "weights/cub200/last_layer_sinkhorn_5.pth",
        "epochs": 30,
        "hidden_size": 256,
        "k": 5
    })
