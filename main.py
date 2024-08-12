from trainer import train_full_sinkhorn
import torch
import random
import numpy as np

# Set random seeds
torch.manual_seed(1120)
random.seed(1120)
np.random.seed(1120)


if __name__ == "__main__":
    # Train sinkhorn with different weights
    train_full_sinkhorn({
        "data": "data/cub200",
        "lr": 10e-6,
        "save_dir": "result/cub200",
        "save_name": "result_full_sinkhorn_7_iter.npy",
        "output_model_prefix": "weights/cub200/model_full_sinkhorn_7_iter.pth",
        "epochs": 20,
        "hidden_size": 256,
    })
