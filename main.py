from trainer import train_baseline


if __name__ == "__main__":
    # Train sinkhorn with different weights
    train_baseline({
        "data": "data/cub200",
        "lr": 10e-6,
        "save_dir": "result/cub200",
        "save_name": "result_baseline.npy",
        "output_model_prefix": "weights/cub200/model_baseline.pth",
        "epochs": 20,
        "hidden_size": 256,
    })
