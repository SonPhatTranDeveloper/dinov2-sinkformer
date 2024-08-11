from trainer import train_last_layer_sinkhorn


if __name__ == "__main__":
    # Train sinkhorn with different weights
    train_last_layer_sinkhorn({
        "data": "data/cub200",
        "lr": 10e-6,
        "save_dir": "result/cub200",
        "save_name": "result_last_layer_sinkhorn_3_iter.npy",
        "output_model_prefix": "weights/cub200/model_last_layer_sinkhorn_3_iter.pth",
        "epochs": 20,
        "hidden_size": 256,
    })
