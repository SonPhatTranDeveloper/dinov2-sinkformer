from trainer import train_last_layer_sinkhorn


if __name__ == "__main__":
    # Train sinkhorn
    train_last_layer_sinkhorn({
        "lr": 10e-6,
        "save_dir": "result",
        "save_name": "full_sinkhorn.npy",
        "output_model_prefix": "weights/model_full_sinkhorn.pth",
        "epochs": 20,
        "hidden_size": 256,
    })
