from trainer import train_full_sinkhorn


if __name__ == "__main__":
    # Train sinkhorn
    train_full_sinkhorn({
        "lr": 10e-6,
        "save_dir": "result",
        "save_name": "full_sinkhorn.npy",
        "output_model_prefix": "weights/model_full_sinkhorn.pth",
        "epochs": 20,
        "hidden_size": 256,
    })
