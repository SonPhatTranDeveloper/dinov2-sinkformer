from trainer import train_last_layer_sinkhorn


if __name__ == "__main__":
    # Train sinkhorn
    train_last_layer_sinkhorn({
        "lr": 10e-6,
        "save_dir": "result",
        "save_name": "result_last_layer_sinkhorn_3_iter.npy",
        "output_model_prefix": "weights/model_last_layer_sinkhorn_iter_3.pth",
        "epochs": 20,
        "hidden_size": 256,
    })
