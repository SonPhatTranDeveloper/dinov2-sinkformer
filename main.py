from trainer import train_full_sinkhorn


if __name__ == "__main__":
    # Train sinkhorn
    train_full_sinkhorn({
        "data": "data/imagenette",
        "lr": 10e-6,
        "save_dir": "result/imagenette",
        "save_name": "result_full_sinkhorn_7_iter.npy",
        "output_model_prefix": "weights/imagenette/model_full_sinkhorn_7_iter.pth",
        "epochs": 20,
        "hidden_size": 256,
    })
