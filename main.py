from trainer import train_baseline


if __name__ == "__main__":
    # Train sinkhorn
    train_baseline({
        "data": "data/imagenette",
        "lr": 10e-6,
        "save_dir": "result/imagenette",
        "save_name": "result_softmax.npy",
        "output_model_prefix": "weights/model_softmax.pth",
        "epochs": 20,
        "hidden_size": 256,
    })
