from trainer import train_last_layer_weighted_fixed


if __name__ == "__main__":
    # Train sinkhorn
    train_last_layer_weighted_fixed({
        "data": "data/imagewoof",
        "lr": 10e-6,
        "save_dir": "result/imagewoof",
        "save_name": "result_last_layer_weighted_0_6_5_iter.npy",
        "output_model_prefix": "weights/imagenette/model_last_layer_weighted_0_6_5_iter.pth",
        "epochs": 20,
        "hidden_size": 256,
    })
