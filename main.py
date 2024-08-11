from trainer import train_last_layer_sinkhorn


if __name__ == "__main__":
    # Train sinkhorn with different weights
    train_last_layer_sinkhorn({
        "data": "data/imagewang",
        "lr": 10e-6,
        "save_dir": "result/imagewang",
        "save_name": "result_last_layer_3_iter.npy",
        "output_model_prefix": "weights/imagenette/model_last_layer_3_iter.pth",
        "epochs": 20,
        "hidden_size": 256,
    })
