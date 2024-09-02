import os
from pathlib import Path

import torch
from torchvision import datasets, transforms

from helpers.dataset import ResizeAndPad

from trainer.trainer import Trainer

from models.baseline_softmax import DINOClassificationModel

from models.attention import SinkhornAttention
from dinov2.layers import NestedTensorBlock as Block
from functools import partial
from dinov2.vision_transformer import DinoVisionTransformer

from copy import deepcopy


def vit_small_sinkhorn(patch_size=16, num_register_tokens=0, **kwargs):
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        block_fn=partial(Block, attn_class=SinkhornAttention),
        num_register_tokens=num_register_tokens,
        **kwargs,
    )
    return model


def get_model(weight_path):
    # Load the model
    model = DINOClassificationModel(
        hidden_size=256,
        num_classes=len(CLASSES),
    )
    model.load_state_dict(
        torch.load(weight_path)["model_state_dict"]
    )

    # Copy the weight of the transformer
    vit_transformers = model.transformers

    # Copy the weights
    vit_sinkformers = vit_small_sinkhorn(patch_size=14,
                                         img_size=526,
                                         init_values=1.0,
                                         num_register_tokens=4,
                                         block_chunks=0)

    # Copy the weight
    vit_sinkformers.load_state_dict(vit_transformers.state_dict(), strict=False)

    # Load the weight of softmax attention layer to sinkhorn attention layer
    for block_sinkformer, block_transformer in zip(vit_sinkformers.blocks, vit_transformers.blocks):
        # Load all other blocks
        block_sinkformer.load_state_dict(block_transformer.state_dict(), strict=False)

        # Load the softmax weight
        attn_sinkformer = block_sinkformer.attn
        attn_transformer = block_transformer.attn

        attn_sinkformer.qkv.load_state_dict(
            attn_transformer.qkv.state_dict()
        )
        attn_sinkformer.attn_drop.load_state_dict(
            attn_transformer.attn_drop.state_dict()
        )
        attn_sinkformer.proj.load_state_dict(
            attn_transformer.proj.state_dict()
        )
        attn_sinkformer.proj_drop.load_state_dict(
            attn_transformer.proj_drop.state_dict()
        )

    # Copy back the last block of the sinkformer to the transformer
    vit_transformers.blocks[11] = vit_sinkformers.blocks[11]

    # Attach back to model
    model.transformers = vit_sinkformers

    # Create a deep copy of the model
    return deepcopy(model)


if __name__ == "__main__":
    # Set the arguments
    args = {
        "data": "data/imagewoof",
        "lr": 10e-6,
        "save_dir": "result/imagewoof",
        "save_name": "baseline.npy",
        "output_model_prefix": "weights/imagewoof/baseline.pth",
        "epochs": 20,
        "hidden_size": 256,
        "k": 5
    }

    # Get the training and validation dataset
    train_dataset_dir = Path(os.path.join(args["data"], "train"))
    valid_dataset_dir = Path(os.path.join(args["data"], "val"))
    weight_dir = Path("weights", "imagewoof", "baseline.pth")

    # Define image size
    IMAGE_SIZE = 256
    TARGET_SIZE = (IMAGE_SIZE, IMAGE_SIZE)

    # Define data transformation
    DATA_TRANSFORM = {
        "train": transforms.Compose(
            [
                ResizeAndPad(TARGET_SIZE, 14),
                transforms.RandomRotation(360),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
        "valid": transforms.Compose(
            [
                ResizeAndPad(TARGET_SIZE, 14),
                transforms.RandomRotation(360),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
    }

    # Define dataset and class
    DATASETS = {
        "train": datasets.ImageFolder(train_dataset_dir, DATA_TRANSFORM["train"]),
        "valid": datasets.ImageFolder(valid_dataset_dir, DATA_TRANSFORM["valid"])
    }

    DATALOADERS = {
        "train": torch.utils.data.DataLoader(DATASETS["train"], batch_size=8, shuffle=True),
        "valid": torch.utils.data.DataLoader(DATASETS["valid"], batch_size=8, shuffle=True)
    }

    CLASSES = DATASETS["train"].classes

    # Define the DEVICE for training the result
    DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Create empty model and load the weight
    model = get_model(weight_dir)

    # Create trainer
    trainer = Trainer(
        model,
        DEVICE,
        DATALOADERS["train"],
        DATALOADERS["valid"],
        args
    )

    # Display info
    print('Train dataset of size %d' % len(DATASETS["train"]))
    print('Validation dataset of size %d' % len(DATASETS["valid"]))
    print()

    # Evaluate
    trainer.validate(epoch=0, k=1)