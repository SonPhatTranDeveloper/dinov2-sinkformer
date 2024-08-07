import os
from pathlib import Path

import torch
from torchvision import datasets, transforms

from helpers.dataset import ResizeAndPad

from trainer.trainer import Trainer

from models.last_layer_sinkhorn import DINOClassificationModel

import numpy as np


def train(args):
    # Get the training and validation dataset
    train_dataset_dir = Path(os.path.join(args["data"], "train"))
    valid_dataset_dir = Path(os.path.join(args["data"], "val"))

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

    # Create result
    model = DINOClassificationModel(
        hidden_size=args["hidden_size"],
        num_classes=len(CLASSES),
    )

    # Display info
    print('Train dataset of size %d' % len(DATASETS["train"]))
    print('Validation dataset of size %d' % len(DATASETS["valid"]))
    print()

    # Create trainer
    trainer = Trainer(
        model,
        DEVICE,
        DATALOADERS["train"],
        DATALOADERS["valid"],
        args
    )

    # Get validation accuracy
    trainer.validate(0)
