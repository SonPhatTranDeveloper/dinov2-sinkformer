from pathlib import Path
import os

import torch
from torchvision import datasets, transforms

from helpers.dataset import ResizeAndPad

from trainer.trainer import Trainer

from models.weighted_fixed import DINOClassificationModel

import numpy as np


def train(args):
    # Get the training and validation dataset
    train_dataset_dir = Path("data/train")
    valid_dataset_dir = Path("data/val")

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

    # Start training
    val_loss_array = []
    train_loss_array = []
    val_accuracy_array = []
    train_accuracy_array = []

    # Model save directory
    stats_save_dir = args["save_dir"]
    stats_save_name = args["save_name"]
    stats_save_address = os.path.join(stats_save_dir, stats_save_name)

    # Train & Validate
    for epoch in range(1, args["epochs"] + 1):
        # Train the result for thi epoch
        epoch_loss, epoch_accuracy = trainer.train(epoch)

        # Validate the result
        epoch_val_loss, epoch_val_accuracy = trainer.validate(epoch)

        # Save the result
        trainer.save(args["output_model_prefix"], epoch)

        # Save the training and validation accuracy
        val_accuracy_array.append(epoch_val_accuracy)
        train_accuracy_array.append(epoch_accuracy)

        # Save the validation and training loss
        val_loss_array.append(epoch_val_loss)
        train_loss_array.append(epoch_loss)

        # Save the training and validation result
        losses = np.asarray([train_loss_array, val_loss_array, train_accuracy_array, val_accuracy_array])
        np.save(stats_save_address, losses)