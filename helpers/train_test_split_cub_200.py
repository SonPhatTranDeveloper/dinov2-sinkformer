import os
from pathlib import Path
import shutil

if __name__ == "__main__":
    # Get the train test split file
    train_test = {}

    # Get the file
    with open('../rawdata/cub200/train_test_split.txt', 'r') as f:
        for line in f.readlines():
            # Get the image number and image test or train
            image_number, image_train = line.split(" ")
            image_number = image_number.strip()
            image_train = image_train.strip()

            # Place in train test
            train_test[image_number] = image_train

    # Read image id
    image_to_id = {}
    with open('../rawdata/cub200/images.txt', 'r') as f:
        for line in f.readlines():
            # Get the image number and image test or train
            image_id, image_name = line.split(" ")
            image_id = image_id.strip()
            image_name = image_name.strip()

            # Place in train test
            image_to_id[image_name] = image_id

    # Get all the image
    train = []
    validation = []
    folder_names = []

    # Read image and place them in the cub200 data folder
    for folder_name in os.listdir('../rawdata/cub200/data'):
        if os.path.isfile(os.path.join('../rawdata/cub200/data', folder_name)):
            continue

        folder_names.append(folder_name)

        for file_name in os.listdir(os.path.join('../rawdata/cub200/data', folder_name)):
            # Get the image name
            image_name = os.path.join(folder_name, file_name)

            # Get the full image path
            image_path = os.path.join('../rawdata/cub200/data', folder_name, file_name)
            image_train = train_test[image_to_id[image_name]]

            # Place in train and validation
            if image_train == "1":
                train.append(image_path)
            else:
                validation.append(image_path)

    # Make directory
    for name in folder_names:
        Path(os.path.join('../data/cub200/train', name)).mkdir(exist_ok=True)
        Path(os.path.join('../data/cub200/val', name)).mkdir(exist_ok=True)

    # Copy
    for file in train:
        folder_name = os.path.join('../data/cub200/train', file.split("/")[-2])
        shutil.copy2(file, folder_name)

    for file in validation:
        folder_name = os.path.join('../data/cub200/val', file.split("/")[-2])
        shutil.copy2(file, folder_name)
