import torch
import torchvision
import torchvision.transforms as transforms
import os
from PIL import Image
from tqdm import tqdm

print("Setting up CIFAR-10 directories...")

# --- Configuration ---
# Match the paths expected by train.py (relative to the script's location)
base_data_dir = '../data/cifar10'
temp_download_dir = os.path.join(base_data_dir, 'temp_download')
target_img_dir = os.path.join(base_data_dir, 'imgs')
target_train_dir = os.path.join(target_img_dir, 'train')
target_test_dir = os.path.join(target_img_dir, 'test')

# We don't need complex transforms here, just converting to PIL Image
# Torchvision dataset returns PIL Images by default if transform is None
transform = None

# --- Download Raw Data using Torchvision ---
print(f"Downloading CIFAR-10 to temporary location: {temp_download_dir}")
try:
    trainset = torchvision.datasets.CIFAR10(root=temp_download_dir, train=True,
                                            download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root=temp_download_dir, train=False,
                                           download=True, transform=transform)
    print("Download complete.")
except Exception as e:
    print(f"Error during download: {e}")
    print("Please check your internet connection and permissions.")
    exit(1)

# Get class names - these will be the folder names
classes = trainset.classes
print(f"CIFAR-10 classes: {classes}")

# --- Function to Save Images to Target Structure ---
def save_images_to_folders(dataset, target_dir, class_names):
    print(f"Processing and saving images to: {target_dir}")
    # Keep track of counts per class for unique filenames
    class_counts = {classname: 0 for classname in class_names}

    for img, label_index in tqdm(dataset, desc=f"Saving to {os.path.basename(target_dir)}"):
        class_name = class_names[label_index]
        class_dir = os.path.join(target_dir, class_name)

        # Create class directory if it doesn't exist
        os.makedirs(class_dir, exist_ok=True)

        # Generate a unique filename within the class folder
        image_index = class_counts[class_name]
        filename = f"{image_index}.png" # Using png format
        filepath = os.path.join(class_dir, filename)

        # Save the PIL Image
        try:
            img.save(filepath, format='PNG')
            class_counts[class_name] += 1
        except Exception as e:
            print(f"\nError saving image {filepath}: {e}")

    print(f"Finished saving images to {target_dir}.")
    print("Image counts per class:")
    for classname, count in class_counts.items():
        print(f"- {classname}: {count}")

# --- Process and Save Train Set ---
save_images_to_folders(trainset, target_train_dir, classes)

# --- Process and Save Test Set ---
save_images_to_folders(testset, target_test_dir, classes)

print("\nDataset setup complete!")
print(f"Train images are in: {target_train_dir}")
print(f"Test images are in: {target_test_dir}")
print(f"You can now optionally remove the temporary download directory: {temp_download_dir}")