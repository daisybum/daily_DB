import os
import shutil
import re
from glob import glob
import random

import pandas as pd
import torch
from PIL import Image, ImageFile
from torchvision import transforms
from torchvision.models.mobilenetv3 import mobilenet_v3_small
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True

def test_transform():
    """Transforms for the test dataset."""
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def evaluate_single_image(model, image_path, transform, device, class_names):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        model.eval()
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        predicted_class = class_names[predicted.cpu().item()]

    return predicted_class


def move_files_to_dst(src_path, dst_folder):
    base_name = os.path.basename(src_path)
    json_name = base_name.replace(".jpg", ".json")
    try:
        shutil.move(src_path.replace(".jpg", ".json"), os.path.join(dst_folder, json_name))
        shutil.move(src_path, os.path.join(dst_folder, base_name))
    except FileNotFoundError:
        pass


if __name__ == '__main__':
    # Load the trained model
    model_path = 'model/mobilenet_epoch_50.pth'  # Update this path
    model = mobilenet_v3_small(pretrained=False)
    model.load_state_dict(torch.load(model_path))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Class names (update as per your dataset)
    class_names = ['dry', 'dry-humid', 'humid', 'humid-wet', 'no use', 'object', 'slush', 'wet', 'wet-slush']

    # Transform
    transform = test_transform()

    # Directory containing images
    image_directory = 'C:/Users\\hirva'  # Update this path
    image_paths = glob(os.path.join(image_directory, '*/20231109/*.jpg'))

    # Evaluate images
    results = []
    for image_path in tqdm(image_paths):
        predicted_class = evaluate_single_image(model, image_path, transform, device, class_names)
        results.append({'Image Path': image_path, 'Predicted Class': predicted_class})

        file_name = os.path.basename(image_path)
        dst_dir = os.path.join(*re.split(r"\\", image_path)[:-1], "1.노면 분류")
        dst_path = os.path.join(dst_dir, predicted_class)
        move_files_to_dst(image_path, dst_path)

    image_paths = glob(os.path.join(image_directory, '*/*/*/*/*.jpg'))
    path_4hundreds = random.sample(image_paths, 400)

    for path in tqdm(image_paths):
        if path not in path_4hundreds:
            json_path = path.replace('.jpg', '.json')
            os.remove(path)
            os.remove(json_path)

