import os
import shutil
import re
from glob import glob
import random

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFile
from torch import nn
from torchvision import transforms
from torchvision.models.mobilenetv3 import mobilenet_v3_small
from tqdm import tqdm

from ultralytics import YOLO

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


def copy_files_to_dst(src_path, dst_folder):
    base_name = os.path.basename(src_path)
    json_name = base_name.replace(".jpg", ".json")
    try:
        shutil.copyfile(src_path.replace(".jpg", ".json"), os.path.join(dst_folder, json_name))
        shutil.copyfile(src_path, os.path.join(dst_folder, base_name))
    except FileNotFoundError:
        print(f"File not found: {src_path}")
        pass


if __name__ == '__main__':
    # Load the trained model
    model_path = 'model/best.pth'  # Update this path
    model = mobilenet_v3_small(pretrained=False)
    num_classes = 9
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    model.load_state_dict(torch.load(model_path))

    model_yolo = YOLO(model="model\\yolov8x.pt")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Class names (update as per your dataset)
    class_names = ['dry', 'dry-humid', 'humid', 'humid-wet', 'no use', 'object', 'slush', 'wet', 'wet-slush']

    # Transform
    transform = test_transform()

    # Directory containing images
    image_directory = 'C:/Users\\hirva'  # Update this path
    name_pattern = ('[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]_[0-9][0-9][0-9]_'
                    '[0-9][0-9][A-Z0-9][A-Z0-9][A-Z][0-9][A-Z0-9][0-9][A-Z0-9][A-Z0-9]_'
                    '[0-9]_[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]_[0-9][0-9][0-9][0-9][0-9][0-9].jpg')
    image_paths = glob(os.path.join(image_directory, '**', name_pattern), recursive=True)
    random_select = random.sample(image_paths, 500)

    # Evaluate images
    results = []
    for image_path in tqdm(random_select):
        yolo_results = model_yolo.predict(image_path, save=False, imgsz=640, conf=0.3, classes=[0, 2])[0]

        if len(yolo_results.boxes):
            img = yolo_results.orig_img
            for i in range(len(yolo_results.boxes)):
                box = yolo_results.boxes[i]
                xyxy = box.xyxy.cpu().numpy()[0]
                start_x, start_y, end_x, end_y = np.rint(xyxy).astype('int')
                down_sample_img = cv2.resize(img[start_y:end_y, start_x:end_x, :], dsize=(0, 0),
                                             interpolation=cv2.INTER_NEAREST, fx=0.15, fy=0.15)
                up_sample_img = cv2.resize(down_sample_img, interpolation=cv2.INTER_NEAREST,
                                           dsize=(end_x - start_x, end_y - start_y))
                img[start_y:end_y, start_x:end_x, :] = up_sample_img
            cv2.imwrite(image_path, img)

        predicted_class = evaluate_single_image(model, image_path, transform, device, class_names)
        results.append({'Image Path': image_path, 'Predicted Class': predicted_class})

        date_str = '20240131'
        dst_dir = os.path.join(*re.split(r"\\", image_path)[:-1], date_str,"1.노면 분류")
        dst_path = os.path.join(dst_dir, predicted_class)
        copy_files_to_dst(image_path, dst_path)

        dst_path = os.path.join(*re.split(r"\\", image_path)[:-1], date_str, "2.기상 분류", "clear")
        copy_files_to_dst(image_path, dst_path)



