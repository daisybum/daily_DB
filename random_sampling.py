import os
from glob import glob
import shutil
import random
from tqdm import tqdm

from ultralytics import YOLO


def init_model_and_paths(model_path, base_path):
    model = YOLO(model_path)
    object_path = os.path.join(base_path, "1.노면 분류", "object")
    dry_path = os.path.join(base_path, "1.노면 분류", "dry")

    return model, object_path, dry_path


def move_files_to_dst(src_path, dst_folder):
    base_name = os.path.basename(src_path)
    json_name = base_name.replace(".jpg", ".json")
    try:
        shutil.move(src_path.replace(".jpg", ".json"), os.path.join(dst_folder, json_name))
        shutil.move(src_path, os.path.join(dst_folder, base_name))
    except FileNotFoundError:
        pass


if __name__ == "__main__":
    model_path = os.path.join(os.getcwd(), "yolo_model", "yolov8n.pt")
    data_path = "C:\\Users\\lww19\\temporal\\20230919"
    model, object_path, dry_path = init_model_and_paths(model_path, data_path)

    path_list = glob(os.path.join(data_path, "*.jpg"))
    path_hundreds = random.sample(path_list, 100)

    for path in tqdm(path_hundreds):
        results = model.predict(source=path, save=False, save_txt=False)

        if len(results[0].boxes) > 0 and max(results[0].boxes.conf) > 0.5:
            move_files_to_dst(path, object_path)
        else:
            move_files_to_dst(path, dry_path)
