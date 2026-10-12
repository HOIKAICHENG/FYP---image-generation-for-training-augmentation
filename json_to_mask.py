import json
import cv2
import numpy as np
import os
from glob import glob

json_dir = "data/animal_dataset/images/train"
output_mask_dir = "data/animal_dataset/annotations/train"
os.makedirs(output_mask_dir, exist_ok=True)
class_mapping = {"animal_body": 1, "animal_face": 2}

json_files = glob(os.path.join(json_dir, "*.json"))
for json_file in json_files:
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    img_shape = (data["imageHeight"], data["imageWidth"])
    mask = np.zeros(img_shape, dtype=np.uint8)
    shapes = data["shapes"]
    shapes.sort(key=lambda x: 1 if x["label"] == "animal_body" else 2)
    
    for shape in shapes:
        label = shape["label"]
        if label in class_mapping:
            points = np.array(shape["points"], dtype=np.int32)
            cv2.fillPoly(mask, [points], class_mapping[label])
            
    base_name = os.path.basename(json_file).replace(".json", ".png")
    cv2.imwrite(os.path.join(output_mask_dir, base_name), mask)
print(f"轉換完成，共處理 {len(json_files)} 張 Mask。")
