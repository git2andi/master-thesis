# preprocess/helper/coco.py

from typing import Dict, Any, List, Optional

# Initialize a COCO dictionary with the given metadata 
# and empty 'images' / 'annotations' lists.
def coco_init(
    info: Dict[str, Any],
    licenses: List[Dict[str, Any]],
    categories: List[Dict[str, Any]],
) -> Dict[str, Any]:
    return {
        "info": info,
        "licenses": licenses,
        "categories": categories,
        "images": [],
        "annotations": [],
    }

# Append one image entry to a COCO dict.
def add_coco_image(
    coco_dict: Dict[str, Any],
    image_id: int,
    file_name: str,
    width: int,
    height: int,
    license_id: int = 1,
) -> None:
    coco_dict["images"].append({
        "id": image_id,
        "file_name": file_name,
        "height": height,
        "width": width,
        "license": license_id,
    })

# Append one rectangular annotation to a COCO dict.
def add_coco_annotation(
    coco_dict: Dict[str, Any],
    ann_id: int,
    image_id: int,
    x: float,
    y: float,
    w: float,
    h: float,
    category_id: int,
    extra_fields: Optional[Dict[str, Any]] = None,
) -> None:
    area = w * h
    l, t = x, y
    r, b = x + w, y + h
    seg = [[l, t, r, t, r, b, l, b]]

    ann = {
        "id": ann_id,
        "image_id": image_id,
        "category_id": category_id,
        "bbox": [x, y, w, h],
        "area": area,
        "iscrowd": 0,
        "segmentation": seg,
    }

    if extra_fields:
        ann.update(extra_fields)

    coco_dict["annotations"].append(ann)
