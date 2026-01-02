from typing import Tuple
from PIL import Image, ImageOps


# Resize the Images and pad to square shape
def letterbox_pil(
        img: Image.Image, 
        new_shape: int, 
        color=(114, 114, 114)
    ) -> Tuple[Image.Image, float, float, int, int]:
    w, h = img.size
    r = min(new_shape / w, new_shape / h)

    new_unpad = (max(int(round(w * r)), 1), max(int(round(h * r)), 1))

    real_scale_w = new_unpad[0] / w
    real_scale_h = new_unpad[1] / h

    img = img.resize(new_unpad, Image.BILINEAR)
    
    pad_w = new_shape - new_unpad[0] # width
    pad_h = new_shape - new_unpad[1] # height
    pad_left = pad_w // 2
    pad_top = pad_h // 2

    img = ImageOps.expand(
        img, 
        border=(pad_left, pad_top, pad_w - pad_left, pad_h - pad_top),
        fill=color,
    )
    
    return img, real_scale_w, real_scale_h, pad_left, pad_top

# Clip a bounding box to lie within [0, w-1] x [0, h-1]
def clip_box(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    w: int,
    h: int,
) -> Tuple[float, float, float, float]:
    
    x1 = max(0.0, min(w - 1.0, x1))
    y1 = max(0.0, min(h - 1.0, y1))
    x2 = max(0.0, min(w - 1.0, x2))
    y2 = max(0.0, min(h - 1.0, y2))
    return x1, y1, x2, y2

# Convert an absolute pixel bounding box (x1,y1,x2,y2) 
# into YOLO-normalized format (cx, cy, bw, bh) with values in [0,1]
def yolo_norm_from_abs(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    w: int,
    h: int,
) -> Tuple[float, float, float, float]:
    bw = max(0.0, x2 - x1)
    bh = max(0.0, y2 - y1)
    if bw <= 0 or bh <= 0:
        return 0.0, 0.0, 0.0, 0.0
    cx = x1 + bw / 2.0
    cy = y1 + bh / 2.0
    return cx / w, cy / h, bw / w, bh / h