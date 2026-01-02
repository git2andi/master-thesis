''''
run validation with code as suggested by ultralytics maintainers to prevent errors (only rtdetr is affected) 
create conda env and init ultralytics with: pip install git+https://github.com/ultralytics/ultralytics@rtdetr-transform (right now this is not yet in main)
> Fix only ensures correct padding etc when writing the predictions.json at the end. Training and Validation logic is not affected
'''

from ultralytics import RTDETR

model = RTDETR("/path/to/best/detr")
metrics = model.val(
    data = "path/to/data.yaml",
    split = "test",
    batch=1,
    device=1,
    max_det=100,
    iou=0.5,
    conf=0.001,
    imgsz=640, # Default is 640
    workers=16,
    save_json=True,
    rect=False,
    seed=42,
    project="path/to/output",
    name="outputName"
)

metrics.box.map
metrics.box.map50
metrics.box.map75
metrics.box.maps
metrics.to_csv()

print(metrics)