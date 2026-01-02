import json
from pathlib import Path
from typing import Any, Dict, List


def load_ultralytics_predictions(pred_path: Path) -> List[Dict[str, Any]]:
    if not pred_path.exists():
        raise FileNotFoundError(f"File not found: {pred_path}")

    with pred_path.open("r", encoding="utf-8") as f:
        data: Any = json.load(f)

    if not isinstance(data, list):
        raise ValueError(
            "Ultralytics predictions JSON is expected to be a list of detections. "
        )

    return data
