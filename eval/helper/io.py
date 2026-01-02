from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict
import re

def load_json_any(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def make_output_dir(pred_path: Path) -> Path:
    """
    Store results in: <entry_script_dir>/results/<pred_parent_folder_name>/

    entry_script_dir is derived from sys.argv[0] (e.g., run_eval.py), not from helper/io.py.
    """
    entry_script = Path(sys.argv[0]).resolve()
    base_dir = entry_script.parent
    stem = pred_path.stem
    m = re.match(r"^(?:predictions[_-])(.+)$", stem)
    run_name = m.group(1) if m else stem

    out_dir = base_dir / "results" / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir

def write_json(out_dir: Path, payload: Dict[str, Any], filename: str) -> Path:
    out_path = out_dir / filename
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return out_path
