import glob

label_path = "/data/local/aschwab/data/realColon_640x640/labels/train/*.txt"

for f in glob.glob(label_path):
    with open(f) as fh:
        lines = fh.readlines()

    # Skip negatives (empty files)
    if len(lines) == 0:
        continue

    # Check each line for validity
    bad = False
    for i, line in enumerate(lines, 1):
        parts = line.strip().split()
        if len(parts) != 5:
            print(f"Bad line in {f}:{i} -> {line.strip()}")
            bad = True
            continue
        try:
            cls, x, y, w, h = map(float, parts)
        except ValueError:
            print(f"Non-numeric values in {f}:{i} -> {line.strip()}")
            bad = True
            continue
        if not (0 <= x <= 1 and 0 <= y <= 1 and 0 <= w <= 1 and 0 <= h <= 1):
            print(f"Out of range box in {f}:{i} -> {line.strip()}")
            bad = True
        if w <= 0 or h <= 0:
            print(f"Zero/negative size in {f}:{i} -> {line.strip()}")
            bad = True

    if bad:
        print(f"⚠️ Invalid boxes found in {f}")

