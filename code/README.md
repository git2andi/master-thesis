````md
# `code/`
Small utilities for dataset inspection and qualitative verification.

## `get_dataset_info_piccolo.py`
Run:
```bash
python get_dataset_info_piccolo.py \
  --orig /data/local/aschwab/data/piccolo \
  --split /data/local/aschwab/data/piccolo_split
````

## `get_dataset_info_sun.py`

Run:
```bash
python get_dataset_info_sun.py \
  --base /data/local/aschwab/data/sun_split
```

## `get_dataset_info_realcolon.py`
Run:

```bash
python get_dataset_info_realcolon.py \
  --root /data/local/aschwab/data/real_colon_allPos_allNeg_onlyPatient
```

## `verify_piccolo.py`
Run:
```bash
python verify_piccolo.py
```

## `verify_sun.py`
Run:
```bash
python verify_sun.py
```