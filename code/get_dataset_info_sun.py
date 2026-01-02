# Get total & per split sums
BASE=/data/local/aschwab/data/sun_split
echo "=== IMAGES ==="
for s in train val test; do
  printf "%6s: " "$s"
  find "$BASE/images/$s" -type f -iname '*.jpg' | wc -l
done

echo "=== LABELS ==="
for s in train val test; do
  printf "%6s: " "$s"
  find "$BASE/labels/$s" -type f -name '*.txt' | wc -l
done

echo "=== SUMMARY ==="
echo "Images total: $(find "$BASE/images" -type f -iname '*.jpg' | wc -l)"
echo "Labels total: $(find "$BASE/labels" -type f -name '*.txt' | wc -l)"


# total img pos/neg
BASE=/data/local/aschwab/data/sun_split

echo "=== TOTAL IMAGES BY CASE TYPE ==="
POS=$(for i in $(seq 1 100); do
  find "$BASE/images" -type f -iname "case${i}_*.jpg"
done | wc -l)

NEG=$(for i in $(seq 101 113); do
  find "$BASE/images" -type f -iname "case${i}_*.jpg"
done | wc -l)

echo "Positive (cases 1–100): $POS"
echo "Negative (cases 101–113): $NEG"
echo "Total: $((POS + NEG))"


# img by case
BASE=/data/local/aschwab/data/sun_split

echo "=== IMAGE COUNT PER CASE (unique split per case) ==="
for i in $(seq 1 113); do
  if [ "$i" -le 100 ]; then
    type="POS"
  else
    type="NEG"
  fi

  split="NONE"
  total=0

  # find which split this case is in (and how many images)
  for s in train val test; do
    cnt=$(find "$BASE/images/$s" -type f -iname "case${i}_*.jpg" | wc -l)
    if [ "$cnt" -gt 0 ]; then
      split="$s"
      total="$cnt"
      break   # since splits are by case, we can stop here
    fi
  done

  printf "case%3d (%s) -> split=%-5s | images=%6d\n" "$i" "$type" "$split" "$total"
done
