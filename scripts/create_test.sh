#!/bin/bash

APPLE="./Apple"
GRAPE="./Grape"
DST="./test"
COUNT=50

rm -rf "$DST"

for species_dir in "$APPLE" "$GRAPE"; do
    species=$(basename "$species_dir")
    for class_dir in "$species_dir"/*/; do
        [ -d "$class_dir" ] || continue
        class=$(basename "$class_dir")
        dest="$DST/$species/$class"
        mkdir -p "$dest"

        find "$class_dir" -maxdepth 1 -name "*.JPG" \
            ! -name "*_Flip.JPG" \
            ! -name "*_Rotate.JPG" \
            ! -name "*_Skew.JPG" \
            ! -name "*_Contrast.JPG" \
            ! -name "*_Crop.JPG" \
            ! -name "*_Distortion.JPG" \
            | shuf -n "$COUNT" | while IFS= read -r f; do
            cp "$f" "$dest/"
        done

        actual=$(find "$dest" -name "*.JPG" | wc -l)
        echo "$class: $actual images"
    done
done

echo "Done. Test set created at $DST/"
