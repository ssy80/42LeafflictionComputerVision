#!/bin/bash

SRC="./test_images"

if [ ! -d "$SRC" ]; then
    echo "Error: $SRC not found."
    exit 1
fi

for unit_dir in "$SRC"/*/; do
    [ -d "$unit_dir" ] || continue
    echo "Augmenting $(basename "$unit_dir")..."
    find "$unit_dir" -maxdepth 1 -name "*.JPG" \
        ! -name "*_Flip.JPG" \
        ! -name "*_Rotate.JPG" \
        ! -name "*_Skew.JPG" \
        ! -name "*_Contrast.JPG" \
        ! -name "*_Crop.JPG" \
        ! -name "*_Distortion.JPG" \
        -print0 | while IFS= read -r -d '' f; do
        ./Augmentation.py "$f"
    done
done

echo "Done. Augmented images saved alongside originals in $SRC/"
