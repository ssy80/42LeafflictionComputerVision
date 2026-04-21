#!/bin/bash

SRC="./dataset/raw"
DST="./dataset/augmented"

for species_dir in "$SRC"/*/; do
    species=$(basename "$species_dir")
    for class_dir in "$species_dir"*/; do
        [ -d "$class_dir" ] || continue
        class=$(basename "$class_dir")
        dst_dir="$DST/$species/$class"
        python3 ./src/Augmentation.py "$class_dir" "$dst_dir"
    done
done
