#!/bin/bash

SRC="./augmented_directory"

if [ ! -d "$SRC" ]; then
    echo "Error: $SRC not found. Run ./scripts/augmentation.sh first."
    exit 1
fi

for species_dir in "$SRC"/*/; do
    [ -d "$species_dir" ] || continue
    species=$(basename "$species_dir")

    for class_dir in "$species_dir"*/; do
        [ -d "$class_dir" ] || continue
        class=$(basename "$class_dir")
        dst="$SRC/${species}_transformed/$class"

        echo "Transforming $class..."
        ./Transformation.py -src "$class_dir" -dst "$dst"
    done
done

echo "Done. Transformed images saved under $SRC/Apple_transformed/ and $SRC/Grape_transformed/"
