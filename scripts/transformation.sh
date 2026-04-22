#!/bin/bash

SRC="./augmented_directory"

if [ ! -d "$SRC" ]; then
    echo "Error: $SRC not found. Run ./scripts/augmentation.sh first."
    exit 1
fi

for class_dir in "$SRC"/*/; do
    [ -d "$class_dir" ] || continue
    class=$(basename "$class_dir")

    case "$class" in
        Apple_*) species="Apple" ;;
        Grape_*) species="Grape" ;;
        *)
            echo "Skipping unknown class: $class"
            continue
            ;;
    esac

    echo "Transforming $class..."
    ./Transformation.py -src "$class_dir" -dst "$SRC/$species/$class"
done

echo "Done. Transformed images saved under $SRC/Apple/ and $SRC/Grape/"
