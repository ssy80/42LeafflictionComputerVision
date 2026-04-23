#!/bin/bash

DIR="./test_augmented"

if [ ! -d "$DIR/Apple" ] || [ ! -d "$DIR/Grape" ]; then
    echo "Error: $DIR/Apple or $DIR/Grape not found. Run ./scripts/augmentation_test.sh first."
    exit 1
fi

echo "Training Apple model..."
./train.py "$DIR/Apple"

echo "Training Grape model..."
./train.py "$DIR/Grape"

echo "Done."
