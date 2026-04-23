#!/bin/bash

DIR="./test"

if [ ! -d "$DIR/Apple" ] || [ ! -d "$DIR/Grape" ]; then
    echo "Error: $DIR/Apple or $DIR/Grape not found."
    exit 1
fi

echo "Training Apple model..."
./train.py "$DIR/Apple" "$DIR/transformed/Apple" 

echo "Training Grape model..."
./train.py "$DIR/Grape" "$DIR/transformed/Grape"

echo "Done."
