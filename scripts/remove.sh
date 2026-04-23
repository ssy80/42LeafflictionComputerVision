#!/bin/bash

# Deactivate virtual environment if active
deactivate 2>/dev/null

# Remove virtual environment
rm -rf my_env

# Remove augmented_directory directory
rm -rf augmented_directory

# Remove augmented files from Apple and Grape class directories
for species_dir in "./Apple" "./Grape"; do
    for class_dir in "$species_dir"/*/; do
        [ -d "$class_dir" ] || continue
        find "$class_dir" -maxdepth 1 -name "*.JPG" \
            \( -name "*_Flip.JPG" -o -name "*_Rotate.JPG" -o -name "*_Skew.JPG" \
               -o -name "*_Contrast.JPG" -o -name "*_Crop.JPG" -o -name "*_Distortion.JPG" \) \
            -delete
    done
done

# Remove augmented files and splited/ from test directories
for species_dir in "./test/Apple" "./test/Grape"; do
    for class_dir in "$species_dir"/*/; do
        [ -d "$class_dir" ] || continue
        [[ "$(basename "$class_dir")" == "splited" ]] && continue
        find "$class_dir" -maxdepth 1 -name "*.JPG" \
            \( -name "*_Flip.JPG" -o -name "*_Rotate.JPG" -o -name "*_Skew.JPG" \
               -o -name "*_Contrast.JPG" -o -name "*_Crop.JPG" -o -name "*_Distortion.JPG" \) \
            -delete
    done
    rm -rf "$species_dir/splited"
done
rm -f ./test/Apple_learnings.zip ./test/Grape_learnings.zip

# Remove test_augmented directory
rm -rf test_augmented

# Remove other generated data
rm -rf models/ logs/ distribution/ debug/ test/ __pycache__ .pytest_cache
