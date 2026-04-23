#!/bin/bash

APPLE="./Apple"
GRAPE="./Grape"

# Step 1: Augment all original images in each class (one Python call per class)
for species_dir in "$APPLE" "$GRAPE"; do
    for class_dir in "$species_dir"/*/; do
        [ -d "$class_dir" ] || continue
        echo "Augmenting $(basename "$class_dir")..."
        ./Augmentation.py "$class_dir"
    done
done

# Step 2: Find minimum class size across all 8 classes
min_count=$(
    for species_dir in "$APPLE" "$GRAPE"; do
        for class_dir in "$species_dir"/*/; do
            [ -d "$class_dir" ] || continue
            find "$class_dir" -maxdepth 1 -name "*.JPG" | wc -l
        done
    done | sort -n | head -1
)
echo "Balancing all classes to $min_count images..."

# Step 3: Trim each class down to min_count by removing excess augmented files
for species_dir in "$APPLE" "$GRAPE"; do
    for class_dir in "$species_dir"/*/; do
        [ -d "$class_dir" ] || continue
        count=$(find "$class_dir" -maxdepth 1 -name "*.JPG" | wc -l)
        excess=$(( count - min_count ))
        if [ "$excess" -gt 0 ]; then
            find "$class_dir" -maxdepth 1 -name "*.JPG" \
                \( -name "*_Flip.JPG" -o -name "*_Rotate.JPG" -o -name "*_Skew.JPG" \
                   -o -name "*_Contrast.JPG" -o -name "*_Crop.JPG" -o -name "*_Distortion.JPG" \) \
                | sort | tail -n "$excess" | while IFS= read -r f; do
                rm "$f"
            done
        fi
    done
done

# Step 4: Create augmented_directory — flat directory with actual copies of all images
rm -rf augmented_directory
mkdir augmented_directory
for species_dir in "$APPLE" "$GRAPE"; do
    for class_dir in "$species_dir"/*/; do
        [ -d "$class_dir" ] || continue
        class=$(basename "$class_dir")
        mkdir -p "augmented_directory/$class"
        cp "$class_dir"*.JPG "augmented_directory/$class/"
    done
done

echo "Done. augmented_directory/ ready — $min_count images per class."
