#!/bin/bash

for i in {1..170}
do
    python3 ./Augmentation.py "./images/Apple/Apple_Black_rot/image ($i).JPG"
done

for i in {1..228}
do
    python3 ./Augmentation.py "./images/Apple/Apple_rust/image ($i).JPG"
done

for i in {1..169}
do
    python3 ./Augmentation.py "./images/Apple/Apple_scab/image ($i).JPG"
done


for i in {1..34}
do
    python3 ./Augmentation.py "./images/Grape/Grape_Black_rot/image ($i).JPG"
done

for i in {1..160}
do
    python3 ./Augmentation.py "./images/Grape/Grape_healthy/image ($i).JPG"
done

for i in {1..51}
do
    python3 ./Augmentation.py "./images/Grape/Grape_spot/image ($i).JPG"
done


