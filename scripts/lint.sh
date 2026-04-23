#!/bin/bash

my_env/bin/python3 -m flake8 \
    Distribution.py Augmentation.py Transformation.py train.py predict.py utils.py split_file.py
