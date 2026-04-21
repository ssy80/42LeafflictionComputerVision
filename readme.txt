Setup
    ./setup.sh

Activate virtual env 
    source my_env/bin/activate

View Distribution
    python3 ./Distribution.py ./images/Apple/
    python3 ./Distribution.py ./images/Grape/
    
Augmentation
    ./augmentation.sh

View Distribution again
    python3 ./Distribution.py ./images/Apple/
    python3 ./Distribution.py ./images/Grape/

Transformations
    Single image
        python3 ./Transformation.py -src "./images/Apple/Apple_Black_rot/image (1).JPG"
    Directory
        python3 ./Transformation.py -src "./images/Apple/Apple_Black_rot" -dst ./images/Transformations/Apple/Apple_Black_rot
        python3 ./Transformation.py -src "./images/Apple/Apple_healthy" -dst ./images/Transformations/Apple/Apple_healthy
        python3 ./Transformation.py -src "./images/Apple/Apple_rust" -dst ./images/Transformations/Apple/Apple_rust
        python3 ./Transformation.py -src "./images/Apple/Apple_scab" -dst ./images/Transformations/Apple/Apple_scab
        python3 ./Transformation.py -src "./images/Grape/Grape_Black_rot" -dst ./images/Transformations/Grape/Grape_Black_rot
        python3 ./Transformation.py -src "./images/Grape/Grape_Esca" -dst ./images/Transformations/Grape/Grape_Esca
        python3 ./Transformation.py -src "./images/Grape/Grape_healthy" -dst ./images/Transformations/Grape/Grape_healthy
        python3 ./Transformation.py -src "./images/Grape/Grape_spot" -dst ./images/Transformations/Grape/Grape_spot

Training
    python3 ./train.py ./images/Apple ./images/transformed/Apple
    python3 ./train.py ./images/Grape ./images/transformed/Grape

Predict
    python3 ./predict.py "./test/Apple/image (1)_Contrast_analyze.JPG" ./images/transformed/Apple/splited ./images/Apple
    python3 ./predict.py "./test/Apple/image (1).JPG" ./images/transformed/Apple/splited ./images/Apple
    python3 ./predict.py "./test/Apple/image (2)_Skew.JPG" ./images/transformed/Apple/splited ./images/Apple
    python3 ./predict.py "./test/Apple/image (4)_Distortion.JPG" ./images/transformed/Apple/splited ./images/Apple
    python3 ./predict.py "./test/Apple/image (29).JPG" ./images/transformed/Apple/splited ./images/Apple
    
    python3 ./predict.py "./test/Grape/image (1)_Crop.JPG" ./images/transformed/Grape/splited ./images/Grape
    python3 ./predict.py "./test/Grape/image (1)_mask.JPG" ./images/transformed/Grape/splited ./images/Grape
    python3 ./predict.py "./test/Grape/image (2)_Rotate.JPG" ./images/transformed/Grape/splited ./images/Grape
    python3 ./predict.py "./test/Grape/image (5)_Flip.JPG" ./images/transformed/Grape/splited ./images/Grape
    python3 ./predict.py "./test/Grape/image (10).JPG" ./images/transformed/Grape/splited ./images/Grape
