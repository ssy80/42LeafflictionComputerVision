Setup
    ./scripts/setup.sh

Activate virtual env 
    source my_env/bin/activate

Download leaves.zip from intra.
Create ./images/Apple directory
Create ./images/Grape directory
Move the respective class images to the above created directories from the unzipped leaves.zip

View Distribution
    python3 ./Distribution.py ./images/Apple/
    python3 ./Distribution.py ./images/Grape/
    
Augmentation
    ./scripts/augmentation.sh

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
    Full Training
    python3 ./train.py ./images/Apple ./images/transformed/Apple
    python3 ./train.py ./images/Grape ./images/transformed/Grape

    Small testing set training and evaluation (400)
    Copy 100 images from each class in the original leaves.zip to ./images/test/Apple and ./images/test/Grape
    python3 ./train.py ./images/test/Apple ./images/transformed/test/Apple
    python3 ./train.py ./images/test/Grape ./images/transformed/test/Grape


Predict - python3 ./predict.py [test image src] [trained model location]
     
    python3 ./predict.py "./test/Apple/image (1).JPG" ./images/transformed/Apple/splited
    python3 ./predict.py "./test/Apple/image (29).JPG" ./images/transformed/Apple/splited
    
    python3 ./predict.py "./test/Grape/image (1).JPG" ./images/transformed/Grape/splited
    python3 ./predict.py "./test/Grape/image (10).JPG" ./images/transformed/Grape/splited
