Setup
    ./setup.sh

Activate virtual env 
    source my_env/bin/activate

View Distribution
    python3 ./Distribution.py ./images/Apple/
    python3 ./Distribution.py ./images/Grape/
    
Augmentation
    ./augmentation.sh

View Distribution
    python3 ./Distribution.py ./images/Apple/
    python3 ./Distribution.py ./images/Grape/

Transformations
    Single image
        python3 ./Transformation.py -src "./images/Apple/Apple_Black_rot/image (1).JPG"
    Directory
        python3 ./Transformation.py -src "./images/Apple/Apple_rust" -dst ./images/Transformations/Apple/Apple_rust

Training
    python3 ./train.py ./images/Apple