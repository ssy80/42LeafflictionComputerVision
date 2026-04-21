
PYTHON  = python3
VENV    = my_env
SRC     = ./src
DATA    = ./dataset/raw
TEST    = ./dataset/test
MODELS  = ./models
TRANSFORMED = ./dataset/transformed
MAKEFLAGS += --no-print-directory
IMG    ?= dataset/test/Apple/image (1).JPG
MODEL  ?= $(MODELS)/Apple/splited

AUGMENTED   = ./dataset/augmented
APPLE_DATA  = $(AUGMENTED)/Apple
GRAPE_DATA  = $(AUGMENTED)/Grape
APPLE_MODEL = $(MODELS)/Apple
GRAPE_MODEL = $(MODELS)/Grape

DIST_BEFORE_APPLE = distribution/before/Apple
DIST_BEFORE_GRAPE = distribution/before/Grape
DIST_AFTER_APPLE  = distribution/after/Apple
DIST_AFTER_GRAPE  = distribution/after/Grape

.PHONY: install train train-apple train-grape predict predict-apple predict-grape distribution distribution-after augment transform transform-one transform-dirs test lint logs clean fclean


# 1. Analyse raw dataset balance
distribution:
	$(PYTHON) $(SRC)/Distribution.py $(DATA)/Apple/ $(DIST_BEFORE_APPLE)
	$(PYTHON) $(SRC)/Distribution.py $(DATA)/Grape/ $(DIST_BEFORE_GRAPE)

distribution-after:
	$(PYTHON) $(SRC)/Distribution.py $(APPLE_DATA)/ $(DIST_AFTER_APPLE)
	$(PYTHON) $(SRC)/Distribution.py $(GRAPE_DATA)/ $(DIST_AFTER_GRAPE)

# 2. Balance dataset via augmentation → dataset/augmented/
augment:
	./scripts/augmentation.sh

# 3. Train separate models for Apple and Grape
train: train-apple train-grape

train-apple:
	$(PYTHON) $(SRC)/train.py $(APPLE_DATA) $(APPLE_MODEL)

train-grape:
	$(PYTHON) $(SRC)/train.py $(GRAPE_DATA) $(GRAPE_MODEL)

# 4. Predict
predict:
	$(PYTHON) $(SRC)/predict.py "$(IMG)" $(MODEL)

predict-apple:
	$(PYTHON) $(SRC)/predict.py "$(IMG)" $(APPLE_MODEL)/splited

predict-grape:
	$(PYTHON) $(SRC)/predict.py "$(IMG)" $(GRAPE_MODEL)/splited

# Supplementary (Use only when needed)
transform: transform-dirs

transform-one:
	$(PYTHON) $(SRC)/Transformation.py -src "$(DATA)/Apple/Apple_Black_rot/image (1).JPG"

transform-dirs:
	$(PYTHON) $(SRC)/Transformation.py -src "$(DATA)/Apple/Apple_Black_rot" -dst $(TRANSFORMED)/Apple/Apple_Black_rot
	$(PYTHON) $(SRC)/Transformation.py -src "$(DATA)/Apple/Apple_healthy"   -dst $(TRANSFORMED)/Apple/Apple_healthy
	$(PYTHON) $(SRC)/Transformation.py -src "$(DATA)/Apple/Apple_rust"      -dst $(TRANSFORMED)/Apple/Apple_rust
	$(PYTHON) $(SRC)/Transformation.py -src "$(DATA)/Apple/Apple_scab"      -dst $(TRANSFORMED)/Apple/Apple_scab
	$(PYTHON) $(SRC)/Transformation.py -src "$(DATA)/Grape/Grape_Black_rot" -dst $(TRANSFORMED)/Grape/Grape_Black_rot
	$(PYTHON) $(SRC)/Transformation.py -src "$(DATA)/Grape/Grape_Esca"      -dst $(TRANSFORMED)/Grape/Grape_Esca
	$(PYTHON) $(SRC)/Transformation.py -src "$(DATA)/Grape/Grape_healthy"   -dst $(TRANSFORMED)/Grape/Grape_healthy
	$(PYTHON) $(SRC)/Transformation.py -src "$(DATA)/Grape/Grape_spot"      -dst $(TRANSFORMED)/Grape/Grape_spot

test:
	pytest tests/ -v

lint:
	flake8 $(SRC)/

logs:
	$(VENV)/bin/tensorboard --logdir logs/

clean:
	rm -rf $(MODELS)/Apple $(MODELS)/Grape $(MODELS)/splited $(MODELS)/test dataset/transformed logs/ dataset/augmented/ distribution/ __pycache__ src/__pycache__ .pytest_cache

fclean: clean
	bash scripts/remove.sh
