
PYTHON  = python3
VENV    = my_env
SRC     = ./src
DATA    = ./dataset/raw
TEST    = ./dataset/test
MODELS  = ./models
TRANSFORMED = ./dataset/transformed
MAKEFLAGS += --no-print-directory
APPLE_IMG  ?= $(shell find $(TEST)/Apple -name "*.JPG" 2>/dev/null | shuf -n 1)
GRAPE_IMG  ?= $(shell find $(TEST)/Grape -name "*.JPG" 2>/dev/null | shuf -n 1)
IMG        ?= $(APPLE_IMG)
MODEL      ?= $(MODELS)/Apple/splited

AUGMENTED   = ./dataset/augmented
APPLE_DATA  = $(AUGMENTED)/Apple
GRAPE_DATA  = $(AUGMENTED)/Grape
APPLE_MODEL = $(MODELS)/Apple
GRAPE_MODEL = $(MODELS)/Grape

DIST_BEFORE_APPLE = distribution/before/Apple
DIST_BEFORE_GRAPE = distribution/before/Grape
DIST_AFTER_APPLE  = distribution/after/Apple
DIST_AFTER_GRAPE  = distribution/after/Grape

export PYTHONPATH := ./src

.PHONY: install train train-apple train-grape predict predict-apple predict-grape distribution distribution-after augment transform transform-one transform-dirs test lint logs clean fclean


# 1. Analyse raw dataset balance
distribution:
	$(PYTHON) -m distribution.distribution $(DATA)/Apple/ $(DIST_BEFORE_APPLE)
	$(PYTHON) -m distribution.distribution $(DATA)/Grape/ $(DIST_BEFORE_GRAPE)

distribution-after:
	$(PYTHON) -m distribution.distribution $(APPLE_DATA)/ $(DIST_AFTER_APPLE)
	$(PYTHON) -m distribution.distribution $(GRAPE_DATA)/ $(DIST_AFTER_GRAPE)

# 2. Balance dataset via augmentation → dataset/augmented/
augment:
	./scripts/augmentation.sh

# 3. Transform the dataset
transform: transform-dirs

transform-one:
	$(PYTHON) -m transformation.transformation -src "$(DATA)/Apple/Apple_Black_rot/image (1).JPG"

transform-dirs:
	$(PYTHON) -m transformation.transformation -src "$(DATA)/Apple/Apple_Black_rot" -dst $(TRANSFORMED)/Apple/Apple_Black_rot
	$(PYTHON) -m transformation.transformation -src "$(DATA)/Apple/Apple_healthy"   -dst $(TRANSFORMED)/Apple/Apple_healthy
	$(PYTHON) -m transformation.transformation -src "$(DATA)/Apple/Apple_rust"      -dst $(TRANSFORMED)/Apple/Apple_rust
	$(PYTHON) -m transformation.transformation -src "$(DATA)/Apple/Apple_scab"      -dst $(TRANSFORMED)/Apple/Apple_scab
	$(PYTHON) -m transformation.transformation -src "$(DATA)/Grape/Grape_Black_rot" -dst $(TRANSFORMED)/Grape/Grape_Black_rot
	$(PYTHON) -m transformation.transformation -src "$(DATA)/Grape/Grape_Esca"      -dst $(TRANSFORMED)/Grape/Grape_Esca
	$(PYTHON) -m transformation.transformation -src "$(DATA)/Grape/Grape_healthy"   -dst $(TRANSFORMED)/Grape/Grape_healthy
	$(PYTHON) -m transformation.transformation -src "$(DATA)/Grape/Grape_spot"      -dst $(TRANSFORMED)/Grape/Grape_spot

# 4. Train separate models for Apple and Grape
train: train-apple train-grape

train-apple:
	$(PYTHON) -m training.train $(APPLE_DATA) $(APPLE_MODEL)

train-grape:
	$(PYTHON) -m training.train $(GRAPE_DATA) $(GRAPE_MODEL)

# 5. Predict
predict:
	$(PYTHON) -m prediction.predict "$(IMG)" $(MODEL)

predict-apple:
	$(PYTHON) -m prediction.predict "$(APPLE_IMG)" $(APPLE_MODEL)/splited

predict-grape:
	$(PYTHON) -m prediction.predict "$(GRAPE_IMG)" $(GRAPE_MODEL)/splited

# Supplementary scripts
test:
	pytest tests/ -v

lint:
	flake8 $(SRC)/

logs:
	$(VENV)/bin/tensorboard --logdir logs/

clean:
	rm -rf $(MODELS)/Apple $(MODELS)/Grape $(MODELS)/splited $(MODELS)/test dataset/transformed logs/ dataset/augmented/ distribution/ __pycache__ src/__pycache__ .pytest_cache

