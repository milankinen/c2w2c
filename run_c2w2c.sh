#!/usr/bin/env bash

export DEVICE=cpu

./run --training data/training.txt \
      --test data/validation.txt \
      --save-weights c2w2c.h5 \
      --load-weights c2w2c.h5 \
      --mode c2w2c \
      -w 21 \
      --gen-text 20
