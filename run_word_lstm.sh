#!/usr/bin/env bash

export DEVICE=cpu

./run --training data/training.txt \
      --test data/validation.txt \
      --save-weights word.h5 \
      --load-weights word.h5 \
      --mode word \
      --gen-text 20
