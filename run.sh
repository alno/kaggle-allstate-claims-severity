#!/bin/sh


prepare() {
  echo "Preparing $1..."
  python prepare-$1.py
}

train() {
  echo "Training $1..."
  time python -u train.py --threads 4 $1 | tee logs/$1.log
}


# Prepare features
prepare basic
prepare numeric-boxcox
prepare numeric-scaled
prepare numeric-rank-norm
prepare categorical-encoded
prepare categorical-counts
prepare categorical-dummy
prepare svd

# Basic models
train lr-ce
train lr-cd
train lr-svd

train et-ce
train rf-ce
train gb-ce

# LibFM
train libfm-cd
train libfm-svd

# LightGBM
train lgb-ce

# XGB
train xgb-ce
train xgb-ce-2

train xgbf-ce
train xgbf-ce-2
