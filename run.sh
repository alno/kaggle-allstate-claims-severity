#!/bin/sh

python prepare-basic.py
python prepare-numeric-boxcox.py
python prepare-categorical-encoded.py
python prepare-categorical-counts.py

python prepare-folds.py
