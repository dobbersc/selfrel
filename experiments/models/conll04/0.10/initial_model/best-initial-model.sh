#!/usr/bin/env bash

# Train the optimal initial teacher model on 10% of the training data from the CoNLL04 corpus.
# The hyperparameters to train the optimal model were found
# using the `initial-model-sweep.yaml` grid search configuration.

selfrel train conll04 \
  --support-dataset /glusterfs/dfs-gfs-dist/dobbersc-pub/cc-news-small/cc-news-ner.conllup \
  --base-path initial-model \
  --down-sample-train 0.10 \
  --max-epochs 20 \
  --batch-size 8 \
  --learning-rate 5e-5 \
  --no-use-final-model-for-evaluation \
  --self-training-iterations 0 \
  --seed 8
