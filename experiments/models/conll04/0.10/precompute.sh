#!/usr/bin/env bash

# Pre-compute the support datasets and relation overviews
# for the optimal initial teacher model on 10% of the training data from the CoNLL04 corpus.
# The hyperparameters to train the optimal model
# were found using the initial-model-sweep.yaml` grid search configuration.

# The entropy selection strategy hyperparameters are arbitrarily picked
# since they do not impact the annotated support dataset and scored relation overview.

selfrel train conll04 \
  --support-dataset /glusterfs/dfs-gfs-dist/dobbersc-pub/cc-news/cc-news-ner.conllup \
  --base-path /glusterfs/dfs-gfs-dist/dobbersc-pub/cc-news/precomputed/conll04/0.10 \
  --self-training-iterations 1 \
  --down-sample-train 0.10 \
  --max-epochs 20 10 \
  --batch-size 8 32 \
  --learning-rate 5e-5 \
  --no-use-final-model-for-evaluation \
  --selection-strategy entropy \
  --base 2 \
  --max-entropy 0.8 \
  --min-occurrence 10 \
  --max-occurrence 100 \
  --top-k 3000 \
  --label-distribution Kill=0.15 Live_In=0.15 Located_In=0.15 OrgBased_In=0.15 Work_For=0.15 no_relation=0.25 \
  --num-actors 4 \
  --seed 8
