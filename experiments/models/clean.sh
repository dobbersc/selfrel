#!/usr/bin/env bash

find . -type f -name '*.pt' -delete

find . -type f -name 'relation-overview.parquet' -delete
find . -type f -name 'scored-relation-overview.parquet' -delete

find . -type f -name 'annotated-support-dataset.conllup' -delete
