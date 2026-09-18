#!/bin/bash
# Bella Stellar Pipeline: Breakthrough Listen Re-analysis
# Dataset: http://blpd0.ssl.berkeley.edu/lband2017/All_hits_turbo_seti.csv

export BELLA_BL_DATASET_PATH=/path/to/bl_hits_692stars.csv

python3 bella.py signal --dry-run
python3 bella.py signal --max-hits 29000000
