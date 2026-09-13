# SoapBowl: Setup

## Install dependencies
pip install -r requirements.txt

## First run
python3 soap_bowl_full.py
# Downloads ~500MB of astronomical data to cache/ on first run (5-10 min).
# Subsequent runs use cached data and complete in seconds.

## Run individual theorems (examples)
python3 soap_bowl_full.py --only 2    # T2 Cosmic Foam Scaling
python3 soap_bowl_full.py --only 31   # T31 Sakharov's Number
python3 soap_bowl_full.py --only 157  # T57 Hayflick's Pressure
python3 soap_bowl_full.py --only 158  # T58 Calment's Ceiling
python3 soap_bowl_full.py --only 159  # T59 Consciousness Threshold
