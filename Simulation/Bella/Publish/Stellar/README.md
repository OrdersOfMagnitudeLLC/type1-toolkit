# Bella Stellar Pipeline: Re-analysis of Breakthrough Listen 692-Star Survey

## Abstract

We processed the full Breakthrough Listen L-band public dataset (28.9M narrowband hits across 692 nearby stars, 1.1-1.9 GHz, GBT) using Bella's streaming stellar pipeline with NS-accelerated ranking criteria. Full scan completed in under 5 minutes on a 16GB consumer CPU. Our drift-filtered hydrogen-line ranking produced 6,787 candidates after removing zero-drift galactic HI emission and OFF-source RFI. Top candidate: HD 111312 (HIP62505, K2.5V, Virgo) with two independent detections showing consistent Doppler drift (0.483 and 0.512 Hz/s) near the hydrogen line at SNR 536 and 78.8: not present in BL's published top 11 events. Full candidate list of 6,787 ranked hits included. We make no detection claim; this is a reproducible re-ranking of public data using different criteria.

## Dataset

- **URL:** http://blpd0.ssl.berkeley.edu/lband2017/All_hits_turbo_seti.csv
- **Size:** 10.2 GB
- **License:** CC BY 4.0
- **Rows:** 28.9M narrowband hits
- **Targets:** 692 stars

## Method

- Streaming CSV processing with constant memory footprint
- OFF-source filter (`"OFF" in Source`)
- Drift threshold: `abs(drift) > 0.01 Hz/s`
- Frequency match within 0.5 MHz of known technosignature frequencies:
  - hydrogen: 1420.405 MHz
  - hydroxyl: 1612.231 MHz
  - water hole: 1420-1727 MHz
  - pi×HI: 4462 MHz
  - deuterium: 327 MHz
- Scoring: `0.4 × (SNR/100) + 0.3 × (drift/2.0) + 0.3`

## Performance

- 27,299,941 hits processed
- 12,007 frequency matches before drift filter
- 6,787 candidates after drift filter
- Top-10,000 min-heap kept in memory
- Constant ~50MB RAM regardless of file size

## Top Candidate

HIP62505 (HD 111312, K2.5Vk:, V=7.87, RA=12h48m32s, Dec=-15d43m08s) produced two matched hits:

- `candidate_HIP62505.json`: the two hit records with exact frequency, SNR, and drift values
- `all_candidates_6787.json`: full ranked candidate list

## Reproduce

See `pipeline_commands.sh` for the exact commands.

```bash
bash Publish/Stellar/pipeline_commands.sh
```

## Limitations

- RFI cannot be excluded without raw filterbank data or re-observation.
- Two consistent drift detections on the same target is necessary but not sufficient for technosignature classification.
- Full-sky coverage requires additional Breakthrough Listen data releases.

License (OOM Stellar Pipeline): CC BY 4.0: Orders of Magnitude LLC: https://creativecommons.org/licenses/by/4.0/
