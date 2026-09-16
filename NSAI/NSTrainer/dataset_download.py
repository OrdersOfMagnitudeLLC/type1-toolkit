#!/usr/bin/env python3
import gzip
import json
import os
import requests

TARGET_BYTES = 50 * 1024 * 1024
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_PATH = os.path.join(OUT_DIR, "c4_sample.txt")
URL = "https://huggingface.co/datasets/allenai/c4/resolve/main/en/c4-validation.00000-of-00008.json.gz"


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    r = requests.get(URL, stream=True, timeout=120)
    r.raise_for_status()
    collected = 0
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        with gzip.open(r.raw, "rt", encoding="utf-8") as gz:
            for line in gz:
                if collected >= TARGET_BYTES:
                    break
                example = json.loads(line)
                text = example["text"]
                f.write(text + "\n")
                collected += len(text.encode("utf-8"))
    print(f"Saved {collected} bytes to {OUT_PATH}")


if __name__ == "__main__":
    main()
