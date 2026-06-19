import os
import sys
import requests

"""
Submission script for the Robustness task (provided by the course, unchanged
except for the placeholders below).

FILE FORMAT     : a PyTorch *state_dict* saved as .pt (not the full model).
MODEL ARCH      : one of resnet18 / resnet34 / resnet50 (must match the weights).
MODEL REQS      : input (1,3,32,32), output (1,9), clean accuracy > 50%.
LIMITS          : one submission / 60 min (2 min cooldown on a client-side error).
"""

BASE_URL = "http://34.63.153.158"
API_KEY = "YOUR_API_KEY_HERE"          # <-- put your team API key here
MODEL_PATH = "model_best.pt"           # <-- path to your saved state_dict
MODEL_NAME = "resnet50"                # resnet18 | resnet34 | resnet50

SUBMIT = True
TASK_ID = "03-robustness"  # do not change


def die(msg):
    print(f"{msg}", file=sys.stderr)
    sys.exit(1)


if SUBMIT:
    if not os.path.isfile(MODEL_PATH):
        die(f"File not found: {MODEL_PATH}")
    try:
        with open(MODEL_PATH, "rb") as f:
            files = {"file": (os.path.basename(MODEL_PATH), f, "application/x-pytorch")}
            resp = requests.post(
                f"{BASE_URL}/submit/{TASK_ID}",
                headers={"X-API-Key": API_KEY},
                files=files,
                data={"model_name": MODEL_NAME},
                timeout=(30, 1800),   # resnet50 is ~94 MB and the server evaluates synchronously
            )
        try:
            body = resp.json()
        except Exception:
            body = {"raw_text": resp.text}
        if resp.status_code == 413:
            die("Upload rejected: file too large (HTTP 413).")
        resp.raise_for_status()
        print("Successfully submitted.")
        print("Server response:", body)
    except requests.exceptions.RequestException as e:
        print(f"Submission error: {e}")
        sys.exit(1)
