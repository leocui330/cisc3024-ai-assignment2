"""Verify published probabilities without downloading images or retraining."""
from pathlib import Path
import json
import numpy as np
from metrics import metrics

def main():
    root=Path(__file__).resolve().parents[1]
    folder=root/"results"
    stored=json.loads((folder/"test_results.json").read_text(encoding="utf-8"))
    values={}
    for seed in [11,22,33]:
        z=np.load(folder/f"seed_{seed}_predictions.npz")
        if len(z["labels"])!=10000:
            raise AssertionError("Unexpected test count")
        for method in ["map","map_temperature","vbll","vbll_mean"]:
            actual=metrics(z[method],z["labels"])
            expected=next(r for r in stored["per_seed"] if r["seed"]==seed and r["method"]==method)
            for key in ["accuracy","nll","brier","ece"]:
                if abs(actual[key]-expected[key])>1e-10:
                    raise AssertionError(f"Mismatch {seed}/{method}/{key}")
                values.setdefault((method,key),[]).append(actual[key])
            if actual["clipped_true_probability_count"]:
                raise AssertionError("NLL clipping")
    for (method,key),numbers in values.items():
        expected=stored["aggregate"][method][key]
        assert abs(np.mean(numbers)-expected["mean"])<1e-10
        assert abs(np.std(numbers,ddof=1)-expected["sample_std"])<1e-10
    print("PASS: 12 prediction sets, 48 per-run metrics, all aggregate means/sample SDs; no NLL clipping.")

if __name__=="__main__":main()
