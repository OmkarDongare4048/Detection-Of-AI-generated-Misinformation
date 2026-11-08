import json
from pathlib import Path
import pandas as pd
def main():
    p=Path("results/metrics.json")
    if not p.exists():
        raise SystemExit("No metrics found. Run: python train.py --data dialogue_data.csv --seed 42")
    d=json.loads(p.read_text(encoding="utf-8"))
    print("Best model by validation F1:", d["best_model_by_validation_f1"])
    rows=[]
    for m in d["models"]:
        rows.append({k:m[k] for k in ["feature_set","model","validation_f1","test_accuracy","test_precision","test_recall","test_f1","paired_accuracy"]})
    print(pd.DataFrame(rows).to_string(index=False))
    print("\nCaution:", d["caution"])
if __name__=="__main__": main()
