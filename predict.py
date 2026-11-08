import argparse
from pathlib import Path
import joblib

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--response", required=True, help="Response text to classify")
    p.add_argument("--model", default="response_only__linear_svc",
                   help="Model name saved in models/, without .joblib")
    p.add_argument("--knowledge", default="")
    p.add_argument("--dialogue-history", default="")
    args = p.parse_args()
    path = Path("models") / f"{args.model}.joblib"
    if not path.exists():
        raise SystemExit(f"Model not found: {path}. Train the project first.")
    model = joblib.load(path)
    if args.model.startswith("context_plus_response__"):
        text = f"KNOWLEDGE: {args.knowledge}\nDIALOGUE: {args.dialogue_history}\nRESPONSE: {args.response}"
    else:
        text = args.response
    pred = int(model.predict([text])[0])
    label = "LIKELY HALLUCINATED (dataset label 1)" if pred == 1 else "LIKELY NON-HALLUCINATED (dataset label 0)"
    print(label)
    print("This is a model prediction, not a factual verification.")
if __name__ == "__main__":
    main()
