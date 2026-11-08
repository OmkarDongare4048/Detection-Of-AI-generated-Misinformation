import argparse, json, logging, os
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
REQUIRED = ["knowledge", "dialogue_history", "right_response", "hallucinated_response"]

def main():
    parser = argparse.ArgumentParser(description="Train hallucination detection baseline models.")
    parser.add_argument("--data", default="dialogue_data.csv")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    root = Path(".").resolve()
    data_path = Path(args.data)
    if not data_path.exists():
        raise SystemExit(f"ERROR: Dataset not found: {data_path.resolve()}")
    df = pd.read_csv(data_path)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise SystemExit(f"ERROR: Missing required columns: {missing}")
    logging.info("Loaded %d original rows and %d columns", len(df), len(df.columns))
    original_report = {
        "original_rows": int(len(df)),
        "columns": df.columns.tolist(),
        "missing_values_by_column": {c: int(df[c].isna().sum()) for c in REQUIRED},
        "exact_duplicate_rows": int(df.duplicated(subset=REQUIRED).sum())
    }
    (root/"results").mkdir(exist_ok=True)
    (root/"results/dataset_inspection.json").write_text(json.dumps(original_report, indent=2), encoding="utf-8")

    # Clean response pairs; keep original row id so both responses stay in one split.
    df = df[REQUIRED].copy()
    df["source_id"] = np.arange(len(df), dtype=int)
    for c in REQUIRED:
        df[c] = df[c].fillna("").astype(str).str.strip()
    before = len(df)
    df = df[(df["right_response"] != "") & (df["hallucinated_response"] != "")]
    df = df[df["right_response"].str.casefold() != df["hallucinated_response"].str.casefold()]
    df = df.drop_duplicates(subset=REQUIRED, keep="first").reset_index(drop=True)
    cleaning_report = {
        "rows_after_cleaning": int(len(df)),
        "rows_removed_total": int(before-len(df)),
        "note": "Rows with missing/empty response pairs, identical paired responses, and exact duplicate records were removed. Missing context is retained as an empty string."
    }
    (root/"results/cleaning_report.json").write_text(json.dumps(cleaning_report, indent=2), encoding="utf-8")
    if len(df) < 20:
        raise SystemExit("ERROR: Too few valid records after cleaning.")

    # Split original records BEFORE expanding paired responses.
    trainval, test = train_test_split(df, test_size=0.15, random_state=args.seed, shuffle=True)
    train, val = train_test_split(trainval, test_size=(0.15/0.85), random_state=args.seed, shuffle=True)
    def expand(source):
        records = []
        for row in source.itertuples(index=False):
            records.append({"source_id": int(row.source_id), "knowledge": row.knowledge,
                            "dialogue_history": row.dialogue_history, "response": row.right_response, "label": 0})
            records.append({"source_id": int(row.source_id), "knowledge": row.knowledge,
                            "dialogue_history": row.dialogue_history, "response": row.hallucinated_response, "label": 1})
        return pd.DataFrame(records)
    splits = {"train": expand(train), "validation": expand(val), "test": expand(test)}
    ids = {k: set(v["source_id"]) for k,v in splits.items()}
    overlaps = {
        "train_validation": len(ids["train"] & ids["validation"]),
        "train_test": len(ids["train"] & ids["test"]),
        "validation_test": len(ids["validation"] & ids["test"])
    }
    if any(overlaps.values()):
        raise SystemExit(f"ERROR: Source-row leakage detected: {overlaps}")
    (root/"results/split_audit.json").write_text(json.dumps({
        "source_rows": {k: int(len(v)) for k,v in [("train",train),("validation",val),("test",test)]},
        "expanded_examples": {k: int(len(v)) for k,v in splits.items()},
        "overlap_counts": overlaps
    }, indent=2), encoding="utf-8")
    for name, part in splits.items():
        part.to_csv(root/f"processed_data/{name}.csv", index=False)
    pd.concat(list(splits.values()), ignore_index=True).to_csv(root/"processed_data/all_labelled_examples.csv", index=False)

    feature_sets = {
        "response_only": lambda d: d["response"].fillna("").astype(str),
        "context_plus_response": lambda d: (
            "KNOWLEDGE: " + d["knowledge"].fillna("").astype(str) +
            "\nDIALOGUE: " + d["dialogue_history"].fillna("").astype(str) +
            "\nRESPONSE: " + d["response"].fillna("").astype(str)
        )
    }
    models = {
        "logistic_regression": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=args.seed),
        "linear_svc": LinearSVC(class_weight="balanced", random_state=args.seed)
    }
    summaries = []
    saved_pipelines = {}
    for feature_name, text_builder in feature_sets.items():
        x_train = text_builder(splits["train"])
        x_val = text_builder(splits["validation"])
        x_test = text_builder(splits["test"])
        y_train = splits["train"]["label"].astype(int)
        y_val = splits["validation"]["label"].astype(int)
        y_test = splits["test"]["label"].astype(int)
        for model_name, estimator in models.items():
            pipe = Pipeline([
                ("tfidf", TfidfVectorizer(ngram_range=(1,2), min_df=2, max_features=150000, sublinear_tf=True)),
                ("classifier", estimator)
            ])
            pipe.fit(x_train, y_train)
            val_pred = pipe.predict(x_val)
            val_p, val_r, val_f1, _ = precision_recall_fscore_support(y_val, val_pred, average="binary", zero_division=0)
            test_pred = pipe.predict(x_test)
            test_p, test_r, test_f1, _ = precision_recall_fscore_support(y_test, test_pred, average="binary", zero_division=0)
            # Paired accuracy: for each source row, did hallucinated response score higher than right response?
            # Available only if model supports a decision score; score both pair members from test set.
            test_rows = splits["test"]
            pair_correct = []
            if hasattr(pipe, "decision_function"):
                scores = pipe.decision_function(x_test)
                scored = test_rows[["source_id","label"]].copy()
                scored["score"] = scores
                for _, pair in scored.groupby("source_id"):
                    if len(pair) == 2 and set(pair["label"]) == {0,1}:
                        s0 = pair.loc[pair.label == 0, "score"].iloc[0]
                        s1 = pair.loc[pair.label == 1, "score"].iloc[0]
                        pair_correct.append(bool(s1 > s0))
            report = {
                "feature_set": feature_name, "model": model_name,
                "validation_accuracy": float(accuracy_score(y_val, val_pred)),
                "validation_precision": float(val_p), "validation_recall": float(val_r), "validation_f1": float(val_f1),
                "test_accuracy": float(accuracy_score(y_test, test_pred)),
                "test_precision": float(test_p), "test_recall": float(test_r), "test_f1": float(test_f1),
                "paired_accuracy": float(np.mean(pair_correct)) if pair_correct else None,
                "test_classification_report": classification_report(y_test, test_pred, output_dict=True, zero_division=0),
                "test_confusion_matrix": confusion_matrix(y_test, test_pred, labels=[0,1]).tolist()
            }
            summaries.append(report)
            cm = confusion_matrix(y_test, test_pred, labels=[0, 1])

            plt.figure(figsize=(7, 5))

            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=["Non-Hallucinated", "Hallucinated"],
                yticklabels=["Non-Hallucinated", "Hallucinated"]
)

            plt.xlabel("Predicted Label")
            plt.ylabel("Actual Label")
            plt.title(f"Confusion Matrix: {feature_name} + {model_name}")
            plt.tight_layout()

            plt.savefig(
                root / "results" / f"{key}_confusion_matrix.png",
                dpi=300,
                bbox_inches="tight"
            )

            plt.close()
            key = f"{feature_name}__{model_name}"
            saved_pipelines[key] = pipe
            joblib.dump(pipe, root/f"models/{key}.joblib")
            # Save predictions for error analysis.
            pred_df = test_rows.copy()
            pred_df["predicted_label"] = test_pred
            pred_df.to_csv(root/f"results/{key}_test_predictions.csv", index=False)
            (root/f"results/{key}_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    results_df = pd.DataFrame([{
        k:v for k,v in r.items() if k not in ("test_classification_report","test_confusion_matrix")
    } for r in summaries])
    results_df.to_csv(root/"results/model_comparison.csv", index=False)
    best = max(summaries, key=lambda r: (r["validation_f1"], r["validation_accuracy"]))
    (root/"results/metrics.json").write_text(json.dumps({
        "best_model_by_validation_f1": f'{best["feature_set"]}__{best["model"]}',
        "selection_basis": "Validation F1; test metrics are reported for comparison and were not used for model selection.",
        "models": summaries,
        "caution": "These metrics measure performance on the dataset's supplied hallucination labels, not general real-world misinformation detection."
    }, indent=2), encoding="utf-8")
    logging.info("Training complete. Best by validation F1: %s__%s", best["feature_set"], best["model"])
    logging.info("Model comparison saved to results/model_comparison.csv")
    print(results_df.to_string(index=False))
    print("\nSplit audit:", json.dumps(overlaps))
    print("\nBest model by validation F1:", f'{best["feature_set"]}__{best["model"]}')
    print("Important: these are dataset-label results, not proof of general misinformation detection.")

if __name__ == "__main__":
    main()
