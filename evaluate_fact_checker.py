"""
evaluate_fact_checker.py — Benchmark evaluation of the evidence-based NLI pipeline.

This script evaluates the fact-checking system against a manually curated,
labelled benchmark of 25 claims with known expected verdicts.

Metrics reported
----------------
    Accuracy             — correct verdicts / total claims
    Precision (per class) — correct positive predictions / all predicted positive
    Recall (per class)   — correct positive predictions / all actual positive
    F1-score (per class) — harmonic mean of precision and recall
    Macro F1             — unweighted mean F1 across classes
    Coverage             — fraction of claims that received a decisive verdict
                           (SUPPORTED or CONTRADICTED, not INSUFFICIENT EVIDENCE)
    Decisive accuracy    — accuracy among decisive predictions only
    Confusion matrix

Important cautions
------------------
  1. The benchmark labels are manually assigned based on widely-accepted knowledge.
     They are NOT ground truth for all possible NLI model configurations.

  2. The benchmark is small (25 claims).  Results on small benchmarks have high
     variance and should not be treated as definitive accuracy figures.

  3. Thresholds were NOT tuned on this benchmark.  The benchmark is used for
     evaluation only, not threshold selection.

  4. The NLI verdict depends on what Wikipedia returns at runtime.  Results may
     vary slightly across runs due to Wikipedia content changes.

  5. INSUFFICIENT EVIDENCE verdicts are a legitimate outcome, not always an
     error.  Coverage measures how often the system is decisive.

Usage
-----
    # From the hallucination_detection/ directory:
    python evaluate_fact_checker.py

    # With custom thresholds:
    python evaluate_fact_checker.py --nli-threshold 0.70 --min-tfidf 0.05

    # With verbose per-claim output:
    python evaluate_fact_checker.py --verbose
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Benchmark dataset
# ---------------------------------------------------------------------------
#
# Each entry:
#   claim       : the natural-language claim to verify
#   expected    : "SUPPORTED", "CONTRADICTED", or "INSUFFICIENT EVIDENCE"
#   rationale   : brief justification for the expected verdict
#   category    : thematic group for analysis
#
# Important: expected labels reflect widely-accepted scientific/historical
# consensus, not the output of this system.  They were assigned before any
# model run and were NOT adjusted to match model outputs.
#
BENCHMARK: list[dict[str, str]] = [
    # ── Well-established scientific facts ──────────────────────────────────
    {
        "claim": "The Sun is a star.",
        "expected": "SUPPORTED",
        "rationale": "Universally accepted astronomy; Wikipedia describes the Sun as a star.",
        "category": "astronomy",
    },
    {
        "claim": "The Earth orbits the Sun.",
        "expected": "SUPPORTED",
        "rationale": "Heliocentric model; well documented on Wikipedia.",
        "category": "astronomy",
    },
    {
        "claim": "Water is composed of hydrogen and oxygen.",
        "expected": "SUPPORTED",
        "rationale": "H2O; basic chemistry; Wikipedia covers this explicitly.",
        "category": "chemistry",
    },
    {
        "claim": "DNA carries genetic information.",
        "expected": "SUPPORTED",
        "rationale": "Molecular biology consensus; Wikipedia covers DNA extensively.",
        "category": "biology",
    },
    {
        "claim": "The speed of light in a vacuum is approximately 300,000 km/s.",
        "expected": "SUPPORTED",
        "rationale": "Physical constant; well documented.",
        "category": "physics",
    },
    {
        "claim": "Humans and chimpanzees share a common ancestor.",
        "expected": "SUPPORTED",
        "rationale": "Evolutionary biology consensus.",
        "category": "biology",
    },
    {
        "claim": "The human body has 206 bones.",
        "expected": "SUPPORTED",
        "rationale": "Standard anatomy; widely cited on Wikipedia.",
        "category": "biology",
    },
    # ── False / contradicted claims ────────────────────────────────────────
    {
        "claim": "The Sun revolves around the Earth.",
        "expected": "CONTRADICTED",
        "rationale": "Geocentric claim contradicted by heliocentric model.",
        "category": "astronomy",
    },
    {
        "claim": "Albert Einstein was born in France.",
        "expected": "CONTRADICTED",
        "rationale": "Einstein was born in Ulm, Germany; Wikipedia states this clearly.",
        "category": "history",
    },
    {
        "claim": "The Moon is a planet.",
        "expected": "CONTRADICTED",
        "rationale": "The Moon is Earth's natural satellite, not a planet.",
        "category": "astronomy",
    },
    {
        "claim": "Alexander Graham Bell invented the telephone.",
        "expected": "SUPPORTED",
        "rationale": "Historically credited with the telephone; Wikipedia covers this.",
        "category": "history",
    },
    {
        "claim": "William Shakespeare was born in Italy.",
        "expected": "CONTRADICTED",
        "rationale": "Shakespeare was born in Stratford-upon-Avon, England.",
        "category": "history",
    },
    {
        "claim": "The Great Wall of China is visible from space with the naked eye.",
        "expected": "CONTRADICTED",
        "rationale": "This is a common myth debunked by astronauts; Wikipedia notes this.",
        "category": "geography",
    },
    {
        "claim": "Antibiotics are effective against viral infections.",
        "expected": "CONTRADICTED",
        "rationale": "Antibiotics target bacteria, not viruses; Wikipedia explicitly states this.",
        "category": "medicine",
    },
    {
        "claim": "The Eiffel Tower is located in Paris.",
        "expected": "SUPPORTED",
        "rationale": "Universally known; Wikipedia confirms the location.",
        "category": "geography",
    },
    # ── Likely INSUFFICIENT EVIDENCE ──────────────────────────────────────
    {
        "claim": "A particular person's favourite colour is blue.",
        "expected": "INSUFFICIENT EVIDENCE",
        "rationale": "Personal preference with no Wikipedia source.",
        "category": "personal",
    },
    {
        "claim": "The best football team in the world is Brazil.",
        "expected": "INSUFFICIENT EVIDENCE",
        "rationale": "Subjective; no objective Wikipedia verdict.",
        "category": "sports",
    },
    {
        "claim": "My neighbour owns three cats.",
        "expected": "INSUFFICIENT EVIDENCE",
        "rationale": "Private personal information; no Wikipedia coverage.",
        "category": "personal",
    },
    # ── Additional positive cases ──────────────────────────────────────────
    {
        "claim": "Photosynthesis is the process by which plants convert sunlight into food.",
        "expected": "SUPPORTED",
        "rationale": "Basic biology; Wikipedia covers photosynthesis.",
        "category": "biology",
    },
    {
        "claim": "The Pythagorean theorem states that a squared plus b squared equals c squared.",
        "expected": "SUPPORTED",
        "rationale": "Fundamental geometry; Wikipedia covers it explicitly.",
        "category": "mathematics",
    },
    {
        "claim": "The Pacific Ocean is the largest ocean on Earth.",
        "expected": "SUPPORTED",
        "rationale": "Geography fact; Wikipedia confirms this.",
        "category": "geography",
    },
    # ── Additional negative cases ──────────────────────────────────────────
    {
        "claim": "Humans only use 10 percent of their brain.",
        "expected": "CONTRADICTED",
        "rationale": "Common myth; Wikipedia explicitly identifies this as a myth.",
        "category": "neuroscience",
    },
    {
        "claim": "Lightning never strikes the same place twice.",
        "expected": "CONTRADICTED",
        "rationale": "Common myth; lightning does strike the same place more than once.",
        "category": "meteorology",
    },
    {
        "claim": "Napoleon Bonaparte was unusually short for his time.",
        "expected": "CONTRADICTED",
        "rationale": "Napoleon was average height for his era; the myth is documented on Wikipedia.",
        "category": "history",
    },
    {
        "claim": "Diamonds are made of carbon.",
        "expected": "SUPPORTED",
        "rationale": "Basic chemistry; Wikipedia covers diamonds as carbon allotropes.",
        "category": "chemistry",
    },
]


# ---------------------------------------------------------------------------
# Evaluation runner
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkResult:
    claim: str
    expected: str
    predicted: str
    correct: bool
    decisive: bool
    confidence: float
    elapsed: float
    explanation: str
    sources: list[str]
    error: str | None = None


def run_benchmark(
    nli_threshold: float = 0.60,
    min_tfidf: float = 0.05,
    top_k: int = 5,
    max_pages: int = 3,
    verbose: bool = False,
) -> list[BenchmarkResult]:
    """Run all benchmark claims through the fact-checking pipeline."""
    import fact_checker as fc

    print(f"\nLoading NLI model (this may take a moment on first run)…")
    try:
        fc.load_nli_model()
    except RuntimeError as exc:
        print(f"ERROR: Cannot load NLI model: {exc}")
        sys.exit(1)

    results: list[BenchmarkResult] = []
    total = len(BENCHMARK)

    for i, entry in enumerate(BENCHMARK, 1):
        claim = entry["claim"]
        expected = entry["expected"]
        category = entry["category"]

        print(f"  [{i:2d}/{total}] {category:>14} | {claim[:65]:<65}", end="", flush=True)
        t0 = time.time()

        try:
            result = fc.check_claim(
                claim,
                nli_threshold=nli_threshold,
                min_tfidf=min_tfidf,
                top_k=top_k,
                max_pages=max_pages,
            )
            elapsed = time.time() - t0
            predicted = result.verdict
            decisive = predicted in ("SUPPORTED", "CONTRADICTED")
            correct = predicted == expected

            br = BenchmarkResult(
                claim=claim,
                expected=expected,
                predicted=predicted,
                correct=correct,
                decisive=decisive,
                confidence=result.confidence,
                elapsed=elapsed,
                explanation=result.explanation,
                sources=[s["title"] for s in result.sources],
            )

            status = "✓" if correct else "✗"
            print(f" → {predicted:<24} {status}  ({elapsed:.1f}s)")

            if verbose:
                print(f"         Expected  : {expected}")
                print(f"         Explain   : {result.explanation[:120]}")
                if result.sources:
                    print(f"         Sources   : {', '.join(s['title'] for s in result.sources)}")
                print()

        except Exception as exc:
            elapsed = time.time() - t0
            print(f" → ERROR: {exc}  ({elapsed:.1f}s)")
            br = BenchmarkResult(
                claim=claim,
                expected=expected,
                predicted="ERROR",
                correct=False,
                decisive=False,
                confidence=0.0,
                elapsed=elapsed,
                explanation="",
                sources=[],
                error=str(exc),
            )

        results.append(br)

    return results


# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------

def compute_metrics(results: list[BenchmarkResult]) -> dict[str, Any]:
    """
    Compute accuracy, per-class precision/recall/F1, macro F1, coverage,
    decisive accuracy, and confusion matrix.
    """
    labels = ["SUPPORTED", "CONTRADICTED", "INSUFFICIENT EVIDENCE"]
    n = len(results)

    # ── Accuracy ──────────────────────────────────────────────────────────
    correct = sum(1 for r in results if r.correct and r.error is None)
    errors = sum(1 for r in results if r.error is not None)
    accuracy = correct / n if n > 0 else 0.0

    # ── Coverage (decisive = SUPPORTED or CONTRADICTED) ──────────────────
    decisive_results = [r for r in results if r.decisive]
    coverage = len(decisive_results) / n if n > 0 else 0.0

    decisive_correct = sum(1 for r in decisive_results if r.correct)
    decisive_accuracy = decisive_correct / len(decisive_results) if decisive_results else 0.0

    # ── Per-class P / R / F1 ─────────────────────────────────────────────
    per_class: dict[str, dict] = {}
    for label in labels:
        tp = sum(1 for r in results if r.predicted == label and r.expected == label)
        fp = sum(1 for r in results if r.predicted == label and r.expected != label)
        fn = sum(1 for r in results if r.expected == label and r.predicted != label)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        per_class[label] = {"tp": tp, "fp": fp, "fn": fn,
                             "precision": precision, "recall": recall, "f1": f1}

    macro_f1 = sum(v["f1"] for v in per_class.values()) / len(labels)

    # ── Confusion matrix ─────────────────────────────────────────────────
    matrix: dict[str, dict[str, int]] = {exp: {pred: 0 for pred in labels + ["ERROR"]} for exp in labels}
    for r in results:
        exp = r.expected
        pred = r.predicted if r.predicted in labels else "ERROR"
        if exp in matrix:
            matrix[exp][pred] += 1

    return {
        "total_claims": n,
        "errors": errors,
        "accuracy": accuracy,
        "coverage": coverage,
        "decisive_accuracy": decisive_accuracy,
        "macro_f1": macro_f1,
        "per_class": per_class,
        "confusion_matrix": matrix,
    }


# ---------------------------------------------------------------------------
# Report printing
# ---------------------------------------------------------------------------

def print_report(metrics: dict, results: list[BenchmarkResult], args) -> None:
    line = "=" * 70

    print(f"\n{line}")
    print("  FACT-CHECKER EVALUATION REPORT")
    print(line)
    print(f"  NLI threshold  : {args.nli_threshold:.2f}")
    print(f"  TF-IDF min sim : {args.min_tfidf:.2f}")
    print(f"  Max Wiki pages : {args.max_pages}")
    print(f"  Sentences/page : {args.top_k}")
    print(f"  Benchmark size : {metrics['total_claims']} claims")
    print(f"  Errors during run: {metrics['errors']}")
    print(line)

    print(f"\n  Overall Accuracy : {metrics['accuracy']:.1%}  ({int(metrics['accuracy'] * metrics['total_claims'])}/{metrics['total_claims']})")
    print(f"  Coverage         : {metrics['coverage']:.1%}  (fraction of claims with a decisive verdict)")
    print(f"  Decisive Acc.    : {metrics['decisive_accuracy']:.1%}  (accuracy among SUPPORTED/CONTRADICTED predictions)")
    print(f"  Macro F1         : {metrics['macro_f1']:.4f}")

    print(f"\n  Per-Class Metrics:")
    print(f"  {'Label':<28} {'P':>6} {'R':>6} {'F1':>6} {'TP':>4} {'FP':>4} {'FN':>4}")
    print(f"  {'-'*28} {'-'*6} {'-'*6} {'-'*6} {'-'*4} {'-'*4} {'-'*4}")
    for label, v in metrics["per_class"].items():
        print(
            f"  {label:<28} {v['precision']:>6.3f} {v['recall']:>6.3f} {v['f1']:>6.3f} "
            f"{v['tp']:>4d} {v['fp']:>4d} {v['fn']:>4d}"
        )

    print(f"\n  Confusion Matrix (rows = expected, cols = predicted):")
    col_labels = ["SUPPORTED", "CONTRADICTED", "INSUFF.", "ERROR"]
    print(f"  {'Expected \\ Predicted':<30} {'SUPPORTED':>10} {'CONTRADICTED':>14} {'INSUFF.':>9} {'ERROR':>7}")
    print(f"  {'-'*70}")
    for exp_label, row in metrics["confusion_matrix"].items():
        s = row.get("SUPPORTED", 0)
        c = row.get("CONTRADICTED", 0)
        i = row.get("INSUFFICIENT EVIDENCE", 0)
        e = row.get("ERROR", 0)
        print(f"  {exp_label:<30} {s:>10d} {c:>14d} {i:>9d} {e:>7d}")

    print(f"\n{line}")
    print("  IMPORTANT CAVEATS")
    print(line)
    print("""
  1. Benchmark labels are manually assigned — not automatically generated or
     validated against multiple sources.

  2. Benchmark size is small (25 claims). Results have high variance and should
     not be interpreted as a precise characterisation of the system's accuracy.

  3. Thresholds were NOT selected based on this benchmark. Using these results
     to tune thresholds and then re-reporting on the same benchmark would be
     methodologically invalid (data leakage).

  4. INSUFFICIENT EVIDENCE is NOT always a wrong answer. When evidence is
     genuinely ambiguous or absent, it is the correct conservative verdict.

  5. Wikipedia content changes over time. Results may vary across runs.

  6. This system is a research prototype. Do not deploy it as a reliable
     fact-checker in production or high-stakes contexts.
""")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(
        description="Evaluate the evidence-based fact checker on a curated benchmark."
    )
    parser.add_argument("--nli-threshold", type=float, default=0.60,
                        help="NLI confidence threshold (default: 0.60)")
    parser.add_argument("--min-tfidf", type=float, default=0.05,
                        help="Minimum TF-IDF similarity (default: 0.05)")
    parser.add_argument("--top-k", type=int, default=5,
                        help="Sentences per Wikipedia page (default: 5)")
    parser.add_argument("--max-pages", type=int, default=3,
                        help="Wikipedia pages per claim (default: 3)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print per-claim explanation and sources")
    parser.add_argument("--output-json", type=str, default=None,
                        help="If specified, write full results to this JSON file")
    args = parser.parse_args()

    print(f"\n{'='*70}")
    print("  EVIDENCE-BASED FACT CHECKER — BENCHMARK EVALUATION")
    print(f"{'='*70}")
    print(f"  Running {len(BENCHMARK)} claims. This will make Wikipedia API calls")
    print("  and NLI model inferences. Estimated time: 2–8 minutes.")
    print(f"{'='*70}\n")

    results = run_benchmark(
        nli_threshold=args.nli_threshold,
        min_tfidf=args.min_tfidf,
        top_k=args.top_k,
        max_pages=args.max_pages,
        verbose=args.verbose,
    )

    metrics = compute_metrics(results)
    print_report(metrics, results, args)

    # Save to JSON if requested
    if args.output_json:
        out_path = Path(args.output_json)
        payload = {
            "parameters": {
                "nli_threshold": args.nli_threshold,
                "min_tfidf": args.min_tfidf,
                "top_k": args.top_k,
                "max_pages": args.max_pages,
            },
            "metrics": metrics,
            "results": [
                {
                    "claim": r.claim,
                    "expected": r.expected,
                    "predicted": r.predicted,
                    "correct": r.correct,
                    "decisive": r.decisive,
                    "confidence": r.confidence,
                    "elapsed_s": round(r.elapsed, 2),
                    "sources": r.sources,
                    "explanation": r.explanation,
                    "error": r.error,
                }
                for r in results
            ],
            "caution": (
                "These results reflect the system's performance on a small manually-curated benchmark. "
                "They are NOT a general measure of fact-checking accuracy."
            ),
        }
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\n  Full results written to: {out_path.resolve()}")


if __name__ == "__main__":
    main()
