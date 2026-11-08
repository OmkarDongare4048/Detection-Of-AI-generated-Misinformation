"""
Evidence-based fact checker using Wikipedia retrieval and Natural Language Inference.

Architecture:
    Step A  — Validate claim input.
    Step B  — Retrieve evidence from Wikipedia (multiple pages).
    Step C  — Process and rank evidence sentences by TF-IDF similarity.
    Step D  — Run NLI model over the top-ranked sentences (full score distribution).
    Step E  — Aggregate NLI scores conservatively; detect conflicting evidence.
    Step F  — Return a structured result with provenance.

Thresholds and limits are exposed as module-level constants so that callers
and evaluators can adjust them without modifying logic:

    MIN_TFIDF_SIMILARITY   — Minimum cosine similarity (TF-IDF) for a sentence
                             to qualify as candidate evidence.  Sentences scoring
                             below this are discarded before NLI.

    NLI_CONFIDENCE_THRESHOLD — The NLI score for a decisive label (entailment or
                               contradiction) must meet this threshold for the
                               sentence to count as decisive evidence.  Set
                               conservatively to reduce false positives.

    TOP_K_SENTENCES        — Number of highest-ranked sentences sent to the NLI
                             model per Wikipedia page.

    MAX_WIKI_PAGES         — Number of Wikipedia search results to fetch and
                             evaluate.  Higher values increase coverage at the
                             cost of latency.

    WIKI_SEARCH_LIMIT      — Number of candidate article titles returned by the
                             Wikipedia search API before relevance re-ranking.

All thresholds are experimental parameters, not universal guarantees.  They
should be evaluated on a held-out benchmark before being changed.

Important limitations:
    - This system does not guarantee factual accuracy.
    - Wikipedia may not contain reliable information for every claim.
    - NLI models trained on general datasets may mis-classify domain-specific text.
    - A SUPPORTED verdict means the retrieved evidence appears to entail the
      claim; it does not mean the claim is objectively true.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configurable thresholds (treat as experimental parameters, not hard truths)
# ---------------------------------------------------------------------------

MIN_TFIDF_SIMILARITY: float = 0.05
"""Minimum TF-IDF cosine similarity for a sentence to qualify as evidence."""

NLI_CONFIDENCE_THRESHOLD: float = 0.60
"""
Minimum NLI score for a decisive label (entailment/contradiction) to count.
Lowering this increases recall but also increases false verdicts.
"""

TOP_K_SENTENCES: int = 5
"""Number of top-ranked sentences to evaluate per Wikipedia page."""

MAX_WIKI_PAGES: int = 3
"""Number of Wikipedia pages to retrieve and evaluate per claim."""

WIKI_SEARCH_LIMIT: int = 8
"""Number of candidate article titles fetched from the Wikipedia search API."""

WIKI_USER_AGENT: str = "HallucinationDetector/2.0 (mtech educational project; contact: student)"
"""User-Agent header sent to Wikipedia APIs (per Wikimedia API etiquette)."""

WIKI_API_URL: str = "https://en.wikipedia.org/w/api.php"
WIKI_REQUEST_TIMEOUT: int = 15  # seconds
MAX_EXTRACT_CHARS: int = 4000   # characters of page extract to keep per page

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class EvidenceSentence:
    """One ranked candidate sentence with its NLI scores and provenance."""

    text: str
    source_title: str
    source_url: str
    tfidf_similarity: float
    # NLI scores (all three labels); populated after NLI inference
    entailment_score: float = 0.0
    contradiction_score: float = 0.0
    neutral_score: float = 0.0
    decisive_label: str = "neutral"   # 'entailment', 'contradiction', 'neutral'
    decisive_score: float = 0.0


@dataclass
class WikiPage:
    """A Wikipedia page extract with metadata."""

    title: str
    text: str
    url: str


@dataclass
class FactCheckResult:
    """Structured output of the fact-checking pipeline."""

    claim: str
    verdict: str                            # SUPPORTED | CONTRADICTED | INSUFFICIENT EVIDENCE
    nli_label: str                          # dominant NLI label
    confidence: float                       # score of the decisive sentence
    matched_evidence: list[EvidenceSentence] = field(default_factory=list)
    sources: list[dict[str, str]] = field(default_factory=list)
    explanation: str = ""
    warnings: list[str] = field(default_factory=list)
    conflicting_evidence: bool = False
    all_scored_sentences: list[EvidenceSentence] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict (for JSON output and Streamlit display)."""
        return {
            "claim": self.claim,
            "verdict": self.verdict,
            "nli_label": self.nli_label,
            "confidence": self.confidence,
            "matched_evidence": [
                {
                    "text": s.text,
                    "source_title": s.source_title,
                    "source_url": s.source_url,
                    "tfidf_similarity": round(s.tfidf_similarity, 4),
                    "entailment_score": round(s.entailment_score, 4),
                    "contradiction_score": round(s.contradiction_score, 4),
                    "neutral_score": round(s.neutral_score, 4),
                    "decisive_label": s.decisive_label,
                    "decisive_score": round(s.decisive_score, 4),
                }
                for s in self.matched_evidence
            ],
            "sources": self.sources,
            "explanation": self.explanation,
            "warnings": self.warnings,
            "conflicting_evidence": self.conflicting_evidence,
        }


# ---------------------------------------------------------------------------
# NLI model — loaded once per process
# ---------------------------------------------------------------------------

_verifier = None           # cached pipeline
_label_map: dict[str, str] = {}   # id → label name (populated on load)


def load_nli_model(model_name: str = "cross-encoder/nli-deberta-v3-small") -> None:
    """
    Load the NLI model into the module-level cache.

    Called automatically the first time verify_claim() is invoked, or can be
    called explicitly at application start-up to front-load the latency.

    Parameters
    ----------
    model_name:
        Hugging Face model identifier.  Must be a sequence-pair classification
        model whose labels include 'entailment', 'contradiction', and 'neutral'.
    """
    global _verifier, _label_map

    if _verifier is not None:
        return  # already loaded

    try:
        from transformers import pipeline as hf_pipeline

        logger.info("Loading NLI model: %s", model_name)
        _verifier = hf_pipeline(
            "text-classification",
            model=model_name,
            truncation=True,
            top_k=None,   # return ALL label scores, not just the highest one
        )

        # Read label mapping directly from model config to avoid hard-coding.
        raw_labels = _verifier.model.config.id2label
        _label_map = {str(k): v.lower() for k, v in raw_labels.items()}
        logger.info("NLI model labels: %s", _label_map)

    except Exception as exc:
        raise RuntimeError(
            f"Failed to load NLI model '{model_name}'. "
            "Ensure 'transformers' and 'torch' are installed and the model "
            f"is accessible.\nOriginal error: {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# Step B — Evidence retrieval
# ---------------------------------------------------------------------------

def _search_wikipedia(claim: str) -> list[str]:
    """
    Query the Wikipedia search API and return a list of candidate page titles.

    Returns an empty list on network or parse errors.
    """
    try:
        params = {
            "action": "query",
            "list": "search",
            "srsearch": claim,
            "srlimit": WIKI_SEARCH_LIMIT,
            "format": "json",
        }
        response = requests.get(
            WIKI_API_URL,
            params=params,
            headers={"User-Agent": WIKI_USER_AGENT},
            timeout=WIKI_REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        results = response.json()["query"]["search"]
        return [r["title"] for r in results]

    except requests.RequestException as exc:
        logger.warning("Wikipedia search network error: %s", exc)
        return []
    except (KeyError, ValueError) as exc:
        logger.warning("Wikipedia search response parse error: %s", exc)
        return []
    except Exception as exc:
        logger.warning("Wikipedia search unexpected error: %s", exc)
        return []


def _fetch_page(title: str) -> WikiPage | None:
    """
    Fetch the introductory extract and URL of a Wikipedia page by title.

    Returns None if the page is missing, has no extract, or a network error
    occurs.
    """
    try:
        params = {
            "action": "query",
            "prop": "extracts|info",
            "exintro": True,
            "explaintext": True,
            "inprop": "url",
            "titles": title,
            "format": "json",
        }
        response = requests.get(
            WIKI_API_URL,
            params=params,
            headers={"User-Agent": WIKI_USER_AGENT},
            timeout=WIKI_REQUEST_TIMEOUT,
        )
        response.raise_for_status()

        pages = response.json()["query"]["pages"]
        for page_data in pages.values():
            # page id -1 means the page was not found
            if str(page_data.get("pageid", -1)) == "-1":
                logger.debug("Wikipedia page not found: %s", title)
                return None

            extract = page_data.get("extract", "").strip()
            if not extract:
                logger.debug("Wikipedia page has no extract: %s", title)
                return None

            url = page_data.get("fullurl", "")
            return WikiPage(
                title=page_data.get("title", title),
                text=extract[:MAX_EXTRACT_CHARS],
                url=url,
            )

    except requests.RequestException as exc:
        logger.warning("Wikipedia page fetch network error for '%s': %s", title, exc)
        return None
    except (KeyError, ValueError) as exc:
        logger.warning("Wikipedia page parse error for '%s': %s", title, exc)
        return None
    except Exception as exc:
        logger.warning("Wikipedia page unexpected error for '%s': %s", title, exc)
        return None

    return None


def retrieve_evidence(claim: str, max_pages: int = MAX_WIKI_PAGES) -> list[WikiPage]:
    """
    Search Wikipedia and retrieve up to *max_pages* relevant page extracts.

    Each page is fetched separately so that network errors on one page do not
    prevent retrieval of others.  Duplicate page titles are skipped.

    Parameters
    ----------
    claim:
        The natural-language claim to search for.
    max_pages:
        Maximum number of pages to return.

    Returns
    -------
    list[WikiPage]
        Retrieved pages, in the order returned by the search API.  May be
        empty if no usable pages were found.
    """
    titles = _search_wikipedia(claim)
    if not titles:
        logger.info("No Wikipedia search results for claim: %s", claim)
        return []

    pages: list[WikiPage] = []
    seen_titles: set[str] = set()

    for title in titles:
        if len(pages) >= max_pages:
            break
        if title in seen_titles:
            continue
        seen_titles.add(title)

        page = _fetch_page(title)
        if page is not None:
            pages.append(page)

    return pages


# ---------------------------------------------------------------------------
# Step C — Evidence processing and ranking
# ---------------------------------------------------------------------------

def _split_into_sentences(text: str) -> list[str]:
    """Split a passage into sentences using punctuation-based heuristics."""
    raw = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in raw if s.strip() and len(s.strip()) > 15]


def rank_sentences(
    claim: str,
    sentences: list[str],
    min_similarity: float = MIN_TFIDF_SIMILARITY,
    top_k: int = TOP_K_SENTENCES,
) -> list[tuple[str, float]]:
    """
    Rank candidate sentences by TF-IDF cosine similarity to the claim and
    return the top-k sentences that exceed *min_similarity*.

    Returns
    -------
    list[tuple[str, float]]
        (sentence_text, similarity_score) pairs, sorted descending by score.
        Empty if no sentence meets *min_similarity*.

    Notes
    -----
    TF-IDF similarity captures lexical overlap, not semantic relatedness.
    A high similarity score means the sentence shares words with the claim,
    not that it is factually relevant.  NLI inference in Step D handles the
    semantic relationship.
    """
    if not sentences:
        return []

    try:
        vectorizer = TfidfVectorizer(stop_words="english")
        vectors = vectorizer.fit_transform([claim] + sentences)
        sims = cosine_similarity(vectors[0:1], vectors[1:])[0]
    except ValueError:
        # Can happen if all tokens are stop words or the vocabulary is empty.
        logger.debug("TF-IDF vectorisation failed (empty vocabulary); returning empty ranking.")
        return []

    ranked = sorted(
        ((sentences[i], float(sims[i])) for i in range(len(sentences))),
        key=lambda x: x[1],
        reverse=True,
    )
    return [(text, score) for text, score in ranked if score >= min_similarity][:top_k]


# ---------------------------------------------------------------------------
# Step D — NLI inference
# ---------------------------------------------------------------------------

def _parse_nli_output(raw_output: Any) -> dict[str, float]:
    """
    Parse the output of the text-classification pipeline (with top_k=None)
    into a normalised dict mapping label names to scores.

    The pipeline with top_k=None returns a list of dicts:
        [{"label": "LABEL_0", "score": 0.9}, {"label": "LABEL_1", "score": 0.07}, ...]

    We use the model's id2label config (stored in _label_map) to convert
    "LABEL_0" → "contradiction" / "entailment" / "neutral".

    Returns
    -------
    dict[str, float]
        Keys: "entailment", "contradiction", "neutral".
        Values: model scores (0–1, summing to approximately 1).
    """
    scores: dict[str, float] = {"entailment": 0.0, "contradiction": 0.0, "neutral": 0.0}

    if isinstance(raw_output, list) and raw_output and isinstance(raw_output[0], list):
        raw_output = raw_output[0]

    if not isinstance(raw_output, list):
        logger.warning("Unexpected NLI output format: %s", type(raw_output))
        return scores

    for item in raw_output:
        raw_label: str = item.get("label", "")
        score: float = float(item.get("score", 0.0))

        # Try direct match first (model may return "entailment" etc.)
        normalised = raw_label.lower()
        if normalised in scores:
            scores[normalised] = score
            continue

        # Try via id2label map (e.g., "LABEL_0" → "contradiction")
        mapped = _label_map.get(raw_label, _label_map.get(raw_label.replace("LABEL_", ""), ""))
        if mapped in scores:
            scores[mapped] = score
        else:
            logger.debug("Unrecognised NLI label '%s'; ignoring.", raw_label)

    return scores


def run_nli(premise: str, hypothesis: str) -> dict[str, float]:
    """
    Run the NLI model with *premise* as the known evidence and *hypothesis*
    as the claim to be verified.

    Returns
    -------
    dict[str, float]
        {"entailment": ..., "contradiction": ..., "neutral": ...}

    Raises
    ------
    RuntimeError
        If the model has not been loaded and load_nli_model() fails.
    """
    if _verifier is None:
        load_nli_model()

    try:
        # The cross-encoder expects (premise, hypothesis) where:
        #   text      = premise  (what we know — the evidence sentence)
        #   text_pair = hypothesis (what we want to verify — the claim)
        raw = _verifier(
            {"text": premise, "text_pair": hypothesis},
            truncation=True,
        )
        return _parse_nli_output(raw)

    except Exception as exc:
        logger.warning("NLI inference error: %s", exc)
        return {"entailment": 0.0, "contradiction": 0.0, "neutral": 1.0}


# ---------------------------------------------------------------------------
# Step E — Evidence aggregation
# ---------------------------------------------------------------------------

def _aggregate_evidence(
    scored_sentences: list[EvidenceSentence],
    threshold: float = NLI_CONFIDENCE_THRESHOLD,
) -> tuple[str, str, float, bool, str]:
    """
    Aggregate NLI-scored sentences into a final verdict.

    Returns
    -------
    (verdict, nli_label, confidence, conflicting, explanation)

    Design (conservative)
    ----------------------
    - SUPPORTED       : ≥1 sentence with entailment ≥ threshold AND
                        no sentence with contradiction ≥ threshold.
    - CONTRADICTED    : ≥1 sentence with contradiction ≥ threshold AND
                        no sentence with entailment ≥ threshold.
    - CONFLICTING     : Both entailment and contradiction evidence found
                        above threshold → reported as INSUFFICIENT EVIDENCE
                        with a conflict warning.
    - INSUFFICIENT EVIDENCE: No decisive evidence found above threshold.
    """
    entailment_hits = [
        s for s in scored_sentences
        if s.entailment_score >= threshold
    ]
    contradiction_hits = [
        s for s in scored_sentences
        if s.contradiction_score >= threshold
    ]

    conflicting = bool(entailment_hits and contradiction_hits)

    if conflicting:
        best_e = max(entailment_hits, key=lambda s: s.entailment_score)
        best_c = max(contradiction_hits, key=lambda s: s.contradiction_score)
        explanation = (
            f"Conflicting evidence found. "
            f"One source appears to support the claim "
            f"(entailment score {best_e.entailment_score:.2%}, "
            f"source: '{best_e.source_title}') while another appears to "
            f"contradict it (contradiction score "
            f"{best_c.contradiction_score:.2%}, source: '{best_c.source_title}'). "
            f"The system cannot make a reliable determination."
        )
        return "INSUFFICIENT EVIDENCE", "neutral", 0.0, True, explanation

    if entailment_hits:
        best = max(entailment_hits, key=lambda s: s.entailment_score)
        explanation = (
            f"The claim appears to be supported by evidence from "
            f"'{best.source_title}' with an entailment confidence of "
            f"{best.entailment_score:.2%}. Evidence: \"{best.text}\""
        )
        return "SUPPORTED", "entailment", best.entailment_score, False, explanation

    if contradiction_hits:
        best = max(contradiction_hits, key=lambda s: s.contradiction_score)
        explanation = (
            f"The claim appears to be contradicted by evidence from "
            f"'{best.source_title}' with a contradiction confidence of "
            f"{best.contradiction_score:.2%}. Evidence: \"{best.text}\""
        )
        return "CONTRADICTED", "contradiction", best.contradiction_score, False, explanation

    # No decisive evidence
    if scored_sentences:
        best = max(scored_sentences, key=lambda s: max(s.entailment_score, s.contradiction_score, s.neutral_score))
        dominant = best.decisive_label
        explanation = (
            f"Retrieved evidence did not pass the confidence threshold "
            f"({threshold:.0%}) needed for a decisive verdict. The strongest "
            f"signal was '{dominant}' ({best.decisive_score:.2%}) from "
            f"'{best.source_title}'. The claim cannot be reliably verified "
            f"from available Wikipedia evidence."
        )
    else:
        explanation = (
            "No relevant evidence sentences were retrieved from Wikipedia "
            "that are sufficiently similar to the claim. The claim cannot "
            "be verified with the available information."
        )

    return "INSUFFICIENT EVIDENCE", "neutral", 0.0, False, explanation


# ---------------------------------------------------------------------------
# Step A + public entry point
# ---------------------------------------------------------------------------

def check_claim(
    claim: str,
    nli_threshold: float = NLI_CONFIDENCE_THRESHOLD,
    min_tfidf: float = MIN_TFIDF_SIMILARITY,
    top_k: int = TOP_K_SENTENCES,
    max_pages: int = MAX_WIKI_PAGES,
) -> FactCheckResult:
    """
    Run the full evidence-based fact-checking pipeline on a claim.

    Parameters
    ----------
    claim:
        The natural-language factual claim to verify.
    nli_threshold:
        Minimum NLI confidence for a decisive label.
    min_tfidf:
        Minimum TF-IDF similarity to include a sentence as candidate evidence.
    top_k:
        Number of top-ranked sentences to evaluate per Wikipedia page.
    max_pages:
        Number of Wikipedia pages to retrieve.

    Returns
    -------
    FactCheckResult
        Structured result with verdict, evidence, sources, and explanation.

    Raises
    ------
    ValueError
        If *claim* is empty or whitespace-only.
    RuntimeError
        If the NLI model cannot be loaded.
    """
    # --- Step A: Validate input ---
    if not claim or not claim.strip():
        raise ValueError("Claim must be a non-empty string.")

    claim = claim.strip()

    # Ensure NLI model is loaded (happens once per process)
    load_nli_model()

    warnings: list[str] = [
        "This system provides an automated assessment based on Wikipedia evidence. "
        "It does not guarantee factual accuracy and should not replace authoritative sources."
    ]

    # --- Step B: Retrieve evidence ---
    logger.info("Retrieving Wikipedia evidence for: %s", claim)
    pages = retrieve_evidence(claim, max_pages=max_pages)

    if not pages:
        return FactCheckResult(
            claim=claim,
            verdict="INSUFFICIENT EVIDENCE",
            nli_label="neutral",
            confidence=0.0,
            explanation=(
                "No Wikipedia pages could be retrieved for this claim. "
                "This may be due to a network error, an ambiguous claim, or "
                "the absence of a relevant Wikipedia article."
            ),
            warnings=warnings + [
                "Wikipedia search returned no results. "
                "Try rephrasing the claim or check your network connection."
            ],
        )

    sources = [{"title": p.title, "url": p.url} for p in pages]

    # --- Step C + D: Process each page ---
    all_scored: list[EvidenceSentence] = []

    for page in pages:
        sentences = _split_into_sentences(page.text)
        if not sentences:
            continue

        ranked = rank_sentences(claim, sentences, min_similarity=min_tfidf, top_k=top_k)
        if not ranked:
            logger.debug("No sentences above TF-IDF threshold in page: %s", page.title)
            continue

        for sent_text, tfidf_score in ranked:
            nli_scores = run_nli(premise=sent_text, hypothesis=claim)

            dominant_label = max(nli_scores, key=nli_scores.get)
            dominant_score = nli_scores[dominant_label]

            ev = EvidenceSentence(
                text=sent_text,
                source_title=page.title,
                source_url=page.url,
                tfidf_similarity=tfidf_score,
                entailment_score=nli_scores["entailment"],
                contradiction_score=nli_scores["contradiction"],
                neutral_score=nli_scores["neutral"],
                decisive_label=dominant_label,
                decisive_score=dominant_score,
            )
            all_scored.append(ev)

    if not all_scored:
        return FactCheckResult(
            claim=claim,
            verdict="INSUFFICIENT EVIDENCE",
            nli_label="neutral",
            confidence=0.0,
            sources=sources,
            explanation=(
                f"Wikipedia pages were retrieved ({len(pages)} page(s): "
                f"{', '.join(p.title for p in pages)}), but no sentences "
                f"scored above the minimum TF-IDF similarity threshold "
                f"({min_tfidf}) for the claim. The pages may not directly "
                f"address the specific claim being made."
            ),
            warnings=warnings,
        )

    # --- Step E: Aggregate ---
    verdict, nli_label, confidence, conflicting, explanation = _aggregate_evidence(
        all_scored, threshold=nli_threshold
    )

    if conflicting:
        warnings.append(
            "Conflicting evidence detected: some retrieved sentences support "
            "the claim while others contradict it. The verdict has been set "
            "to INSUFFICIENT EVIDENCE to reflect this ambiguity."
        )

    # Collect matched evidence (decisive sentences used in the verdict)
    if verdict == "SUPPORTED":
        matched = [s for s in all_scored if s.entailment_score >= nli_threshold]
    elif verdict == "CONTRADICTED":
        matched = [s for s in all_scored if s.contradiction_score >= nli_threshold]
    else:
        # Return the top-3 sentences by maximum decisive score for transparency
        matched = sorted(
            all_scored,
            key=lambda s: max(s.entailment_score, s.contradiction_score),
            reverse=True,
        )[:3]

    return FactCheckResult(
        claim=claim,
        verdict=verdict,
        nli_label=nli_label,
        confidence=confidence,
        matched_evidence=matched,
        sources=sources,
        explanation=explanation,
        warnings=warnings,
        conflicting_evidence=conflicting,
        all_scored_sentences=all_scored,
    )


# ---------------------------------------------------------------------------
# Legacy compatibility wrapper (keeps existing main() API working)
# ---------------------------------------------------------------------------

def retrieve_evidence_legacy(claim: str) -> dict | None:
    """
    Legacy wrapper that returns a single-page dict for backward compatibility
    with any code that used the original retrieve_evidence() signature.

    New code should use check_claim() directly.
    """
    pages = retrieve_evidence(claim, max_pages=1)
    if not pages:
        return None
    p = pages[0]
    return {"title": p.title, "text": p.text, "url": p.url}


def verify_claim_legacy(claim: str, evidence: str) -> dict:
    """
    Legacy wrapper for backward compatibility with the original verify_claim().

    Accepts a plain evidence string and returns a flat dict.
    New code should use check_claim() directly.
    """
    load_nli_model()
    sentences = _split_into_sentences(evidence)
    ranked = rank_sentences(claim, sentences)

    if not ranked:
        return {
            "prediction": "INSUFFICIENT EVIDENCE",
            "nli_label": "neutral",
            "confidence": 0.0,
            "matched_evidence": "",
        }

    best_sent, best_score, best_label, best_conf = None, -1.0, "neutral", 0.0
    for sent_text, _ in ranked:
        scores = run_nli(premise=sent_text, hypothesis=claim)
        dom = max(scores, key=scores.get)
        if scores[dom] > best_score:
            best_score = scores[dom]
            best_sent = sent_text
            best_label = dom
            best_conf = scores[dom]

    mapping = {"entailment": "SUPPORTED", "contradiction": "CONTRADICTED", "neutral": "INSUFFICIENT EVIDENCE"}
    return {
        "prediction": mapping.get(best_label, "INSUFFICIENT EVIDENCE"),
        "nli_label": best_label,
        "confidence": best_conf,
        "matched_evidence": best_sent or "",
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Interactive command-line interface for manual testing."""
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

    print("\n" + "=" * 60)
    print("  EVIDENCE-BASED FACT CHECKER  (v2)")
    print("=" * 60)
    print("  NLI model : cross-encoder/nli-deberta-v3-small")
    print(f"  NLI threshold  : {NLI_CONFIDENCE_THRESHOLD:.0%}")
    print(f"  TF-IDF min sim : {MIN_TFIDF_SIMILARITY}")
    print(f"  Max Wiki pages : {MAX_WIKI_PAGES}")
    print("=" * 60)

    claim = input("\nEnter a factual claim: ").strip()

    if not claim:
        print("No claim entered. Exiting.")
        return

    print("\nLoading NLI model (first run may take a minute)...")
    try:
        load_nli_model()
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        return

    print(f"\nSearching Wikipedia for evidence about: {claim!r}")
    result = check_claim(claim)

    print("\n" + "=" * 60)
    print(f"  VERDICT: {result.verdict}")
    print("=" * 60)
    print(f"  NLI label  : {result.nli_label}")
    if result.confidence > 0:
        print(f"  Confidence : {result.confidence:.2%}")
    if result.conflicting_evidence:
        print("  ⚠  CONFLICTING evidence detected")

    print(f"\nExplanation:\n  {result.explanation}")

    if result.sources:
        print("\nSources consulted:")
        for src in result.sources:
            print(f"  • {src['title']}")
            if src["url"]:
                print(f"    {src['url']}")

    if result.matched_evidence:
        print("\nKey evidence sentences:")
        for ev in result.matched_evidence[:3]:
            print(f"\n  [{ev.source_title}]")
            print(f"  TF-IDF sim   : {ev.tfidf_similarity:.4f}")
            print(f"  Entailment   : {ev.entailment_score:.2%}")
            print(f"  Contradiction: {ev.contradiction_score:.2%}")
            print(f"  Neutral      : {ev.neutral_score:.2%}")
            print(f"  Text: {ev.text[:200]}")

    print("\nWarnings:")
    for w in result.warnings:
        print(f"  ⚠  {w}")

    print()


if __name__ == "__main__":
    main()
