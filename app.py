"""
Streamlit application: AI Hallucination and Misinformation Detector
Evidence-based claim verification using Wikipedia IR and NLI.

Run with:  streamlit run app.py
(from the hallucination_detection/ directory)
"""

from __future__ import annotations
import joblib
from pathlib import Path

import json
import logging
import time
from pathlib import Path

import streamlit as st

# ─── Page config (must be the FIRST Streamlit call) ──────────────────────────
st.set_page_config(
    page_title="AI Hallucination & Misinformation Detector",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
/* Google Font */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* Global reset */
*, *::before, *::after { box-sizing: border-box; }

html, body, [data-testid="stAppViewContainer"] {
    background: #0d1117;
    color: #e6edf3;
    font-family: 'Inter', sans-serif;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #161b22;
    border-right: 1px solid #21262d;
}
[data-testid="stSidebar"] * { color: #c9d1d9 !important; }

/* Remove default Streamlit padding */
.block-container { padding: 2rem 2rem 4rem; max-width: 1200px; }

/* ── Hero header ── */
.hero-container {
    background: linear-gradient(135deg, #0d1117 0%, #161b22 50%, #0d1117 100%);
    border: 1px solid #21262d;
    border-radius: 16px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero-container::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; height: 3px;
    background: linear-gradient(90deg, #58a6ff, #bc8cff, #ff7b72, #ffa657);
}
.hero-title {
    font-size: 2.4rem; font-weight: 700; letter-spacing: -0.02em;
    background: linear-gradient(135deg, #58a6ff 0%, #bc8cff 50%, #ff7b72 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; margin: 0 0 0.5rem;
}
.hero-subtitle {
    font-size: 1rem; color: #8b949e; font-weight: 400; margin: 0;
    line-height: 1.6;
}
.hero-badges {
    display: flex; gap: 0.6rem; flex-wrap: wrap; margin-top: 1.2rem;
}
.badge {
    background: #21262d; border: 1px solid #30363d;
    border-radius: 20px; padding: 0.25rem 0.75rem;
    font-size: 0.78rem; color: #8b949e; font-weight: 500;
}

/* ── Input card ── */
.input-card {
    background: #161b22; border: 1px solid #21262d;
    border-radius: 12px; padding: 1.5rem; margin-bottom: 1.5rem;
}
.input-label {
    font-size: 0.9rem; font-weight: 600; color: #58a6ff;
    text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 0.5rem;
}

/* Text area override */
textarea {
    background: #0d1117 !important;
    border: 1px solid #30363d !important;
    border-radius: 8px !important;
    color: #e6edf3 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 1rem !important;
}
textarea:focus { border-color: #58a6ff !important; outline: none !important; }

/* ── Verdict cards ── */
.verdict-card {
    border-radius: 14px; padding: 1.8rem 2rem;
    margin-bottom: 1.5rem; position: relative; overflow: hidden;
}
.verdict-card::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; height: 4px;
}
.verdict-supported {
    background: linear-gradient(135deg, #0d2817 0%, #112810 100%);
    border: 1px solid #238636;
}
.verdict-supported::before { background: linear-gradient(90deg, #3fb950, #2ea043); }
.verdict-contradicted {
    background: linear-gradient(135deg, #2d0d0d 0%, #1f0a0a 100%);
    border: 1px solid #da3633;
}
.verdict-contradicted::before { background: linear-gradient(90deg, #f85149, #da3633); }
.verdict-insufficient {
    background: linear-gradient(135deg, #1c1a0d 0%, #17160a 100%);
    border: 1px solid #9e6a03;
}
.verdict-insufficient::before { background: linear-gradient(90deg, #e3b341, #9e6a03); }

.verdict-icon { font-size: 2.8rem; margin-bottom: 0.5rem; }
.verdict-label {
    font-size: 1.5rem; font-weight: 700; letter-spacing: 0.02em;
    margin: 0 0 0.4rem;
}
.verdict-supported .verdict-label { color: #3fb950; }
.verdict-contradicted .verdict-label { color: #f85149; }
.verdict-insufficient .verdict-label { color: #e3b341; }
.verdict-explanation { color: #8b949e; font-size: 0.95rem; line-height: 1.6; }
.verdict-claim { color: #c9d1d9; font-style: italic; font-size: 0.9rem; margin-top: 0.8rem; }

/* ── Confidence bar ── */
.conf-container { margin: 1rem 0; }
.conf-label {
    display: flex; justify-content: space-between;
    font-size: 0.82rem; color: #8b949e; margin-bottom: 0.3rem;
}
.conf-bar-bg {
    height: 8px; background: #21262d; border-radius: 4px; overflow: hidden;
}
.conf-bar-fill {
    height: 100%; border-radius: 4px;
    transition: width 0.8s ease;
}
.conf-bar-supported { background: linear-gradient(90deg, #3fb950, #2ea043); }
.conf-bar-contradicted { background: linear-gradient(90deg, #f85149, #da3633); }
.conf-bar-neutral { background: linear-gradient(90deg, #e3b341, #9e6a03); }

/* ── Evidence cards ── */
.evidence-card {
    background: #161b22; border: 1px solid #21262d;
    border-radius: 10px; padding: 1.2rem 1.4rem; margin-bottom: 0.8rem;
    transition: border-color 0.2s;
}
.evidence-card:hover { border-color: #388bfd; }
.evidence-meta {
    display: flex; align-items: center; gap: 0.6rem;
    margin-bottom: 0.6rem;
}
.evidence-source {
    font-size: 0.8rem; font-weight: 600; color: #58a6ff;
    background: #1f2937; padding: 0.15rem 0.5rem; border-radius: 4px;
}
.evidence-sim {
    font-size: 0.75rem; color: #6e7681;
    font-family: 'JetBrains Mono', monospace;
}
.evidence-text { color: #c9d1d9; font-size: 0.93rem; line-height: 1.6; }
.evidence-scores {
    display: flex; gap: 1rem; margin-top: 0.8rem; flex-wrap: wrap;
}
.score-chip {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem; padding: 0.2rem 0.6rem; border-radius: 4px;
}
.score-entailment { background: #0d2817; color: #3fb950; border: 1px solid #238636; }
.score-contradiction { background: #2d0d0d; color: #f85149; border: 1px solid #da3633; }
.score-neutral { background: #1c1a0d; color: #e3b341; border: 1px solid #9e6a03; }

/* ── Source links ── */
.source-link {
    display: inline-flex; align-items: center; gap: 0.4rem;
    color: #58a6ff; text-decoration: none; font-size: 0.88rem;
    padding: 0.3rem 0.7rem; border: 1px solid #21262d;
    border-radius: 6px; background: #161b22; margin: 0.2rem;
    transition: border-color 0.2s, background 0.2s;
}
.source-link:hover { border-color: #58a6ff; background: #1f2a3d; }

/* ── Warning box ── */
.warning-box {
    background: #1c1700; border: 1px solid #9e6a03;
    border-radius: 8px; padding: 0.8rem 1rem; margin: 0.5rem 0;
    color: #e3b341; font-size: 0.85rem;
}
.conflict-box {
    background: #2d1a0d; border: 1px solid #bd561d;
    border-radius: 8px; padding: 0.8rem 1rem; margin: 0.5rem 0;
    color: #ffa657; font-size: 0.85rem; font-weight: 500;
}

/* ── Section titles ── */
.section-title {
    font-size: 0.85rem; font-weight: 700; color: #58a6ff;
    text-transform: uppercase; letter-spacing: 0.08em;
    border-bottom: 1px solid #21262d; padding-bottom: 0.5rem;
    margin: 1.5rem 0 1rem;
}

/* ── Metric chips ── */
.metric-row { display: flex; gap: 1rem; flex-wrap: wrap; margin: 0.8rem 0; }
.metric-chip {
    background: #161b22; border: 1px solid #21262d;
    border-radius: 8px; padding: 0.6rem 1rem; min-width: 120px;
}
.metric-chip-label { font-size: 0.72rem; color: #6e7681; text-transform: uppercase; letter-spacing: 0.06em; }
.metric-chip-value { font-size: 1.1rem; font-weight: 600; color: #e6edf3; margin-top: 0.1rem; }

/* ── Baseline comparison table ── */
.comparison-table {
    width: 100%; border-collapse: collapse;
    font-size: 0.88rem; margin-top: 0.5rem;
}
.comparison-table th {
    background: #21262d; color: #8b949e;
    font-weight: 600; text-transform: uppercase;
    letter-spacing: 0.05em; font-size: 0.78rem;
    padding: 0.6rem 1rem; text-align: left;
    border: 1px solid #30363d;
}
.comparison-table td {
    padding: 0.6rem 1rem; border: 1px solid #21262d; color: #c9d1d9;
}
.comparison-table tr:hover td { background: #161b22; }

/* ── Buttons ── */
.stButton button {
    background: linear-gradient(135deg, #1f6feb, #388bfd) !important;
    color: #ffffff !important; border: none !important;
    border-radius: 8px !important; font-weight: 600 !important;
    font-size: 1rem !important; padding: 0.6rem 2rem !important;
    transition: opacity 0.2s, transform 0.1s !important;
    font-family: 'Inter', sans-serif !important;
}
.stButton button:hover { opacity: 0.9 !important; transform: translateY(-1px) !important; }
.stButton button:active { transform: translateY(0) !important; }

/* ── Spinner ── */
.stSpinner > div { border-top-color: #58a6ff !important; }

/* ── Info / error boxes ── */
.stAlert { border-radius: 8px !important; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] { background: #161b22; border-radius: 8px; gap: 0; }
.stTabs [data-baseweb="tab"] {
    color: #8b949e !important; font-weight: 500 !important;
    border-radius: 6px !important; padding: 0.5rem 1.2rem !important;
}
.stTabs [aria-selected="true"] {
    background: #21262d !important; color: #e6edf3 !important;
}

/* ── Expander ── */
details { background: #161b22; border-radius: 8px; border: 1px solid #21262d; }
summary { padding: 0.8rem 1rem; cursor: pointer; color: #8b949e; font-size: 0.9rem; }
</style>
""",
    unsafe_allow_html=True,
)

# ─── Lazy imports (after Streamlit config) ───────────────────────────────────
@st.cache_resource
def load_hallucination_model():
    model_path = (
        Path(__file__).parent
        / "models"
        / "response_only__linear_svc.joblib"
    )
    return joblib.load(model_path)

@st.cache_resource(show_spinner=False)
def get_nli_pipeline():
    """Load and cache the NLI model for the full Streamlit session."""
    import fact_checker as fc
    fc.load_nli_model()
    return fc




# ─── Helpers ─────────────────────────────────────────────────────────────────

def _verdict_class(verdict: str) -> str:
    mapping = {
        "SUPPORTED": "verdict-supported",
        "CONTRADICTED": "verdict-contradicted",
        "INSUFFICIENT EVIDENCE": "verdict-insufficient",
    }
    return mapping.get(verdict, "verdict-insufficient")


def _verdict_icon(verdict: str) -> str:
    return {"SUPPORTED": "✅", "CONTRADICTED": "❌", "INSUFFICIENT EVIDENCE": "⚠️"}.get(verdict, "⚠️")


def _conf_bar_class(label: str) -> str:
    return {
        "entailment": "conf-bar-supported",
        "contradiction": "conf-bar-contradicted",
    }.get(label, "conf-bar-neutral")


def _render_confidence_bar(label: str, value: float, title: str) -> str:
    pct = int(value * 100)
    bar_class = _conf_bar_class(label)
    return f"""
    <div class="conf-container">
        <div class="conf-label"><span>{title}</span><span>{pct}%</span></div>
        <div class="conf-bar-bg">
            <div class="conf-bar-fill {bar_class}" style="width:{pct}%"></div>
        </div>
    </div>
    """


def _load_baseline_metrics() -> list[dict]:
    results_path = Path(__file__).parent / "results" / "metrics.json"
    if not results_path.exists():
        return []
    try:
        data = json.loads(results_path.read_text(encoding="utf-8"))
        return data.get("models", [])
    except Exception:
        return []


# ─── Sidebar ─────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚙️ Settings")

    nli_threshold = st.slider(
        "NLI Confidence Threshold",
        min_value=0.40, max_value=0.95, value=0.60, step=0.05,
        help="Minimum NLI score required to declare a claim SUPPORTED or CONTRADICTED. "
             "Higher = fewer false positives but more INSUFFICIENT EVIDENCE results.",
    )
    min_tfidf = st.slider(
        "Min TF-IDF Similarity",
        min_value=0.01, max_value=0.30, value=0.05, step=0.01,
        help="Minimum sentence-to-claim similarity for a sentence to qualify as candidate evidence.",
    )
    max_pages = st.select_slider(
        "Wikipedia Pages to Search",
        options=[1, 2, 3, 4, 5], value=3,
        help="More pages improve recall but increase response time.",
    )
    top_k = st.select_slider(
        "Sentences per Page",
        options=[3, 5, 7, 10], value=5,
        help="Number of top-ranked sentences evaluated per Wikipedia page.",
    )

    st.markdown("---")
    st.markdown("### 📖 About")
    st.markdown(
        """
        **Evidence Retrieval**: Wikipedia Search API  
        **Ranking**: TF-IDF cosine similarity  
        **NLI Model**: `cross-encoder/nli-deberta-v3-small`  
        **Pipeline**: Hugging Face Transformers  
        """
    )

    st.markdown("---")
    st.markdown(
        """
        <div style='font-size:0.78rem; color:#6e7681; line-height:1.5'>
        ⚠️ This tool provides automated assessments based on Wikipedia.
        It is not a substitute for expert fact-checking or authoritative sources.
        A <strong>SUPPORTED</strong> verdict means the retrieved evidence
        appears to entail the claim -- not that the claim is objectively true.
        </div>
        """,
        unsafe_allow_html=True,
    )
            # Machine learning hallucination prediction
                # Machine learning hallucination prediction
                            

                                           
    st.markdown("### ML Model Prediction")

    try:
        ml_model = load_hallucination_model()
        ml_prediction = int(ml_model.predict([raw_claim])[0])

        if ml_prediction == 1:
            st.warning(
                "Likely hallucinated "
                "(predicted dataset label 1)"
            )
        else:
            st.success(
                "Likely non-hallucinated "
                "(predicted dataset label 0)"
            )

        st.caption(
            "This is a statistical prediction from your trained "
            "response-only model, not a factual verification. "
            "Here it is applied to the claim text for demonstration."
        )

    except Exception as exc:
        st.error(f"ML model prediction failed: {exc}")                 
# ─── Hero header ─────────────────────────────────────────────────────────────

st.markdown(
    """
<div class="hero-container">
    <div class="hero-title">🔬 AI Hallucination & Misinformation Detector</div>
    <div class="hero-subtitle">
        Evidence-based claim verification using Wikipedia information retrieval
        and Natural Language Inference (NLI). Submit a factual claim to retrieve
        relevant evidence and receive an automated verdict.
    </div>
    <div class="hero-badges">
        <span class="badge">🌐 Wikipedia IR</span>
        <span class="badge">🤖 DeBERTa NLI</span>
        <span class="badge">📊 TF-IDF Ranking</span>
        <span class="badge">⚡ Real-time</span>
        <span class="badge">M.Tech Project</span>
    </div>
</div>
""",
    unsafe_allow_html=True,
)

# ─── Main tabs ───────────────────────────────────────────────────────────────

tab_verify, tab_baseline, tab_about = st.tabs(
    ["🔍 Verify Claim", "📊 Baseline Comparison", "ℹ️ About & Limitations"]
)

# ═══════════════════════════════════════════════════════════════════════════
# Tab 1 -- Verify Claim
# ═══════════════════════════════════════════════════════════════════════════
with tab_verify:

    st.markdown('<div class="section-title">Enter a Factual Claim</div>', unsafe_allow_html=True)

    claim_input = st.text_area(
        label="Claim",
        label_visibility="collapsed",
        placeholder='e.g. "The Sun is a star." or "Water boils at 100°C at sea level."',
        height=110,
        key="claim_input",
    )

    col_btn, col_example = st.columns([1, 3])
    with col_btn:
        verify_btn = st.button("🔍 Verify Claim", use_container_width=True, key="verify_btn")
    with col_example:
        example = st.selectbox(
            "Or try an example:",
            options=[
                "-- select --",
                "The Sun is a star.",
                "The Sun revolves around the Earth.",
                "Water is composed of hydrogen and oxygen.",
                "Albert Einstein was born in France.",
                "The Great Wall of China is visible from space.",
                "DNA carries genetic information.",
                "Shakespeare wrote the play Hamlet.",
            ],
            label_visibility="visible",
            key="example_select",
        )
        
if example != "-- select --":
    def set_claim_example(example):
        st.session_state["claim_input"] = example

    st.button(
        "Use Example",
        on_click=set_claim_example,
        args=(example,),
        key="use_example_button"
    )



    # ── Run verification ──────────────────────────────────────────────────
    if verify_btn:
        raw_claim = st.session_state.get("claim_input", "").strip()

        if not raw_claim:
            st.error("⚠️ Please enter a claim before clicking Verify.")
        else:
            # Load model
            with st.spinner("Loading NLI model (first run may take ~30s)..."):
                try:
                    fc = get_nli_pipeline()
                except Exception as exc:
                    st.error(
                        f"❌ **Model loading failed.**\n\n"
                        f"```\n{exc}\n```\n\n"
                        "Ensure `transformers` and `torch` are installed: "
                        "`pip install transformers torch`"
                    )
                    st.stop()

            # Run pipeline
            progress_bar = st.progress(0, text="Searching Wikipedia...")
            start_time = time.time()

            try:
                progress_bar.progress(20, text=f"Searching Wikipedia for: *{raw_claim}*")
                result = fc.check_claim(
                    raw_claim,
                    nli_threshold=nli_threshold,
                    min_tfidf=min_tfidf,
                    top_k=top_k,
                    max_pages=max_pages,
                )
                progress_bar.progress(100, text="Done.")
                elapsed = time.time() - start_time

            except ValueError as exc:
                progress_bar.empty()
                st.error(f"⚠️ Invalid input: {exc}")
                st.stop()
            except RuntimeError as exc:
                progress_bar.empty()
                st.error(f"❌ Model error: {exc}")
                st.stop()
            except Exception as exc:
                progress_bar.empty()
                st.error(
                    f"❌ An unexpected error occurred:\n\n```\n{exc}\n```"
                )
                st.stop()

            time.sleep(0.3)
            progress_bar.empty()

            # ── Verdict card ──────────────────────────────────────────────
            v_class = _verdict_class(result.verdict)
            v_icon = _verdict_icon(result.verdict)

            st.markdown(
                f"""
<div class="verdict-card {v_class}">
    <div class="verdict-icon">{v_icon}</div>
    <div class="verdict-label">{result.verdict}</div>
    <div class="verdict-explanation">{result.explanation}</div>
    <div class="verdict-claim">Claim: "{result.claim}"</div>
</div>
""",
                unsafe_allow_html=True,
            )

            # ── Conflict warning ──────────────────────────────────────────
            if result.conflicting_evidence:
                st.markdown(
                    '<div class="conflict-box">⚡ <strong>Conflicting evidence detected.</strong> '
                    "Different Wikipedia sources appear to support and contradict this claim. "
                    "The verdict has been set to INSUFFICIENT EVIDENCE to reflect this ambiguity.</div>",
                    unsafe_allow_html=True,
                )

            # ── Metrics row ───────────────────────────────────────────────
            st.markdown(
                f"""
<div class="metric-row">
    <div class="metric-chip">
        <div class="metric-chip-label">NLI Label</div>
        <div class="metric-chip-value">{result.nli_label.capitalize()}</div>
    </div>
    <div class="metric-chip">
        <div class="metric-chip-label">Confidence</div>
        <div class="metric-chip-value">{f"{result.confidence:.1%}" if result.confidence > 0 else "--"}</div>
    </div>
    <div class="metric-chip">
        <div class="metric-chip-label">Sources Found</div>
        <div class="metric-chip-value">{len(result.sources)}</div>
    </div>
    <div class="metric-chip">
        <div class="metric-chip-label">Sentences Scored</div>
        <div class="metric-chip-value">{len(result.all_scored_sentences)}</div>
    </div>
    <div class="metric-chip">
        <div class="metric-chip-label">Time</div>
        <div class="metric-chip-value">{elapsed:.1f}s</div>
    </div>
</div>
""",
                unsafe_allow_html=True,
            )

            # ── Confidence bars (per scored sentence, top 3) ──────────────
        # -- Confidence bars (per scored sentence, top 3) --
        if result.all_scored_sentences:
            top3 = sorted(
                result.all_scored_sentences,
                key=lambda s: max(
                    s.entailment_score,
                    s.contradiction_score
                ),
                reverse=True,
            )[:3]

            bars_html = "".join([
                _render_confidence_bar(
                    "entailment",
                    s.entailment_score,
                    f"Entailment -- {s.text[:60]}..."
                )
                + _render_confidence_bar(
                    "contradiction",
                    s.contradiction_score,
                    f"Contradiction -- {s.text[:60]}..."
                )
                for s in top3
            ])

            with st.expander(
                "📊 NLI Score Distribution (top 3 sentences)",
                expanded=False
            ):
                st.markdown(bars_html, unsafe_allow_html=True)



            # ── Sources ───────────────────────────────────────────────────
            if result.sources:
                st.markdown('<div class="section-title">📚 Sources Consulted</div>', unsafe_allow_html=True)
                links_html = " ".join(
                    f'<a class="source-link" href="{src["url"]}" target="_blank" rel="noopener">🔗 {src["title"]}</a>'
                    if src.get("url")
                    else f'<span class="source-link">📄 {src["title"]}</span>'
                    for src in result.sources
                )
                st.markdown(links_html, unsafe_allow_html=True)

            # ── Evidence sentences ────────────────────────────────────────
            if result.matched_evidence:
                st.markdown('<div class="section-title">🔎 Key Evidence</div>', unsafe_allow_html=True)
                for ev in result.matched_evidence[:5]:
                    e_html = f"""
<div class="evidence-card">
    <div class="evidence-meta">
        <span class="evidence-source">{ev.source_title}</span>
        <span class="evidence-sim">TF-IDF: {ev.tfidf_similarity:.3f}</span>
    </div>
    <div class="evidence-text">{ev.text}</div>
    <div class="evidence-scores">
        <span class="score-chip score-entailment">✅ Entailment {ev.entailment_score:.1%}</span>
        <span class="score-chip score-contradiction">❌ Contradiction {ev.contradiction_score:.1%}</span>
        <span class="score-chip score-neutral">⚠️ Neutral {ev.neutral_score:.1%}</span>
    </div>
</div>
"""
                    st.markdown(e_html, unsafe_allow_html=True)

            # ── Warnings ──────────────────────────────────────────────────
            st.markdown('<div class="section-title">⚠️ Warnings & Caveats</div>', unsafe_allow_html=True)
            for w in result.warnings:
                st.markdown(f'<div class="warning-box">⚠️ {w}</div>', unsafe_allow_html=True)

            # ── Raw JSON debug ────────────────────────────────────────────
            with st.expander("🛠️ Raw Result (JSON)", expanded=False):
                st.json(result.to_dict())


# ═══════════════════════════════════════════════════════════════════════════
# Tab 2 -- Baseline Comparison
# ═══════════════════════════════════════════════════════════════════════════
with tab_baseline:
    st.markdown(
        """
<div class="section-title">TF-IDF Classification Baseline vs Evidence-Based NLI System</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown(
        """
<div class="warning-box" style="margin-bottom:1rem">
⚠️ <strong>These two systems solve fundamentally different problems and their metrics
cannot be directly compared.</strong> The baseline measures agreement with dataset
hallucination labels (supervised classification). The NLI system verifies claims
against Wikipedia evidence (unsupervised retrieval + inference).
</div>
""",
        unsafe_allow_html=True,
    )

    # ── System comparison table ───────────────────────────────────────────
    st.markdown("#### System Architecture Comparison")
    st.markdown(
        """
| Property | TF-IDF Baseline | Evidence-Based NLI |
|---|---|---|
| **Task** | Binary classification (hallucinated vs. non-hallucinated) | Claim verification (SUPPORTED / CONTRADICTED / INSUFFICIENT) |
| **Training** | Supervised (labelled dialogue dataset) | Unsupervised (retrieval + pre-trained NLI) |
| **Evidence** | None -- pattern-matches text style | Wikipedia articles (live retrieval) |
| **Input** | Response text (± dialogue context) | Free-form factual claim |
| **Output** | Label 0 or 1 | Verdict + sources + evidence sentences |
| **Model** | TF-IDF + Logistic Regression / LinearSVC | DeBERTa-v3-small NLI cross-encoder |
| **Limitations** | Reflects dataset label patterns, not ground truth | Depends on Wikipedia coverage and NLI accuracy |
""",
        unsafe_allow_html=True,
    )

    # ── Baseline metrics table ────────────────────────────────────────────
    st.markdown("#### Baseline Model Results (on Dialogue Dataset Test Set)")
    st.caption(
        "Metrics reflect performance on dataset-supplied hallucination labels. "
        "They do not establish real-world misinformation detection accuracy."
    )

    models = _load_baseline_metrics()
    if models:
        rows = []
        for m in models:
            rows.append(
                {
                    "Feature Set": m.get("feature_set", "--").replace("_", " ").title(),
                    "Classifier": m.get("model", "--").replace("_", " ").title(),
                    "Val F1": f"{m.get('validation_f1', 0):.4f}",
                    "Test Acc": f"{m.get('test_accuracy', 0):.4f}",
                    "Test P": f"{m.get('test_precision', 0):.4f}",
                    "Test R": f"{m.get('test_recall', 0):.4f}",
                    "Test F1": f"{m.get('test_f1', 0):.4f}",
                    "Paired Acc": f"{m.get('paired_accuracy') or 0:.4f}"
                    if m.get("paired_accuracy") is not None
                    else "--",
                }
            )

        header_html = "".join(f"<th>{k}</th>" for k in rows[0])
        body_html = ""
        for row in rows:
            body_html += "<tr>" + "".join(f"<td>{v}</td>" for v in row.values()) + "</tr>"

        st.markdown(
            f'<table class="comparison-table"><thead><tr>{header_html}</tr></thead>'
            f"<tbody>{body_html}</tbody></table>",
            unsafe_allow_html=True,
        )
    else:
        st.info(
            "No baseline metrics found. "
            "Run `python train.py --data dialogue_data.csv --seed 42` first."
        )

    # ── NLI system summary ────────────────────────────────────────────────
    st.markdown("#### Evidence-Based NLI System")
    st.markdown(
        """
The NLI system does not produce a scalar accuracy figure by default because it
operates on open-domain claims rather than a labelled dataset.  Its performance
depends on:

- **Wikipedia coverage**: whether a relevant article exists and is retrieved.
- **TF-IDF ranking quality**: whether the most factually relevant sentences are
  selected as candidates.
- **NLI model accuracy**: whether the DeBERTa cross-encoder correctly infers
  entailment or contradiction from the evidence.
- **Threshold calibration**: the confidence thresholds are experimental
  parameters that should be evaluated on a held-out benchmark.

Run `python evaluate_fact_checker.py` to measure precision, recall, F1, and
coverage on a curated benchmark of labelled claims.
"""
    )


# ═══════════════════════════════════════════════════════════════════════════
# Tab 3 -- About & Limitations
# ═══════════════════════════════════════════════════════════════════════════
with tab_about:
    st.markdown(
        """
## About This System

This application is an **M.Tech project** demonstrating two complementary
approaches to detecting AI-generated hallucinations and misinformation:

---

### 1. TF-IDF Classification Baseline

A supervised text-classification pipeline trained on a dialogue dataset that
pairs factually correct responses with hallucinated responses.

- **Model**: TF-IDF vectoriser (uni+bigrams, sublinear TF, 150K features)
  + Logistic Regression or LinearSVC.
- **Input**: A response (and optionally knowledge + dialogue context).
- **Output**: Label 0 (likely non-hallucinated) or 1 (likely hallucinated).
- **Limitation**: Reflects patterns in the *dataset's labels*, not general
  factual truth. A response that stylistically resembles hallucinated text
  will be flagged even if it happens to be factually correct.

---

### 2. Evidence-Based NLI Verification

A retrieval-augmented verification pipeline that searches Wikipedia and uses
a Natural Language Inference model to determine whether the evidence supports
or contradicts a given claim.

#### Pipeline Steps

| Step | Description |
|------|-------------|
| **A -- Input** | Validate the claim text |
| **B -- Retrieval** | Search Wikipedia API; fetch up to 3 page extracts |
| **C -- Ranking** | Split extracts into sentences; rank by TF-IDF similarity to claim |
| **D -- NLI** | Run `cross-encoder/nli-deberta-v3-small` on top-k sentences |
| **E -- Aggregation** | Combine NLI scores; detect conflicts; apply confidence threshold |
| **F -- Output** | Return structured verdict with evidence and source URLs |

#### NLI Model

**`cross-encoder/nli-deberta-v3-small`** (Hugging Face)

- Architecture: DeBERTa-v3-small fine-tuned on NLI datasets as a cross-encoder.
- Labels: `entailment`, `contradiction`, `neutral`.
- Input: (premise = evidence sentence, hypothesis = claim).
- The model scores all three labels simultaneously (`top_k=None`).

#### Key Design Choices

- **Full label distribution**: All three NLI scores are inspected, not just
  the top-1 prediction. This prevents misclassification when the model is
  uncertain between entailment and neutral.
- **Configurable thresholds**: The NLI confidence threshold (default 0.60) is
  an experimental parameter -- not a calibrated guarantee.
- **Conflict detection**: If both entailment and contradiction evidence are
  found above threshold, the verdict is INSUFFICIENT EVIDENCE with a warning.
- **Conservative by design**: The system prefers INSUFFICIENT EVIDENCE over
  incorrect SUPPORTED or CONTRADICTED verdicts.

---

### ⚠️ Important Limitations

1. **Wikipedia is not an authoritative fact-checker.** Articles may contain
   errors, outdated information, or contested claims.
2. **NLI models are not truth detectors.** They measure linguistic entailment
   between text passages -- not objective factual accuracy.
3. **TF-IDF ranking is lexical.** A sentence that shares words with the claim
   is not necessarily factually relevant to it.
4. **Coverage is limited.** Claims about niche topics, recent events, or
   personal information may not have Wikipedia coverage.
5. **Thresholds are not calibrated.** The NLI threshold should be evaluated
   on a representative benchmark before being treated as reliable.
6. **This is a research prototype**, not a production fact-checking system.

---

### Hallucination Detection vs. Factual Verification

These are **different problems**:

| | Hallucination Detection | Factual Verification |
|---|---|---|
| **Question** | Does this text *look like* a hallucination? | Is this claim *factually supported*? |
| **Method** | Pattern-matching against labelled examples | Evidence retrieval + logical inference |
| **Output** | Binary label | Three-way verdict + evidence |
| **Requires** | Labelled training data | External knowledge base |
| **Limitation** | Style-dependent; not grounded in facts | Depends on knowledge base quality |

The TF-IDF baseline addresses the first problem. The NLI pipeline addresses
the second. Neither system alone constitutes a reliable general-purpose
fact-checker.
""",
        unsafe_allow_html=True,
    )
