import json
from pathlib import Path
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Strategic–Action Sync Dashboard", layout="wide")

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
DATA = ROOT / "data_processed"

def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))

def load_jsonl(path: Path):
    recs = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            recs.append(json.loads(line))
    return recs

st.title("Strategic Plan ↔ Action Plan Synchronization")

# ---- Load core outputs
overall = load_json(OUT / "overall_metrics.json") if (OUT / "overall_metrics.json").exists() else {}
gaps = load_json(OUT / "gaps.json") if (OUT / "gaps.json").exists() else {}
strategy_metrics = pd.read_csv(OUT / "strategy_metrics.csv") if (OUT / "strategy_metrics.csv").exists() else pd.DataFrame()
"""
mapping_file = OUT / "mapping_topk_explained.csv"
if not mapping_file.exists():
    mapping_file = OUT / "mapping_topk.csv"
mapping = pd.read_csv(mapping_file) if mapping_file.exists() else pd.DataFrame()
"""
# Prefer best mapping output
mapping_candidates = [
    OUT / "mapping_topk_goal_filtered_hybrid.csv",
    OUT / "mapping_topk_goal_filtered_v2.csv",
    OUT / "mapping_topk_goal_filtered.csv",
    OUT / "mapping_topk_explained.csv",
    OUT / "mapping_topk.csv",
]
mapping_file = next((p for p in mapping_candidates if p.exists()), None)
mapping = pd.read_csv(mapping_file) if mapping_file else pd.DataFrame()

suggestions_path = OUT / "suggestions.json"
suggestions = load_json(suggestions_path) if suggestions_path.exists() else []

improvements_path = OUT / "improvement_actions.csv"
improvements = pd.read_csv(improvements_path) if improvements_path.exists() else pd.DataFrame()

strategies = load_jsonl(DATA / "strategies.jsonl") if (DATA / "strategies.jsonl").exists() else []
strat_by_id = {s["strategy_id"]: s for s in strategies}

# ---- KPIs top row
c1, c2, c3, c4 = st.columns(4)
c1.metric("Overall Sync Score", f"{overall.get('overall_sync_score', 0):.3f}")
c2.metric("# Strategies", overall.get("num_strategies", len(strategy_metrics)))
c3.metric("# Actions (total)", overall.get("num_actions_total", "—"))
c4.metric("Orphan Actions", overall.get("num_orphan_actions", "—"))

st.divider()

# ---- Tabs
tab1, tab2, tab3, tab4 = st.tabs(["Strategy Mapping", "Gaps", "LLM Suggestions", "Data & Evidence"])

with tab1:
    st.subheader("Strategy-wise Mapping (Top-K Actions)")
    if strategy_metrics.empty or mapping.empty:
        st.warning("Missing mapping/metrics. Run Part 4 first.")
    else:
        # choose strategy
        strat_ids = sorted(mapping["strategy_id"].unique())
        sid = st.selectbox("Select Strategy", strat_ids)

        st.write("### Strategy Text")
        st.write(strat_by_id.get(sid, {}).get("text", "Strategy text not found."))

        st.write("### Top-K Actions")
        sub = mapping[mapping["strategy_id"] == sid].sort_values("rank")
        show_cols = [c for c in sub.columns if c not in ["strategy_title"]]
        st.dataframe(sub[show_cols], use_container_width=True)

        st.write("### Per-Strategy Metrics")
        sm = strategy_metrics[strategy_metrics["strategy_id"] == sid]
        st.dataframe(sm, use_container_width=True)

with tab2:
    st.subheader("Alignment Gaps")
    weak = gaps.get("weak_strategies", [])
    orphan = gaps.get("orphan_actions_sample", [])

    colA, colB = st.columns(2)

    with colA:
        st.write("#### Weak Strategies")
        if not weak:
            st.success("No weak strategies under current threshold.")
        else:
            st.dataframe(pd.DataFrame(weak), use_container_width=True)

    with colB:
        st.write("#### Orphan Actions (sample)")
        if not orphan:
            st.info("No orphan sample found.")
        else:
            st.dataframe(pd.DataFrame(orphan), use_container_width=True)

with tab3:
    st.subheader("LLM-powered Improvement Recommendations")
    if not suggestions:
        st.warning("No suggestions.json found. Run Part 5 first.")
    else:
        sids = [s.get("strategy_id") for s in suggestions]
        sid = st.selectbox("Select Strategy for Recommendations", sorted(set(sids)))

        s_item = next((x for x in suggestions if x.get("strategy_id") == sid), None)
        if s_item:
            st.write("### Gaps")
            st.write(s_item.get("gaps", []))

            st.write("### Improvements to Existing Actions")
            st.dataframe(pd.DataFrame(s_item.get("improvements_to_existing_actions", [])), use_container_width=True)

            st.write("### Recommended New/Revised Actions")
            st.dataframe(pd.DataFrame(s_item.get("recommended_actions", [])), use_container_width=True)

            st.write("### Quick Wins")
            st.write(s_item.get("quick_wins", []))

            st.write("### Risks")
            st.write(s_item.get("risks", []))

    if not improvements.empty:
        st.write("## Combined Improvement Actions Table")
        st.dataframe(improvements, use_container_width=True)

with tab4:
    st.subheader("Evidence Files (for report screenshots)")
    st.write("These are the artifacts you should reference in your report:")
    files = [
        "outputs/mapping_topk_explained.csv",
        "outputs/strategy_metrics.csv",
        "outputs/overall_metrics.json",
        "outputs/gaps.json",
        "outputs/suggestions.json",
        "outputs/improvement_actions.csv",
    ]
    for f in files:
        p = ROOT / f
        st.write(f"- {f} " if p.exists() else f"- {f} (missing)")

    st.write("### Download key outputs")
    for f in files:
        p = ROOT / f
        if p.exists():
            st.download_button(
                label=f"Download {p.name}",
                data=p.read_bytes(),
                file_name=p.name
            )
