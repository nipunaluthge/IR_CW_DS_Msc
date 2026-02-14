from pathlib import Path
import pandas as pd
import json
from config import HIGH_SIM, MED_SIM

def label(sim):
    if sim >= HIGH_SIM:
        return "High"
    if sim >= MED_SIM:
        return "Medium"
    return "Low"

def main():
    df = pd.read_csv("outputs/mapping_topk.csv")

    df["label"] = df["similarity"].apply(label)

    # per strategy metrics
    g = df.groupby("strategy_id")

    metrics = []
    for sid, sub in g:
        sims = sub["similarity"].tolist()
        metrics.append({
            "strategy_id": sid,
            "avg_similarity_topk": float(sub["similarity"].mean()),
            "max_similarity": float(sub["similarity"].max()),
            "high_count": int((sub["label"] == "High").sum()),
            "medium_count": int((sub["label"] == "Medium").sum()),
            "low_count": int((sub["label"] == "Low").sum()),
            "best_action_id": sub.loc[sub["similarity"].idxmax(), "action_id"],
            "best_service": sub.loc[sub["similarity"].idxmax(), "service"],
        })

    out = pd.DataFrame(metrics).sort_values("avg_similarity_topk", ascending=False)
    out_path = Path("outputs/strategy_metrics.csv")
    out.to_csv(out_path, index=False)
    print(f"[OK] Wrote {out_path} with {len(out)} strategies")

if __name__ == "__main__":
    main()
