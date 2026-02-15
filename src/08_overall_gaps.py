from pathlib import Path
import pandas as pd
import json

WEAK_THRESHOLD = 0.40   # tune later after you see your scores

def main():
    map_df = pd.read_csv("outputs/mapping_topk.csv")
    strat_df = pd.read_csv("outputs/strategy_metrics.csv")

    overall_score = float(strat_df["avg_similarity_topk"].mean())

    weak = strat_df[strat_df["avg_similarity_topk"] < WEAK_THRESHOLD].copy()
    weak_strategies = weak.to_dict(orient="records")

    # Orphan actions = not appearing in any top-k results
    seen_actions = set(map_df["action_id"].dropna().astype(str).tolist())

    # Load all actions
    all_actions = []
    with open("data/processed/actions.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            all_actions.append(json.loads(line))

    orphan = [a for a in all_actions if a["action_id"] not in seen_actions]

    overall = {
        "overall_sync_score": overall_score,
        "weak_threshold": WEAK_THRESHOLD,
        "num_strategies": int(strat_df.shape[0]),
        "num_actions_total": int(len(all_actions)),
        "num_actions_seen_in_topk": int(len(seen_actions)),
        "num_orphan_actions": int(len(orphan)),
    }

    Path("outputs").mkdir(exist_ok=True)

    with open("outputs/overall_metrics.json", "w", encoding="utf-8") as f:
        json.dump(overall, f, indent=2)

    gaps = {
        "weak_strategies": weak_strategies,
        "orphan_actions_sample": orphan[:30],   # keep file manageable
    }
    with open("outputs/gaps.json", "w", encoding="utf-8") as f:
        json.dump(gaps, f, indent=2)

    print("[OK] outputs/overall_metrics.json written")
    print("[OK] outputs/gaps.json written")
    print("Overall sync score:", overall_score)
    print("Weak strategies:", len(weak_strategies))
    print("Orphan actions:", len(orphan))

if __name__ == "__main__":
    main()
