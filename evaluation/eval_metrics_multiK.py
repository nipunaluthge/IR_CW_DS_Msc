from pathlib import Path
import pandas as pd

def eval_for_k(pred_df, gold_df, k):
    pred_k = pred_df[pred_df["rank_hybrid"] <= k].copy() if "rank_hybrid" in pred_df.columns else pred_df[pred_df["rank"] <= k].copy()

    pred_pairs = set(zip(pred_k["strategy_id"].astype(str), pred_k["action_id"].astype(str)))
    gold_pairs = set(zip(gold_df["strategy_id"].astype(str), gold_df["action_id"].astype(str)))

    tp = len(pred_pairs.intersection(gold_pairs))
    precision = tp / max(len(pred_pairs), 1)
    recall = tp / max(len(gold_pairs), 1)
    return tp, precision, recall

def main():
    gold = pd.read_csv("evaluation/gold_mapping.csv")
    """
    pred_files = [
        "outputs/mapping_topk_explained.csv",
        "outputs/mapping_topk_goal_filtered.csv",
        "outputs/mapping_topk_goal_filtered_hybrid.csv"
    ]
    """

    pred_files = [
        "outputs/mapping_topk_explained.csv",
        "outputs/mapping_topk_goal_filtered_v2.csv",
        "outputs/mapping_topk_goal_filtered_hybrid.csv"
    ]


    for pf in pred_files:
        p = Path(pf)
        if not p.exists():
            continue

        pred = pd.read_csv(p)
        print(f"\n=== Evaluating: {pf} ===")
        for k in [5, 10]:
            tp, prec, rec = eval_for_k(pred, gold, k)
            print(f"K={k}  TP={tp}  Precision@K={prec:.3f}  Recall@K={rec:.3f}")

if __name__ == "__main__":
    main()
