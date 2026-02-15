from pathlib import Path
import pandas as pd

K = 10

def main():
    mapping_path = Path("outputs/mapping_topk.csv")
    if (Path("outputs/mapping_topk_explained.csv")).exists():
        mapping_path = Path("outputs/mapping_topk_explained.csv")

    pred = pd.read_csv(mapping_path)
    gold = pd.read_csv("evaluation/gold_mapping.csv")

    # Keep only top-K predictions per strategy
    pred = pred[pred["rank"] <= K].copy()

    # Build sets
    pred_pairs = set(zip(pred["strategy_id"].astype(str), pred["action_id"].astype(str)))
    gold_pairs = set(zip(gold["strategy_id"].astype(str), gold["action_id"].astype(str)))

    true_pos = len(pred_pairs.intersection(gold_pairs))
    precision_at_k = true_pos / max(len(pred_pairs), 1)
    recall_at_k = true_pos / max(len(gold_pairs), 1)

    # per-strategy metrics
    rows = []
    for sid in sorted(gold["strategy_id"].unique()):
        pred_s = set(zip(pred[pred["strategy_id"] == sid]["strategy_id"].astype(str),
                         pred[pred["strategy_id"] == sid]["action_id"].astype(str)))
        gold_s = set(zip(gold[gold["strategy_id"] == sid]["strategy_id"].astype(str),
                         gold[gold["strategy_id"] == sid]["action_id"].astype(str)))

        tp = len(pred_s.intersection(gold_s))
        prec = tp / max(len(pred_s), 1)
        rec = tp / max(len(gold_s), 1)

        rows.append({"strategy_id": sid, "precision@K": prec, "recall@K": rec, "tp": tp,
                     "pred_count": len(pred_s), "gold_count": len(gold_s)})

    out = pd.DataFrame(rows)
    Path("outputs").mkdir(exist_ok=True)
    out.to_csv("outputs/eval_per_strategy.csv", index=False)

    print("=== Evaluation ===")
    print(f"K={K}")
    print(f"TP={true_pos}")
    print(f"Precision@K={precision_at_k:.3f}")
    print(f"Recall@K={recall_at_k:.3f}")
    print("[OK] Wrote outputs/eval_per_strategy.csv")

if __name__ == "__main__":
    main()
