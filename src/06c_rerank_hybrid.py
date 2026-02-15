from pathlib import Path
import json
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

SEM_WEIGHT = 0.5
LEX_WEIGHT = 0.5

def load_jsonl(path: Path):
    recs = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            recs.append(json.loads(line))
    return recs

def main():
    # mapping_path = Path("outputs/mapping_topk_goal_filtered.csv")
    mapping_path = Path("outputs/mapping_topk_goal_filtered_v2.csv")

    if not mapping_path.exists():
        raise FileNotFoundError("Run goal-filtered mapping first (06b).")

    df = pd.read_csv(mapping_path)

    strategies = load_jsonl(Path("data/processed/strategies.jsonl"))
    strat_by_id = {s["strategy_id"]: s["text"] for s in strategies}

    out_rows = []

    for sid in sorted(df["strategy_id"].unique()):
        sub = df[df["strategy_id"] == sid].copy()
        s_text = strat_by_id.get(sid, "")

        # TF-IDF over [strategy + candidate actions]
        corpus = [s_text] + sub["action_text"].astype(str).tolist()
        vec = TfidfVectorizer(stop_words="english", ngram_range=(1,2), max_features=8000)
        X = vec.fit_transform(corpus)

        s_vec = X[0]
        a_vecs = X[1:]
        lex_scores = cosine_similarity(s_vec, a_vecs).flatten()

        sub["lexical_sim"] = lex_scores
        sub["hybrid_score"] = SEM_WEIGHT * sub["similarity"] + LEX_WEIGHT * sub["lexical_sim"]

        # rerank
        sub = sub.sort_values("hybrid_score", ascending=False).reset_index(drop=True)
        sub["rank_hybrid"] = np.arange(1, len(sub) + 1)

        out_rows.append(sub)

    out = pd.concat(out_rows, ignore_index=True)
    out.to_csv("outputs/mapping_topk_goal_filtered_hybrid.csv", index=False)
    print("[OK] outputs/mapping_topk_goal_filtered_hybrid.csv written")

if __name__ == "__main__":
    main()
