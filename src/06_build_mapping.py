from pathlib import Path
import json
import pandas as pd
import chromadb
from config import CHROMA_PATH, ACTIONS_COLLECTION, TOP_K, dist_to_sim

def load_jsonl(path: Path):
    recs = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            recs.append(json.loads(line))
    return recs

def main():
    strategies = load_jsonl(Path("data/processed/strategies.jsonl"))

    client = chromadb.PersistentClient(path=CHROMA_PATH)
    col_a = client.get_collection(name=ACTIONS_COLLECTION)

    rows = []
    for s in strategies:
        s_id = s["strategy_id"]
        s_text = s["text"]

        res = col_a.query(
            query_texts=[s_text],
            n_results=TOP_K,
            include=["documents", "metadatas", "distances"]
        )

        docs = res["documents"][0]
        metas = res["metadatas"][0]
        dists = res["distances"][0]

        for rank, (doc, meta, dist) in enumerate(zip(docs, metas, dists), start=1):
            sim = dist_to_sim(dist)

            rows.append({
                "strategy_id": s_id,
                "strategy_goal_no": s.get("goal_no"),
                "strategy_title": s.get("title", ""),
                "rank": rank,
                "action_id": meta.get("action_id", ""),  # may be empty if not stored
                "action_goal_no": meta.get("goal_no"),
                "csp_ref": meta.get("csp_ref"),
                "service": meta.get("service"),
                "delivery_stream": meta.get("delivery_stream"),
                "distance": float(dist),
                "similarity": float(sim),
                "action_text": doc[:500]  # keep CSV readable (full text still in actions.jsonl)
            })

    df = pd.DataFrame(rows)
    Path("outputs").mkdir(exist_ok=True)
    out_csv = Path("outputs/mapping_topk.csv")
    df.to_csv(out_csv, index=False)
    print(f"[OK] Saved mapping to {out_csv} with {len(df)} rows")

if __name__ == "__main__":
    main()
