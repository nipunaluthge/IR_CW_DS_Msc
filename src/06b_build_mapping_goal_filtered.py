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
        goal_no = s.get("goal_no")
        res = col_a.query(
            query_texts=[s["text"]],
            n_results=TOP_K,
            where={"goal_no": goal_no},  
            include=["documents", "metadatas", "distances"]
        )

        for rank, (doc, meta, dist) in enumerate(zip(res["documents"][0], res["metadatas"][0], res["distances"][0]), start=1):
            rows.append({
                "strategy_id": s["strategy_id"],
                "strategy_goal_no": goal_no,
                "rank": rank,
                "action_id": meta.get("action_id", ""),
                "action_goal_no": meta.get("goal_no"),
                "csp_ref": meta.get("csp_ref"),
                "service": meta.get("service"),
                "distance": float(dist),
                "similarity": float(dist_to_sim(dist)),
                "action_text": doc[:600],
            })

    df = pd.DataFrame(rows)
    Path("outputs").mkdir(exist_ok=True)
    df.to_csv("outputs/mapping_topk_goal_filtered.csv", index=False)
    print("[OK] outputs/mapping_topk_goal_filtered.csv written")

if __name__ == "__main__":
    main()
