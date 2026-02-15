from pathlib import Path
import json
import pandas as pd
import chromadb
from config import CHROMA_PATH, ACTIONS_COLLECTION, dist_to_sim

TOP_K_FINAL = 10
CANDIDATES = 50  # fetch more candidates before filtering by goal_no

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
        s_goal = int(s.get("goal_no"))

        res = col_a.query(
            query_texts=[s["text"]],
            n_results=CANDIDATES,
            include=["documents", "metadatas", "distances"]
        )

        ids = res["ids"][0]
        docs = res["documents"][0]
        metas = res["metadatas"][0]
        dists = res["distances"][0]

        cand = []
        for aid, doc, meta, dist in zip(ids, docs, metas, dists):
            a_goal = meta.get("goal_no")
            try:
                a_goal = int(a_goal)
            except:
                a_goal = None

            cand.append({
                "strategy_id": s["strategy_id"],
                "strategy_goal_no": s_goal,
                "action_id": aid,  # always available
                "action_goal_no": a_goal,
                "csp_ref": meta.get("csp_ref"),
                "service": meta.get("service"),
                "delivery_stream": meta.get("delivery_stream"),
                "distance": float(dist),
                "similarity": float(dist_to_sim(dist)),
                "action_text": doc[:600],
            })

        # filter candidates by goal number
        cand = [x for x in cand if x["action_goal_no"] == s_goal]

        # sort by similarity and keep best 10
        cand = sorted(cand, key=lambda x: x["similarity"], reverse=True)[:TOP_K_FINAL]

        for i, r in enumerate(cand, start=1):
            r["rank"] = i
            rows.append(r)

    df = pd.DataFrame(rows)
    Path("outputs").mkdir(exist_ok=True)
    df.to_csv("outputs/mapping_topk_goal_filtered_v2.csv", index=False)
    print("[OK] outputs/mapping_topk_goal_filtered_v2.csv written", df.shape)

if __name__ == "__main__":
    main()
