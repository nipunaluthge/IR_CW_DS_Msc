from pathlib import Path
import json
import chromadb

from config import CHROMA_PATH, ACTIONS_COLLECTION

def load_jsonl(path: Path):
    recs = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            recs.append(json.loads(line))
    return recs

def main():
    actions = load_jsonl(Path("data/processed/actions.jsonl"))

    client = chromadb.PersistentClient(path=CHROMA_PATH)
    col_a = client.get_collection(name=ACTIONS_COLLECTION)

    # update metadata by re-adding in batches with same ids
    ids = [a["action_id"] for a in actions]
    docs = [a["text"] for a in actions]
    metas = []
    for a in actions:
        metas.append({
            "action_id": a["action_id"],
            "goal_no": a.get("goal_no"),
            "csp_ref": a.get("csp_ref"),
            "service": a.get("service"),
            "delivery_stream": a.get("delivery_stream"),
            "page_index": a.get("page_index"),
            "source_doc": a.get("source_doc"),
        })

    # Fetch existing embeddings already stored; easiest safe route is delete+reinsert via embeddings again.
    # But that costs compute.
    # We'll instead do an "update" style: Chroma add() with same ids overwrites metadata & docs.
    B = 200
    for i in range(0, len(ids), B):
        col_a.add(
            ids=ids[i:i+B],
            documents=docs[i:i+B],
            metadatas=metas[i:i+B]
        )

    print("[OK] Updated action metadata to include action_id")

if __name__ == "__main__":
    main()
