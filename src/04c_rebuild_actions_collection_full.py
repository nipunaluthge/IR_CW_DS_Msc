from pathlib import Path
import json
import chromadb
from sentence_transformers import SentenceTransformer

from config import CHROMA_PATH, ACTIONS_COLLECTION

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
BATCH = 200

def load_jsonl(path: Path):
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            out.append(json.loads(line))
    return out

def main():
    actions = load_jsonl(Path("data/processed/actions.jsonl"))
    ids = [a["action_id"] for a in actions]
    docs = [a["text"] for a in actions]
    metas = [{
        "action_id": a["action_id"],
        "goal_no": a.get("goal_no"),
        "csp_ref": a.get("csp_ref"),
        "service": a.get("service"),
        "delivery_stream": a.get("delivery_stream"),
        "page_index": a.get("page_index"),
        "source_doc": a.get("source_doc"),
    } for a in actions]

    print("[INFO] Loading embedding model...")
    model = SentenceTransformer(MODEL_NAME)

    print("[INFO] Connecting to Chroma...")
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    # ✅ safest: drop and recreate collection
    try:
        client.delete_collection(ACTIONS_COLLECTION)
        print("[INFO] Deleted old actions collection")
    except Exception as e:
        print("[WARN] Could not delete collection (maybe doesn't exist):", e)

    col = client.get_or_create_collection(name=ACTIONS_COLLECTION)

    print("[INFO] Computing embeddings and inserting...")
    for i in range(0, len(ids), BATCH):
        batch_docs = docs[i:i+BATCH]
        batch_emb = model.encode(batch_docs, show_progress_bar=False).tolist()

        col.add(
            ids=ids[i:i+BATCH],
            documents=batch_docs,
            metadatas=metas[i:i+BATCH],
            embeddings=batch_emb
        )
        print(f"[OK] Inserted {min(i+BATCH, len(ids))}/{len(ids)}")

    print("[DONE] Actions collection rebuilt with embeddings + metadata.")

if __name__ == "__main__":
    main()
