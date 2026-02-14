from pathlib import Path
import json
from tqdm import tqdm
import chromadb
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def load_jsonl(path: Path):
    recs = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            recs.append(json.loads(line))
    return recs

def main():
    strategies = load_jsonl(Path("data/processed/strategies.jsonl"))
    actions = load_jsonl(Path("data/processed/actions.jsonl"))

    print(f"Loaded strategies={len(strategies)}, actions={len(actions)}")

    model = SentenceTransformer(MODEL_NAME)

    # Persistent Chroma
    client = chromadb.PersistentClient(path="db_chroma")

    # Two collections: strategies + actions
    col_s = client.get_or_create_collection(name="strategies")
    col_a = client.get_or_create_collection(name="actions")

    # ---- Index strategies
    s_ids = [s["strategy_id"] for s in strategies]
    s_docs = [s["text"] for s in strategies]
    s_metas = [{"goal_no": s["goal_no"], "title": s["title"], "source_doc": s["source_doc"]} for s in strategies]
    s_emb = model.encode(s_docs, show_progress_bar=True).tolist()

    # Clear old and add fresh (simple approach)
    try:
        col_s.delete(ids=s_ids)
    except Exception:
        pass
    col_s.add(ids=s_ids, documents=s_docs, metadatas=s_metas, embeddings=s_emb)

    # ---- Index actions
    a_ids = [a["action_id"] for a in actions]
    a_docs = [a["text"] for a in actions]
    a_metas = [{
        "goal_no": a["goal_no"],
        "csp_ref": a["csp_ref"],
        "service": a["service"],
        "delivery_stream": a["delivery_stream"],
        "page_index": a["page_index"],
        "source_doc": a["source_doc"]
    } for a in actions]
    a_emb = model.encode(a_docs, show_progress_bar=True).tolist()

    # delete may fail on huge list; ignore
    try:
        col_a.delete(ids=a_ids)
    except Exception:
        pass
    # Add in batches
    BATCH = 200
    for i in tqdm(range(0, len(a_ids), BATCH), desc="Upserting actions"):
        col_a.add(
            ids=a_ids[i:i+BATCH],
            documents=a_docs[i:i+BATCH],
            metadatas=a_metas[i:i+BATCH],
            embeddings=a_emb[i:i+BATCH]
        )

    print("[OK] Chroma DB built at ./db_chroma")
    print("Collections:", [c.name for c in client.list_collections()])

if __name__ == "__main__":
    main()
