import chromadb

def retrieve_actions_for_strategy(strategy_text: str, k: int = 10):
    client = chromadb.PersistentClient(path="db_chroma")
    col_a = client.get_collection(name="actions")

    res = col_a.query(
        query_texts=[strategy_text],
        n_results=k,
        include=["documents", "metadatas", "distances"]
    )

    results = []
    for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        results.append({
            "action_text": doc,
            "meta": meta,
            "distance": dist
        })
    return results

if __name__ == "__main__":
    # Example: use a hardcoded strategy query
    strategy_query = "Goal 1 We value and protect our environment. Reduce emissions, protect natural areas, net zero, waste reduction."
    hits = retrieve_actions_for_strategy(strategy_query, k=8)

    for i, h in enumerate(hits, start=1):
        print(f"\n#{i} distance={h['distance']:.4f} goal_no={h['meta'].get('goal_no')} service={h['meta'].get('service')}")
        print("  ", h["action_text"][:220])
