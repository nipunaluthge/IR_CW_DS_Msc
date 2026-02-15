from pathlib import Path
import json
import pandas as pd
from rdflib import Graph, Namespace, Literal, RDF, RDFS

BASE = Namespace("http://example.org/ir_cw/")
SCHEMA = Namespace("http://schema.org/")

def load_jsonl(path: Path):
    recs = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            recs.append(json.loads(line))
    return recs

def main():
    strategies = load_jsonl(Path("data/processed/strategies.jsonl"))
    actions = load_jsonl(Path("data/processed/actions.jsonl"))

    mapping_path = Path("outputs/mapping_topk.csv")
    if not mapping_path.exists():
        raise FileNotFoundError("Run Part 4 first to create outputs/mapping_topk.csv")
    map_df = pd.read_csv(mapping_path)

    g = Graph()
    g.bind("base", BASE)
    g.bind("schema", SCHEMA)

    # Classes
    Strategy = BASE.Strategy
    Action = BASE.Action
    Service = BASE.Service

    # Properties
    supports = BASE.supportsStrategy
    hasService = BASE.hasService
    hasCSPRef = BASE.hasCSPRef
    hasGoalNo = BASE.hasGoalNo
    hasSimilarity = BASE.hasSimilarity

    # Add strategies
    for s in strategies:
        s_uri = BASE[s["strategy_id"]]
        g.add((s_uri, RDF.type, Strategy))
        g.add((s_uri, RDFS.label, Literal(s.get("title", s["strategy_id"]))))
        g.add((s_uri, SCHEMA.text, Literal(s["text"][:1500])))
        g.add((s_uri, hasGoalNo, Literal(s.get("goal_no"))))

    # Add actions
    for a in actions:
        a_uri = BASE[a["action_id"]]
        g.add((a_uri, RDF.type, Action))
        g.add((a_uri, SCHEMA.text, Literal(a["text"][:1500])))
        g.add((a_uri, hasGoalNo, Literal(a.get("goal_no"))))
        if a.get("csp_ref"):
            g.add((a_uri, hasCSPRef, Literal(a["csp_ref"])))

        if a.get("service"):
            service_id = "Service_" + "".join(ch if ch.isalnum() else "_" for ch in a["service"])[:80]
            svc_uri = BASE[service_id]
            g.add((svc_uri, RDF.type, Service))
            g.add((svc_uri, RDFS.label, Literal(a["service"])))
            g.add((a_uri, hasService, svc_uri))

    # Add edges from mapping (top-k supports)
    for _, r in map_df.iterrows():
        s_uri = BASE[str(r["strategy_id"])]
        a_uri = BASE[str(r["action_id"])]
        g.add((a_uri, supports, s_uri))
        g.add((a_uri, hasSimilarity, Literal(float(r["similarity"]))))

    out_ttl = Path("outputs/ir_ontology.ttl")
    g.serialize(destination=str(out_ttl), format="turtle")

    # Stats for report
    stats = {
        "num_triples": len(g),
        "num_strategies": len(strategies),
        "num_actions": len(actions),
        "num_mapping_edges": int(map_df.shape[0])
    }
    Path("outputs/ontology_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print("[OK] Wrote outputs/ir_ontology.ttl")
    print("[OK] Wrote outputs/ontology_stats.json", stats)

if __name__ == "__main__":
    main()
