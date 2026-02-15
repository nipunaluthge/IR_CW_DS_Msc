import pandas as pd
import re

STOP = set("""
a an the of to in on for and or with without is are was were be been being
this that these those by as at from it its into we our your they their
""".split())

def tokens(text):
    ws = re.findall(r"[A-Za-z]{3,}", str(text).lower())
    return [w for w in ws if w not in STOP]

def explain(strategy_text, action_text, topn=8):
    s = set(tokens(strategy_text))
    a = set(tokens(action_text))
    overlap = list(s.intersection(a))
    overlap = sorted(overlap, key=lambda x: (-len(x), x))
    return "Common terms: " + ", ".join(overlap[:topn]) if overlap else "Low lexical overlap; likely semantic match."

def main():
    df = pd.read_csv("outputs/mapping_topk.csv")

    # Build dict for strategy full text (from strategies.jsonl)
    import json
    strat_text = {}
    with open("data/processed/strategies.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            s = json.loads(line)
            strat_text[s["strategy_id"]] = s["text"]

    df["explanation"] = df.apply(
        lambda r: explain(strat_text.get(r["strategy_id"], ""), r["action_text"]),
        axis=1
    )

    df.to_csv("outputs/mapping_topk_explained.csv", index=False)
    print("[OK] outputs/mapping_topk_explained.csv written")

if __name__ == "__main__":
    main()
