from pathlib import Path
import json
import re

GOAL_RE = re.compile(r"\bGoal\s+(\d+)\b", re.IGNORECASE)

def load_pages(path: Path):
    pages = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            pages.append(json.loads(line))
    return pages

def build_strategies(pages):
    full_text = "\n".join(p["text"] for p in pages if p["text"])
    # Find all goal starts
    matches = list(GOAL_RE.finditer(full_text))
    strategies = []

    if not matches:
        # fallback: entire doc as one strategy
        strategies.append({
            "strategy_id": "S1",
            "goal_no": 1,
            "title": "Strategic Plan (whole document)",
            "text": full_text.strip(),
            "source_doc": "strategic_plan.pdf",
        })
        return strategies

    for idx, m in enumerate(matches):
        goal_no = int(m.group(1))
        start = m.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(full_text)
        chunk = full_text[start:end].strip()

        # Try to grab first line after "Goal X" as title-ish
        first_line = chunk.splitlines()[0].strip()
        title = first_line

        strategies.append({
            "strategy_id": f"S{goal_no}",
            "goal_no": goal_no,
            "title": title,
            "text": chunk,
            "source_doc": "strategic_plan.pdf",
        })

    return strategies

def write_jsonl(records, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    pages = load_pages(Path("data/processed/strategic_pages.jsonl"))
    strategies = build_strategies(pages)
    write_jsonl(strategies, Path("data/processed/strategies.jsonl"))
    print(f"[OK] strategies.jsonl created with {len(strategies)} strategies")
    for s in strategies:
        print(s["strategy_id"], "-", s["title"][:70])
