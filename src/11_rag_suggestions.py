from pathlib import Path
import json
import pandas as pd
from dotenv import load_dotenv
from config import HIGH_SIM, MED_SIM
from llm_client import llm_complete  # if your editor dislikes this import, see note below
import json
import json5
import re

load_dotenv()

def load_jsonl(path: Path):
    recs = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            recs.append(json.loads(line))
    return recs

def index_by_id(records, key):
    return {r[key]: r for r in records}

def label(sim):
    if sim >= HIGH_SIM:
        return "High"
    if sim >= MED_SIM:
        return "Medium"
    return "Low"

PROMPT_TEMPLATE = """
You are an analyst checking synchronization between a Strategic Plan and an Action Plan.

TASK:
Given the Strategy and the retrieved Actions, do 3 things:
1) Identify gaps / missing work items in the Action Plan that are needed to support the Strategy.
2) Suggest improvements to existing actions (make them SMART: clear owner, timeline, KPI).
3) Propose 3-6 new or revised actions that would better align the Action Plan to the Strategy.

RULES:
- Output MUST be valid JSON only (no extra text).
- Keep suggestions realistic for a City Council context.
- Each recommended action must include: title, description, owner_role, timeframe, KPI, priority (High/Med/Low).
- Also include risks and quick wins.

INPUT:
STRATEGY_ID: {strategy_id}
STRATEGY_TEXT:
{strategy_text}

TOP_MATCHED_ACTIONS (ranked):
{actions_block}

Now output JSON with this schema:
{{
  "strategy_id": "...",
  "gaps": ["...", "..."],
  "improvements_to_existing_actions": [
    {{
      "action_id": "A00001",
      "suggested_change": "...",
      "why": "..."
    }}
  ],
  "recommended_actions": [
    {{
      "title": "...",
      "description": "...",
      "owner_role": "...",
      "timeframe": "...",
      "KPI": "...",
      "priority": "High|Med|Low"
    }}
  ],
  "quick_wins": ["...", "..."],
  "risks": ["...", "..."]
}}
"""

def build_actions_block(mapping_rows):
    lines = []
    for _, r in mapping_rows.iterrows():
        lines.append(
            f"- rank={int(r['rank'])} action_id={r['action_id']} "
            f"sim={r['similarity']:.3f} label={label(r['similarity'])} "
            f"service={r.get('service','')}\n  text={r['action_text']}"
        )
    return "\n".join(lines)

def safe_json_parse(text: str):
    """
    Robust JSON extraction + repair:
    - Extract first {...} block
    - Try strict json
    - If it fails, try json5 (handles single quotes, trailing commas)
    - If it still fails, apply a light cleanup and retry
    """
    text = (text or "").strip()

    # Extract JSON object block
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in LLM output.")

    block = text[start:end+1].strip()

    # 1) strict JSON
    try:
        return json.loads(block)
    except Exception:
        pass

    # 2) json5 (tolerant)
    try:
        return json5.loads(block)
    except Exception:
        pass

    # 3) cleanup then try again
    cleaned = block

    # remove trailing commas before } or ]
    cleaned = re.sub(r",\s*([}\]])", r"\1", cleaned)

    # ensure property names use double quotes if they look like: key: "value"
    cleaned = re.sub(r"(?m)^\s*([A-Za-z_][A-Za-z0-9_]*)\s*:", r'"\1":', cleaned)

    # replace single quotes with double quotes (best-effort)
    cleaned = cleaned.replace("'", '"')

    # try again
    try:
        return json.loads(cleaned)
    except Exception as e:
        # Write bad output for debugging
        Path("outputs").mkdir(exist_ok=True)
        Path("outputs/last_bad_llm_output.txt").write_text(text, encoding="utf-8")
        Path("outputs/last_bad_llm_json_block.txt").write_text(block, encoding="utf-8")
        Path("outputs/last_bad_llm_json_cleaned.txt").write_text(cleaned, encoding="utf-8")
        raise ValueError(f"JSON parse failed even after repair: {e}")

def main():
    strategies = load_jsonl(Path("data/processed/strategies.jsonl"))
    strat_by_id = index_by_id(strategies, "strategy_id")

    # Choose mapping file (use goal filtered if you created it)
    mapping_path = Path("outputs/mapping_topk_explained.csv")
    if not mapping_path.exists():
        mapping_path = Path("outputs/mapping_topk.csv")

    df = pd.read_csv(mapping_path)
    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)

    all_suggestions = []

    for sid in sorted(df["strategy_id"].unique()):
        s = strat_by_id[sid]
        rows = df[df["strategy_id"] == sid].sort_values("rank")

        actions_block = build_actions_block(rows)
        prompt = PROMPT_TEMPLATE.format(
            strategy_id=sid,
            strategy_text=s["text"][:2500],  # keep within context
            actions_block=actions_block[:6000]
        )

        print(f"\n[LLM] Generating suggestions for {sid} ...")
        try:
            resp = llm_complete(prompt)
            sug = safe_json_parse(resp)
            sug["strategy_id"] = sid
            all_suggestions.append(sug)

            Path("outputs").mkdir(exist_ok=True)
            Path("outputs/suggestions_checkpoint.json").write_text(
                json.dumps(all_suggestions, indent=2),
                encoding="utf-8"
            )

        except Exception as e:
            print(f"[WARN] Failed for {sid}: {e}")
            continue

        # Minimal sanity fields
        sug["strategy_id"] = sid
        all_suggestions.append(sug)

    with open(out_dir / "suggestions.json", "w", encoding="utf-8") as f:
        json.dump(all_suggestions, f, indent=2)

    print("\n[OK] Wrote outputs/suggestions.json")

if __name__ == "__main__":
    main()
