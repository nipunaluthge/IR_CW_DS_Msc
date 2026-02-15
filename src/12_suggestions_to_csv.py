from pathlib import Path
import json
import pandas as pd

def main():
    p = Path("outputs/suggestions.json")
    data = json.loads(p.read_text(encoding="utf-8"))

    rows = []
    for item in data:
        sid = item["strategy_id"]

        for rec in item.get("recommended_actions", []):
            rows.append({
                "strategy_id": sid,
                "title": rec.get("title", ""),
                "description": rec.get("description", ""),
                "owner_role": rec.get("owner_role", ""),
                "timeframe": rec.get("timeframe", ""),
                "KPI": rec.get("KPI", ""),
                "priority": rec.get("priority", "")
            })

    df = pd.DataFrame(rows)
    out = Path("outputs/improvement_actions.csv")
    df.to_csv(out, index=False)
    print(f"[OK] Wrote {out} with {len(df)} rows")

if __name__ == "__main__":
    main()
