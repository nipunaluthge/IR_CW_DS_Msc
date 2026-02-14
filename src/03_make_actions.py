from pathlib import Path
import json
import re

# Example row ending pattern:
# "... some action text ... 3 3 3 3 1.5 Development Assessment"
ROW_END_RE = re.compile(r"\s(\d)\s+(\d)?\s*(\d)?\s*(\d)?\s+(\d+\.\d+)\s+(.+)$")

def load_pages(path: Path):
    pages = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            pages.append(json.loads(line))
    return pages

def guess_service_title(lines):
    """
    Try to identify service heading from page text.
    Usually appears as a standalone title near the top (after page header).
    """
    candidates = []
    for ln in lines[:25]:
        ln = ln.strip()
        if not ln:
            continue
        # Skip obvious headers
        if "Wollongong City Council" in ln or "Delivery Program" in ln:
            continue
        if ln.lower().startswith("actions"):
            continue
        # Short-ish title case line
        if 3 <= len(ln) <= 60 and not any(ch.isdigit() for ch in ln):
            candidates.append(ln)

    return candidates[0] if candidates else None

def parse_actions_from_page(page_text, page_index, source_doc):
    lines = [ln.rstrip() for ln in (page_text or "").splitlines()]
    if not lines:
        return []

    if not any("Actions" in ln for ln in lines[:10] + lines[-10:]) and "Actions" not in page_text:
        # Still allow pages with action rows even if "Actions" not obvious, but keep it strict:
        pass

    service = guess_service_title(lines) or "Unknown Service"

    actions = []
    buffer = []

    for ln in lines:
        ln = ln.strip()
        if not ln:
            continue

        # Skip table headers
        if ln.startswith("Actions") or "Operational Plan" in ln or ln.startswith("CSP") or ln.startswith("Delivery"):
            continue

        m = ROW_END_RE.search(ln)
        if m:
            csp_ref = m.group(5)       # e.g., "1.5"
            delivery_stream = m.group(6).strip()
            goal_no = int(csp_ref.split(".")[0])

            # Everything accumulated + anything before the numbers is action description
            # Remove the matched tail from ln
            action_desc_tail_removed = ROW_END_RE.sub("", ln).strip()

            desc_parts = buffer + ([action_desc_tail_removed] if action_desc_tail_removed else [])
            desc = " ".join(desc_parts).strip()
            desc = re.sub(r"\s+", " ", desc)

            if len(desc) >= 15:  # avoid garbage rows
                actions.append({
                    "goal_no": goal_no,
                    "csp_ref": csp_ref,
                    "service": service,
                    "delivery_stream": delivery_stream,
                    "text": desc,
                    "source_doc": source_doc,
                    "page_index": page_index
                })

            buffer = []
        else:
            # accumulate multi-line action descriptions
            # skip obvious noise lines:
            if ln in {"Supporting Documents", "Finances (000’S)", "Revenue", "Expense", "Net"}:
                continue
            buffer.append(ln)

    return actions

def write_jsonl(records, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    pages = load_pages(Path("data/processed/action_pages.jsonl"))

    raw_actions = []
    for p in pages:
        raw_actions.extend(
            parse_actions_from_page(
                p["text"], p["page_index"], p["doc"]
            )
        )

    # Assign A-IDs
    actions = []
    for i, a in enumerate(raw_actions, start=1):
        a["action_id"] = f"A{i:05d}"
        actions.append(a)

    out_path = Path("data/processed/actions.jsonl")
    write_jsonl(actions, out_path)

    print(f"[OK] actions.jsonl created with {len(actions)} actions -> {out_path}")
    # Print a few samples
    for ex in actions[:8]:
        print(ex["action_id"], ex["csp_ref"], ex["service"], "-", ex["text"][:90])
