from pathlib import Path
import json
from pypdf import PdfReader

def extract_pdf_to_jsonl(pdf_path: Path, out_jsonl: Path):
    reader = PdfReader(str(pdf_path))
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    with out_jsonl.open("w", encoding="utf-8") as f:
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            rec = {
                "doc": pdf_path.name,
                "page_index": i,
                "text": text
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"[OK] Extracted {len(reader.pages)} pages -> {out_jsonl}")

if __name__ == "__main__":
    raw_dir = Path("data/raw")
    proc_dir = Path("data/processed")
    extract_pdf_to_jsonl(raw_dir / "strategic_plan.pdf", proc_dir / "strategic_pages.jsonl")
    extract_pdf_to_jsonl(raw_dir / "action_plan.pdf",    proc_dir / "action_pages.jsonl")