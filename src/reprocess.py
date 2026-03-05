#!/usr/bin/env python3
"""
Reprocess — Regenerate .txt and .docx from saved raw model outputs.
Skips the slow OCR step entirely.

Usage:
    python3 src/reprocess.py --output ./results
"""

import argparse
import glob
import os
import re
from collections import defaultdict

from postprocess import extract_plain_text
from docx_writer import create_docx_from_pages


def main():
    parser = argparse.ArgumentParser(description="Reprocess raw OCR outputs")
    parser.add_argument("--output", required=True, help="Directory containing .raw.json files")
    args = parser.parse_args()

    # Find all raw files and group by PDF name
    raw_files = sorted(glob.glob(os.path.join(args.output, "*.raw.json")))
    if not raw_files:
        print(f"No .raw.json files found in '{args.output}'")
        print("Run the full pipeline first with raw output saving enabled.")
        return

    # Group by document: "docname_page1.raw.json" -> "docname"
    docs = defaultdict(list)
    for raw_path in raw_files:
        fname = os.path.basename(raw_path)
        match = re.match(r"(.+)_page(\d+)\.raw\.json$", fname)
        if match:
            doc_name = match.group(1)
            page_num = int(match.group(2))
            docs[doc_name].append((page_num, raw_path))

    print(f"Found {len(docs)} document(s) to reprocess\n")

    for doc_name, pages in docs.items():
        pages.sort(key=lambda x: x[0])
        print(f"Reprocessing: {doc_name} ({len(pages)} pages)")

        # Read raw outputs
        pages_raw = []
        for page_num, raw_path in pages:
            with open(raw_path, "r", encoding="utf-8") as f:
                pages_raw.append(f.read())

        # Generate .txt
        txt_path = os.path.join(args.output, f"{doc_name}.txt")
        texts = [extract_plain_text(raw) for raw in pages_raw]
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("\n\n".join(texts))
        print(f"  ✓ {txt_path}")

        # Generate .docx
        docx_path = os.path.join(args.output, f"{doc_name}.docx")
        try:
            create_docx_from_pages(pages_raw, docx_path, source_filename=f"{doc_name}.pdf")
            print(f"  ✓ {docx_path}")
        except Exception as e:
            print(f"  ⚠ docx failed: {e}")

    print("\nDone!")


if __name__ == "__main__":
    main()