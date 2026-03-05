#!/usr/bin/env python3
"""
Palme OCR — Main CLI entrypoint.
Processes all PDFs in an input directory and writes .txt and .docx files to an output directory.
"""

import argparse
import glob
import os
import sys
import time


from ocr_engine import OCREngine
from pdf_processor import pdf_to_images
from preprocess import resize_for_ocr
from postprocess import extract_plain_text
from docx_writer import create_docx_from_pages
from preprocess import enhance_image


def process_pdf(pdf_path: str, engine: OCREngine, output_dir: str) -> None:
    """Process a single PDF file and write the extracted text."""
    basename = os.path.splitext(os.path.basename(pdf_path))[0]
    txt_path = os.path.join(output_dir, f"{basename}.txt")
    docx_path = os.path.join(output_dir, f"{basename}.docx")

    print(f"  [1/4] Converting PDF to images...")
    images = pdf_to_images(pdf_path, dpi=120)
    print(f"        → {len(images)} page(s)")

    all_pages_text = []
    all_pages_raw = []  # Keep raw model output for docx generation

    for i, img in enumerate(images):
        print(f"  [2/4] Pre-processing page {i + 1}/{len(images)}...")
        enhanced = enhance_image(img)
        enhanced = resize_for_ocr(enhanced)
        print(f"        Image size: {enhanced.size}")

        print(f"  [3/4] Running OCR on page {i + 1}/{len(images)}...")
        raw_output = engine.extract(enhanced)
        all_pages_raw.append(raw_output)

        raw_path = os.path.join(output_dir, f"{basename}_page{i+1}.raw.json")
        with open(raw_path, "w", encoding="utf-8") as f:
            f.write(raw_output)
        print(f"  ✓ Saved raw output → {raw_path}")

        print(f"  [4/4] Post-processing page {i + 1}/{len(images)}...")
        clean_text = extract_plain_text(raw_output)
        all_pages_text.append(clean_text)

    # Save plain text
    full_text = "\n\n".join(all_pages_text)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(full_text)
    print(f"  ✓ Saved text → {txt_path}")

    # Save formatted docx
    try:
        create_docx_from_pages(
            all_pages_raw,
            docx_path,
            source_filename=os.path.basename(pdf_path)
        )
        print(f"  ✓ Saved docx → {docx_path}")
    except Exception as e:
        print(f"  ⚠ Could not create docx: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Palme OCR: Offline document digitization pipeline"
    )
    parser.add_argument(
        "--input", required=True, help="Path to directory containing PDF files"
    )
    parser.add_argument(
        "--output", required=True, help="Path to output directory for .txt files"
    )
    parser.add_argument(
        "--model-path",
        default="/app/weights/DotsOCR",
        help="Path to dots.ocr model weights (default: /app/weights/DotsOCR)",
    )
    args = parser.parse_args()

    # Validate input directory
    if not os.path.isdir(args.input):
        print(f"Error: Input directory '{args.input}' does not exist.")
        sys.exit(1)

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Find all PDFs
    pdf_files = sorted(glob.glob(os.path.join(args.input, "*.pdf")))
    if not pdf_files:
        pdf_files = sorted(glob.glob(os.path.join(args.input, "*.PDF")))
    if not pdf_files:
        print(f"No PDF files found in '{args.input}'")
        sys.exit(1)

    print(f"Found {len(pdf_files)} PDF file(s) in '{args.input}'")
    print(f"Output directory: '{args.output}'")
    print()

    # Load model once
    print("Loading dots.ocr model...")
    start = time.time()
    engine = OCREngine(args.model_path)
    print(f"Model loaded in {time.time() - start:.1f}s\n")

    # Process each PDF
    total_start = time.time()
    for idx, pdf_path in enumerate(pdf_files, 1):
        print(f"[{idx}/{len(pdf_files)}] Processing: {os.path.basename(pdf_path)}")
        page_start = time.time()
        try:
            process_pdf(pdf_path, engine, args.output)
        except Exception as e:
            print(f"  ✗ Error processing {os.path.basename(pdf_path)}: {e}")
        elapsed = time.time() - page_start
        print(f"  Time: {elapsed:.1f}s\n")

    total_elapsed = time.time() - total_start
    print(f"Done! Processed {len(pdf_files)} files in {total_elapsed:.1f}s")


if __name__ == "__main__":
    main()