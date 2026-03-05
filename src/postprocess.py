"""
Post-processing — Convert dots.ocr structured JSON output to clean plain text.
"""

import json
import re
from typing import Optional


def parse_model_output(raw_output: str) -> Optional[list]:
    """Parse the JSON output from dots.ocr, handling common model quirks."""
    text = raw_output.strip()

    # Remove markdown code fences if present
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()

    # Try direct parse first
    try:
        parsed = json.loads(text)
        return _normalize_parsed(parsed)
    except json.JSONDecodeError:
        pass

    # Try to find JSON array in the text (model sometimes adds preamble/postamble)
    match = re.search(r'\[\s*\{.*\}\s*\]', text, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group())
            return _normalize_parsed(parsed)
        except json.JSONDecodeError:
            pass

    # Try fixing common JSON issues
    fixed = text
    # Fix trailing commas before ] or }
    fixed = re.sub(r',\s*([}\]])', r'\1', fixed)
    # Fix single quotes to double quotes
    fixed = fixed.replace("'", '"')
    try:
        parsed = json.loads(fixed)
        return _normalize_parsed(parsed)
    except json.JSONDecodeError:
        pass

    return None


def _normalize_parsed(parsed) -> Optional[list]:
    """Normalize parsed JSON into a list of elements."""
    if isinstance(parsed, list):
        return parsed
    elif isinstance(parsed, dict):
        for key in ["layout_dets", "layout", "elements", "results"]:
            if key in parsed:
                return parsed[key]
        if "text" in parsed:
            return [parsed]
    return None


def clean_html_table(html: str) -> str:
    """Convert an HTML table to simple text."""
    text = re.sub(r"<tr[^>]*>", "", html)
    text = re.sub(r"</tr>", "\n", text)
    text = re.sub(r"<t[dh][^>]*>", "", text)
    text = re.sub(r"</t[dh]>", "\t", text)
    text = re.sub(r"<[^>]+>", "", text)
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    return "\n".join(lines)


def extract_plain_text(raw_output: str) -> str:
    """Extract clean plain text from dots.ocr model output."""
    elements = parse_model_output(raw_output)

    if elements and isinstance(elements, list):
        text_parts = []
        for elem in elements:
            if not isinstance(elem, dict):
                continue

            category = elem.get("category", "").strip()
            text = elem.get("text", "").strip()

            if not text:
                continue

            # Skip headers/footers
            if category in ("Page-header", "Page-footer"):
                continue

            # Clean up markdown bold/italic markers from model output
            text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
            text = re.sub(r'\*(.+?)\*', r'\1', text)

            # Handle tables
            if category == "Table" and "<" in text:
                text = clean_html_table(text)

            # Replace \n with actual newlines if they're escaped
            text = text.replace("\\n", "\n")

            text_parts.append(text)

        if text_parts:
            return "\n\n".join(text_parts)

    # Fallback: clean up raw text
    fallback = raw_output.strip()

    # If it looks like it contains JSON, try to extract just the text fields
    if '"category"' in fallback and '"text"' in fallback:
        texts = re.findall(r'"text"\s*:\s*"((?:[^"\\]|\\.)*)"\s*[,}]', fallback)
        if texts:
            cleaned = []
            for t in texts:
                t = t.replace("\\n", "\n").replace('\\"', '"')
                # Skip if it looks like a category name
                if t in ("SÄPO", "PM", "2", "3") or len(t) < 3:
                    continue
                cleaned.append(t)
            if cleaned:
                return "\n\n".join(cleaned)

    # Last resort: strip JSON artifacts
    fallback = re.sub(r'\[\s*\{.*?\}\s*\]', '', fallback, flags=re.DOTALL)
    fallback = re.sub(r'[{}\[\]]', '', fallback)
    fallback = re.sub(r'"(?:bbox|category|text)"\s*:', '', fallback)
    lines = [line.strip() for line in fallback.split("\n") if line.strip()]
    return "\n".join(lines)