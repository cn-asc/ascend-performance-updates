#!/usr/bin/env python3
"""
Parse 2025 Monthly Investment Update [Running].docx and create a Notion page per investment
under the Family Investment Updates parent. Each page gets update write-ups with the most
recent on top; new updates can be appended to the top later.

Usage:
  pip install python-dotenv requests  # optional: python-docx for docx parsing; else stdlib zip/xml fallback is used
  Set NOTION_API_KEY and NOTION_PARENT_PAGE_ID in .env (or env).
  NOTION_PARENT_PAGE_ID = the UUID of https://www.notion.so/Family-Investment-Updates-2ff3ea66f702809f9e37d20da4435395
  (use the ID from the page URL; add dashes if needed: 2ff3ea66-f702-809f-9e37-d20da4435395)

  python notion_sync_2025_updates.py

Then connect your integration to the parent page in Notion (Add connections).
"""

import os
import re
import zipfile
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

try:
    from docx import Document
    _HAS_DOCX = True
except ImportError:
    _HAS_DOCX = False

import requests

# Word XML namespaces in document.xml
W_NS = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}

NOTION_VERSION = "2022-06-28"
NOTION_API = "https://api.notion.com/v1"
# Parent page ID = the ID in the URL *before* "?" (not the view ID after "?v=").
# Example: https://www.notion.so/2ff3ea66f70280e89c3ddde8a4dc3694?v=... -> use 2ff3ea66f70280e89c3ddde8a4dc3694
NOTION_PARENT_PAGE_ID = os.getenv("NOTION_PARENT_PAGE_ID", "2ff3ea66f70280e89c3ddde8a4dc3694").replace("-", "")


def _notion_headers():
    key = os.getenv("NOTION_API_KEY") or os.getenv("NOTION_KEY")
    if not key:
        raise ValueError("Set NOTION_API_KEY or NOTION_KEY in .env")
    return {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "Notion-Version": NOTION_VERSION,
    }


def _resolve_parent(parent_id: str) -> tuple[str, str, str | None]:
    """
    Detect whether the ID is a page or a database (Notion uses the same URL shape for both).
    Returns (parent_type, parent_id, title_property_name).
    - For a page: ("page", id, None) — we use standard "title" when creating.
    - For a database: ("database", id, "Name") — we use the DB's title column (e.g. "Name").
    """
    parent_id = parent_id.replace("-", "")
    # Try as page first
    r = requests.get(
        f"{NOTION_API}/pages/{parent_id}",
        headers=_notion_headers(),
        timeout=15,
    )
    if r.status_code == 200:
        return ("page", parent_id, None)
    if r.status_code != 404:
        r.raise_for_status()
    # Not a page; try as database
    r2 = requests.get(
        f"{NOTION_API}/databases/{parent_id}",
        headers=_notion_headers(),
        timeout=15,
    )
    if r2.status_code == 200:
        data = r2.json()
        props = data.get("properties") or {}
        for name, config in props.items():
            if isinstance(config, dict) and config.get("type") == "title":
                return ("database", parent_id, name)
        return ("database", parent_id, "Name")
    if r2.status_code == 404:
        raise ValueError(
            f"Parent ID {parent_id!r} is neither a page nor a database (404). "
            "Check NOTION_PARENT_PAGE_ID and that the page/database is shared with your integration."
        )
    r2.raise_for_status()
    raise RuntimeError("Unexpected response from Notion API")


# ---- Parsing docx ----

MONTH_NAMES = [
    "JANUARY", "FEBRUARY", "MARCH", "APRIL", "MAY", "JUNE",
    "JULY", "AUGUST", "SEPTEMBER", "OCTOBER", "NOVEMBER", "DECEMBER",
]
# Only treat a line as the section date if the line is exactly "MONTH 2025" (no extra text)
MONTH_PATTERN = re.compile(
    r"^(" + "|".join(MONTH_NAMES) + r")\s+2025\s*$",
    re.I,
)
# Investment name optionally followed by " - " then [As Expected] or [Outperforming] or [Underperforming]
INVESTMENT_HEADER_PATTERN = re.compile(
    r"^(.+?)\s*(?:-\s*)?\[(?:As Expected|Outperforming|Underperforming)\](?:\s*\(Add relevant content as needed\))?\s*$",
    re.I,
)
UNDERSCORE_LINE = re.compile(r"_{3,}")


def _full_text_from_docx_lib(path: str) -> str:
    """Requires: pip install python-docx"""
    doc = Document(path)
    return "\n".join(p.text for p in doc.paragraphs)


def _full_text_from_docx_zip(path: str) -> str:
    """Stdlib-only: read .docx as ZIP and extract text from word/document.xml."""
    with zipfile.ZipFile(path, "r") as z:
        with z.open("word/document.xml") as f:
            tree = ET.parse(f)
    root = tree.getroot()
    paragraphs = []
    for p in root.iter("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}p"):
        texts = []
        for t in p.iter("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t"):
            if t.text:
                texts.append(t.text)
            if t.tail:
                texts.append(t.tail)
        paragraphs.append("".join(texts))
    return "\n".join(paragraphs)


def _full_text_from_docx(path: str) -> str:
    if _HAS_DOCX:
        return _full_text_from_docx_lib(path)
    return _full_text_from_docx_zip(path)


def _parse_updates_by_investment(full_text: str) -> dict[str, list[tuple[str, str]]]:
    """
    Returns dict: investment_name -> list of (date_label, content) with NEWEST FIRST.
    date_label e.g. "November 2025"
    """
    # Split by long underscore lines into sections
    sections = UNDERSCORE_LINE.split(full_text)
    # First section is often title/template; skip or use to detect structure
    by_investment: dict[str, list[tuple[str, str]]] = defaultdict(list)

    for section in sections:
        section = section.strip()
        if not section or len(section) < 50:
            continue
        lines = section.split("\n")
        current_date: str | None = None
        current_investment: str | None = None
        current_content: list[str] = []

        def flush():
            if current_investment and current_content:
                content = "\n\n".join(current_content).strip()
                if content and current_date:
                    by_investment[current_investment].append((current_date, content))

        for line in lines:
            line_stripped = line.strip()
            # Month 2025 header
            m = MONTH_PATTERN.search(line_stripped)
            if m:
                flush()
                current_date = line_stripped[: m.end()].strip().title()
                current_investment = None
                current_content = []
                continue
            # Investment header: "Name - [As Expected]"
            inv = INVESTMENT_HEADER_PATTERN.match(line_stripped)
            if inv:
                flush()
                current_investment = inv.group(1).strip()
                current_content = []
                continue
            if current_investment and line_stripped:
                current_content.append(line_stripped)

        flush()

    # Sort each investment's updates by date (newest first): parse "November 2025" -> (2025, 11)
    def _sort_key(item: tuple[str, str]) -> tuple[int, int]:
        date_label = item[0]
        try:
            for i, mon in enumerate(MONTH_NAMES, 1):
                if mon in date_label.upper():
                    return (2025, -i)  # negative so newer month first
        except Exception:
            pass
        return (0, 0)

    for inv in by_investment:
        by_investment[inv].sort(key=_sort_key)

    return dict(by_investment)


# ---- Notion API ----

def _rich_text_chunks(text: str, max_chars: int = 2000) -> list[dict]:
    """Notion rich_text has ~2000 char limit per element. Split into chunks."""
    out = []
    text = (text or "").replace("\x00", "")
    while text:
        chunk = text[:max_chars]
        if len(text) > max_chars:
            # Break at last newline to avoid mid-word
            last_nl = chunk.rfind("\n")
            if last_nl > max_chars // 2:
                chunk = chunk[: last_nl + 1]
                text = text[last_nl + 1 :]
            else:
                text = text[max_chars:]
        else:
            text = ""
        out.append({"type": "text", "text": {"content": chunk, "link": None}})
    return out


def _paragraph_block(text: str) -> dict:
    return {
        "type": "paragraph",
        "paragraph": {
            "rich_text": _rich_text_chunks(text),
            "color": "default",
        },
    }


def _heading2_block(text: str) -> dict:
    return {
        "type": "heading_2",
        "heading_2": {
            "rich_text": [{"type": "text", "text": {"content": text[:2000], "link": None}}],
            "color": "default",
            "is_toggleable": False,
        },
    }


def _divider_block() -> dict:
    return {"type": "divider", "divider": {}}


def create_notion_page_for_investment(
    investment_name: str,
    updates: list[tuple[str, str]],
    parent_type: str,
    parent_id: str,
    database_title_prop: str | None,
    dry_run: bool = False,
) -> str | None:
    """
    Create one Notion page under parent (page or database) with title investment_name.
    Append blocks: for each update (newest first), add heading_2 (date), then paragraph(s) (content), then divider.
    Returns new page id or None.
    """
    parent_id = parent_id.replace("-", "")
    if parent_type == "page":
        parent = {"type": "page_id", "page_id": parent_id}
        title_prop_name = "title"
    else:
        parent = {"type": "database_id", "database_id": parent_id}
        title_prop_name = database_title_prop or "Name"
    title_block = [{"type": "text", "text": {"content": investment_name[:2000], "link": None}}]
    payload = {
        "parent": parent,
        "properties": {
            title_prop_name: {"title": title_block},
        },
        "children": [],
    }

    for date_label, content in updates:
        payload["children"].append(_heading2_block(date_label))
        payload["children"].append(_paragraph_block(content))
        payload["children"].append(_divider_block())

    if dry_run:
        print(f"[DRY RUN] Would create page: {investment_name} with {len(updates)} update(s)")
        return None

    resp = requests.post(
        f"{NOTION_API}/pages",
        json=payload,
        headers=_notion_headers(),
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    return data.get("id")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Sync 2025 investment updates from docx to Notion")
    parser.add_argument("--docx", default="2025 Monthly Investment Update [Running].docx", help="Path to the docx file")
    parser.add_argument("--dry-run", action="store_true", help="Parse and print only; do not create Notion pages")
    parser.add_argument("--parent", default=NOTION_PARENT_PAGE_ID, help="Notion parent page ID")
    args = parser.parse_args()

    path = Path(args.docx)
    if not path.exists():
        print(f"File not found: {path}")
        return 1

    print(f"Parsing {path}...")
    full_text = _full_text_from_docx(str(path))
    by_investment = _parse_updates_by_investment(full_text)

    print(f"Found {len(by_investment)} investments with updates.")
    for name, updates in sorted(by_investment.items(), key=lambda x: x[0].lower()):
        print(f"  - {name}: {len(updates)} update(s) (newest: {updates[0][0] if updates else 'n/a'})")

    if args.dry_run:
        print("\nDry run. Run without --dry-run to create Notion pages.")
        return 0

    parent_id_clean = args.parent.replace("-", "")
    print(f"\nResolving parent {parent_id_clean} (page or database)...")
    try:
        parent_type, parent_id, db_title_prop = _resolve_parent(args.parent)
        print(f"  → Detected: {parent_type}" + (f" (title property: {db_title_prop!r})" if db_title_prop else ""))
    except ValueError as e:
        print(f"Error: {e}")
        return 1

    print("\nCreating Notion pages...")
    for investment_name, updates in sorted(by_investment.items(), key=lambda x: x[0].lower()):
        try:
            page_id = create_notion_page_for_investment(
                investment_name,
                updates,
                parent_type=parent_type,
                parent_id=parent_id,
                database_title_prop=db_title_prop,
                dry_run=False,
            )
            if page_id:
                # Notion page URL format
                print(f"  Created: {investment_name} -> https://www.notion.so/{page_id.replace('-', '')}")
        except requests.HTTPError as e:
            err = e.response.text
            if e.response.status_code == 404 and "object_not_found" in err:
                print(f"  Failed {investment_name}: 404 – parent not found or not shared with your integration.")
                print("     → Open the parent page/database in Notion → ⋯ menu → Add connections → select your integration.")
            else:
                print(f"  Failed {investment_name}: {e.response.status_code} {err[:200]}")
        except Exception as e:
            print(f"  Failed {investment_name}: {e}")

    print("Done.")
    return 0


if __name__ == "__main__":
    exit(main())
