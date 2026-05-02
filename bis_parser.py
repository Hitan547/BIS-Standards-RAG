"""
bis_parser.py
Parses BIS SP 21 (Summaries of Indian Standards for Building Materials) PDF.
Extracts each IS standard as a structured record.
Each record = one "chunk" for retrieval.

FIXES vs original:
1. Block splitting now uses "SUMMARY OF" as the delimiter instead of IS codes
   at line start — IS codes appear constantly as cross-references inside other
   standards' text, causing hundreds of false splits.
2. IS code extraction now captures Part numbers: IS 1489 (Part 2):1991 instead
   of just IS 1489:1991. Without this, PUB-04 and PUB-07 score zero.
3. Title extraction rewritten for SP 21 format: title follows the IS code on
   the same line (and may wrap to the next line).
4. Duplicate IS code blocks are MERGED (raw_text appended) instead of dropped.
"""

import re
import json
import pdfplumber
from pathlib import Path
from config import BIS_CATEGORIES, STANDARDS_PATH

# ── Regex patterns ──

# Matches: IS 269 : 1989 / IS 1489 (PART1) : 1991 / IS 2185 (PART 2) : 1983
# Groups: (1) number, (2) optional part number, (3) optional year
IS_CODE_RE = re.compile(
    r"IS\s+(\d{3,5})"                   # IS followed by 3-5 digit number
    r"\s*(?:\(\s*PART\s*(\d+)\s*\))?"   # optional (PART N) — case handled by flag
    r"\s*(?::\s*(\d{4}))?",             # optional :YYYY
    re.IGNORECASE,
)

YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")


def _clean(text: str) -> str:
    """Remove excessive whitespace and strip."""
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\.{3,}", "...", text)
    return text.strip()


def _extract_is_code(text: str) -> str:
    """
    Extract IS code string from text, INCLUDING Part numbers.

    Examples:
      'IS 269 : 1989'            → 'IS 269:1989'
      'IS 1489 (PART1) : 1991'   → 'IS 1489 (Part 1):1991'
      'IS 2185 (PART 2) : 1983'  → 'IS 2185 (Part 2):1983'
      'IS 383 : 1970'            → 'IS 383:1970'

    The stored format normalises identically to the expected_standards format
    used in the eval script (which strips all spaces and lowercases).
    e.g. 'IS 1489 (Part 2):1991' → 'is1489(part2):1991' ✓
    """
    match = IS_CODE_RE.search(text)
    if not match:
        return ""

    number   = match.group(1)
    part_num = match.group(2)   # None if no Part
    year     = match.group(3)   # None if no year on same token

    if not year:
        # Try finding a year in the nearby window
        window = text[max(0, match.start() - 30): match.end() + 40]
        ym = YEAR_RE.search(window)
        if ym:
            year = ym.group()

    # Build the standard number in a consistent format
    if part_num:
        code = f"IS {number} (Part {part_num})"
    else:
        code = f"IS {number}"

    if year:
        code += f":{year}"

    return code


def _detect_category(text: str) -> str:
    """Detect which building material category this standard belongs to."""
    text_lower = text.lower()
    category_keywords = {
        "Cement":        ["cement", "opc", "ppc", "fly ash cement", "portland"],
        "Steel":         ["steel", "rebar", "tmt", "bar", "wire rod", "structural steel", "iron"],
        "Concrete":      ["concrete", "rcc", "pcc", "ready mix", "precast", "reinforced"],
        "Aggregates":    ["aggregate", "sand", "gravel", "coarse", "fine aggregate", "crushed stone"],
        "Bricks":        ["brick", "block", "masonry", "fly ash brick", "clay brick"],
        "Tiles":         ["tile", "flooring", "ceramic", "vitrified", "mosaic"],
        "Glass":         ["glass", "glazing", "window glass"],
        "Timber":        ["timber", "wood", "plywood", "particle board", "hardboard"],
        "Paint":         ["paint", "coating", "primer", "distemper", "enamel", "varnish"],
        "Lime":          ["lime", "quicklime", "slaked lime", "hydraulic lime"],
        "Gypsum":        ["gypsum", "plaster of paris"],
        "Waterproofing": ["waterproof", "bitumen", "bituminous", "damp proof", "sealant"],
        "Insulation":    ["insulation", "thermal", "acoustic", "mineral wool", "fibre"],
        "Asbestos":      ["asbestos", "corrugated sheet", "roofing sheet"],
    }
    for category, keywords in category_keywords.items():
        if any(kw in text_lower for kw in keywords):
            return category
    return "General"


def _extract_title_from_header(block: str) -> str:
    """
    Extract the title from an SP 21 block.

    SP 21 format:
        IS 383 : 1970 COARSE AND FINE AGGREGATES FROM NATURAL
        SOURCES FOR CONCRETE
        (Second Revision)

    The title follows the IS code on the same line and may continue
    on the very next non-empty line before the revision note.
    """
    match = IS_CODE_RE.search(block)
    if not match:
        return ""

    after = block[match.end():].strip()
    after = re.sub(r"^[\s:\-–]+", "", after)

    lines = after.split("\n")
    title_parts = []
    for line in lines[:3]:
        line = line.strip()
        if not line:
            break
        # Stop at revision notes or numbered sections
        if re.match(r"^\(\s*[A-Za-z]", line) and "PART" not in line.upper():
            break
        if re.match(r"^\d+\.", line):
            break
        title_parts.append(line)

    title = " ".join(title_parts)
    # Remove trailing revision notes
    title = re.sub(r"\s*\([^)]*[Rr]evision[^)]*\)\s*$", "", title).strip()
    return _clean(title[:200])


def _parse_standard_block(block: str) -> dict | None:
    """
    Parse a single 'SUMMARY OF' block into a structured record.
    Returns None if the block doesn't contain a valid IS standard.
    """
    block_clean = _clean(block)
    if len(block_clean) < 50:
        return None

    is_code = _extract_is_code(block_clean)
    if not is_code:
        return None

    title    = _extract_title_from_header(block_clean)
    category = _detect_category(block_clean)

    # Extract scope
    scope = ""
    scope_match = re.search(
        r"(?:Scope|This standard|This specification|Covers?)[:\s]+(.{20,500}?)(?=\n\n|\.\s+\d+\.|\.$)",
        block_clean, re.IGNORECASE | re.DOTALL,
    )
    if scope_match:
        scope = _clean(scope_match.group(1).replace("\n", " "))[:400]

    # Extract key requirements
    req_matches = re.findall(
        r"(?:(?:\d+[\.\)]\d*\s+)|(?:[-•]\s+))([A-Z][^.\n]{10,150})",
        block_clean,
    )
    requirements = [_clean(r) for r in req_matches[:6]]

    full_text = _clean(f"{is_code} {title}. {scope} " + " ".join(requirements))

    return {
        "standard_number":  is_code,
        "title":            title,
        "category":         category,
        "scope":            scope,
        "key_requirements": requirements,
        "full_text":        full_text,
        "raw_text":         block_clean,
    }


def _split_into_blocks(full_text: str) -> list[str]:
    """
    Split the full PDF text into per-standard blocks.

    PRIMARY: Split on 'SUMMARY OF' — the explicit section header SP 21
    uses before every standard entry. Far more reliable than IS codes at
    line start because IS codes appear constantly as references in body text.

    FALLBACK: If fewer than 10 blocks found (unexpected PDF layout), falls
    back to IS-code-at-line-start splitting.
    """
    parts = re.split(r"SUMMARY\s+OF\s*\n?", full_text, flags=re.IGNORECASE)
    blocks = [b for b in parts if IS_CODE_RE.search(b) and len(b.strip()) > 100]

    if len(blocks) < 10:
        print("  WARNING: 'SUMMARY OF' split found few blocks — falling back to IS-code split")
        pattern = re.compile(r"(?=(?:^|\n)\s*IS\s+\d{3,5})", re.MULTILINE)
        fallback = pattern.split(full_text)
        blocks = [b for b in fallback if len(b.strip()) > 100]

    return blocks


def parse_sp21_pdf(pdf_path: str) -> list[dict]:
    """
    Main entry point. Parse the SP 21 PDF and return list of standard records.

    Args:
        pdf_path: Path to SP 21 PDF file

    Returns:
        List of dicts, each representing one IS standard
    """
    print(f"\n=== Parsing SP 21 PDF: {pdf_path} ===")
    all_text_parts = []

    with pdfplumber.open(pdf_path) as pdf:
        total = len(pdf.pages)
        print(f"Total pages: {total}")

        for i, page in enumerate(pdf.pages):
            if i % 50 == 0:
                print(f"  Processing page {i + 1}/{total}...")

            text = page.extract_text(x_tolerance=3, y_tolerance=3) or ""

            tables = page.extract_tables()
            for table in tables:
                for row in table:
                    if row:
                        row_text = " | ".join(str(cell or "").strip() for cell in row)
                        text += "\n" + row_text

            all_text_parts.append(text)

    full_text = "\n".join(all_text_parts)
    print(f"Extracted {len(full_text):,} characters total")

    blocks = _split_into_blocks(full_text)
    print(f"Found {len(blocks)} potential standard blocks")

    standards   = []
    seen_codes: dict[str, int] = {}

    for block in blocks:
        record = _parse_standard_block(block)
        if record is None:
            continue

        code = record["standard_number"]

        if code not in seen_codes:
            seen_codes[code] = len(standards)
            standards.append(record)
        else:
            # MERGE duplicate blocks instead of dropping
            existing = standards[seen_codes[code]]
            existing["raw_text"]  += "\n\n" + record["raw_text"]
            existing["full_text"] += " " + record["full_text"][:200]

            existing_set = set(existing["key_requirements"])
            for req in record["key_requirements"]:
                if req not in existing_set:
                    existing["key_requirements"].append(req)
                    existing_set.add(req)

            if not existing["scope"] and record["scope"]:
                existing["scope"] = record["scope"]
            if not existing["title"] and record["title"]:
                existing["title"] = record["title"]

    print(f"\n✅ Successfully parsed {len(standards)} unique IS standards")
    print(f"   ({len(blocks) - len(standards)} duplicate blocks merged)")

    from collections import Counter
    cat_counts = Counter(s["category"] for s in standards)
    print("\nCategory breakdown:")
    for cat, count in cat_counts.most_common():
        print(f"  {cat}: {count}")

    part_stds = [s for s in standards if "(Part" in s["standard_number"]]
    print(f"\n  Standards with Part numbers: {len(part_stds)}")
    for s in part_stds[:8]:
        print(f"    {s['standard_number']} — {s['title'][:60]}")

    return standards


def save_standards(standards: list[dict], path: str = None):
    """Save parsed standards to JSON."""
    if path is None:
        path = STANDARDS_PATH
    Path(path).parent.mkdir(exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(standards, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(standards)} standards to {path}")


def load_standards(path: str = None) -> list[dict]:
    """Load previously parsed standards from JSON."""
    if path is None:
        path = STANDARDS_PATH
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    import sys
    pdf_path  = sys.argv[1] if len(sys.argv) > 1 else "docs/sp21.pdf"
    standards = parse_sp21_pdf(pdf_path)
    save_standards(standards)

    print("\n--- SAMPLE STANDARDS ---")
    for s in standards[:3]:
        print(f"Code:     {s['standard_number']}")
        print(f"Title:    {s['title']}")
        print(f"Category: {s['category']}")
        print(f"Scope:    {s['scope'][:150]}")
        print(f"Raw len:  {len(s['raw_text'])} chars")
        print()