"""Inventory internal cross-references in the thesis .docx.

Finds every "Section X.Y", "RQn", "Table n" and "Figure n" reference and reports
the paragraph index it occurs in, so a section reorder can be remapped safely.

Usage:
    python scripts/audit_crossrefs.py "docs/write-ups/Thesis_Assembled (1).docx"
"""
import re
import sys
from collections import defaultdict

from docx import Document

PATTERNS = {
    "section": re.compile(r"Section[s]? \d+(?:\.\d+)*(?:\s*[–-]\s*\d+(?:\.\d+)*)?"),
    "rq": re.compile(r"\bRQ\d\b"),
    "table": re.compile(r"\bTables? \d+[a-z]?\b"),
    "figure": re.compile(r"\bFigures? \d+[a-z]?\b"),
}


def main() -> None:
    doc = Document(sys.argv[1])

    for kind, pattern in PATTERNS.items():
        found: dict[str, list[int]] = defaultdict(list)
        for i, par in enumerate(doc.paragraphs):
            for match in pattern.findall(par.text):
                found[match].append(i)

        print(f"=== {kind.upper()} references ({len(found)} distinct) ===")
        for ref in sorted(found, key=lambda r: (len(r), r)):
            print(f"  {ref:24s} -> paragraphs {found[ref]}")
        print()


if __name__ == "__main__":
    main()
