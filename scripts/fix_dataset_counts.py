"""Correct the dataset counts in the thesis .docx.

Three defects, all in the same place:

  1. Table 2 reported BUTTER-E as "63,527 runs" -- the dataset's published
     figure, not the number of runs this work actually models. It contradicted
     Section 5.1, which reports 37,055.
  2. The Section 3.2 opening sentence begins mid-clause ("and EC-NAS
     (Bakhtiarifard et al., 2024), whose..."), because an earlier edit deleted
     the clause introducing BUTTER-E and the sentence opening with it.
  3. The Table 2 row label "Number of architectures" was carrying run counts.

Reads and writes the same path; the document is loaded fully into memory
before saving, and a pre-change copy exists in docs/write-ups/_backup/.

Usage:
    python scripts/fix_dataset_counts.py "docs/write-ups/Thesis_Assembled_v2.docx"
"""
from __future__ import annotations

import sys

from docx import Document
from docx.text.paragraph import Paragraph

# ---- replacement prose ----------------------------------------------------

DATASETS_OPENING = (
    "Two datasets jointly provide the empirical foundation for this work. BUTTER-E (Tripp et al., "
    "2024) supplies multilayer perceptron (MLP) architectures of varying depth, width and shape; of "
    "the 64,988 runs recorded in its metadata, 37,055 carry matched energy measurements and form the "
    "subset used throughout this work. EC-NAS (Bakhtiarifard et al., 2024) has a full architecture "
    "space spanning approximately 423,000 convolutional neural network (CNN) architectures drawn "
    "from the NAS-Bench-101 search space, of which 2,805 have directly measured training energy; the "
    "remainder are surrogate-model estimates and are not used in this work. Between them, these "
    "datasets provide the two architecture families and the scale of real measured data required to "
    "train and evaluate a cross-family predictor \u2014 a combination that, per the systematic "
    "review, does not exist in any single prior dataset."
)

TABLE_ROW_LABEL = "Training runs used"

BUTTER_E_CELL = "37,055 with matched energy measurements (of 64,988 runs collected)"

EC_NAS_CELL = (
    "2,805 directly measured, from a ~423,000-architecture search space whose remaining entries are "
    "surrogate estimates; 91 architectures per GPU in the 4V hardware benchmark"
)


def set_text(par: Paragraph, text: str) -> None:
    """Replace a paragraph's text, preserving the first run's formatting."""
    if not par.runs:
        par.add_run(text)
        return
    par.runs[0].text = text
    for run in par.runs[1:]:
        run.text = ""


def main() -> None:
    src, dst = sys.argv[1], sys.argv[2]
    doc = Document(src)
    log: list[str] = []

    # -- 1. Section 3.2 opening sentence ------------------------------------
    for par in doc.paragraphs:
        if par.text.strip().startswith("and EC-NAS (Bakhtiarifard"):
            set_text(par, DATASETS_OPENING)
            log.append("rewrote the Section 3.2 opening sentence")
            break
    else:
        raise SystemExit("could not find the Section 3.2 opening sentence")

    # -- 2. Table 2 "Number of architectures" row ---------------------------
    fixed = False
    for table in doc.tables:
        for row in table.rows:
            label = row.cells[0].text.strip()
            if label != "Number of architectures":
                continue
            values = [cell.text.strip() for cell in row.cells]
            if "63,527" not in " ".join(values):
                continue

            set_text(row.cells[0].paragraphs[0], TABLE_ROW_LABEL)
            set_text(row.cells[1].paragraphs[0], BUTTER_E_CELL)
            set_text(row.cells[2].paragraphs[0], EC_NAS_CELL)
            log.append(f"fixed Table 2 row {label!r}: {values[1]!r} -> {BUTTER_E_CELL!r}")
            fixed = True
            break

    if not fixed:
        raise SystemExit("could not find the Table 2 'Number of architectures' row")

    doc.save(dst)
    print("\n".join(log))

    # -- verification -------------------------------------------------------
    check = Document(dst)
    print("\n=== verification ===")
    for i, par in enumerate(check.paragraphs):
        if "63,527" in par.text:
            print(f"  !! stale count still at paragraph {i}")
    for i, par in enumerate(check.paragraphs):
        if par.text.strip().startswith("37,055") or "37,055 with matched" in par.text:
            print(f"  para {i}: {par.text[:200]}")
    for table in check.tables:
        for row in table.rows:
            if row.cells[0].text.strip() == TABLE_ROW_LABEL:
                for cell in row.cells:
                    print(f"  [{cell.text.strip()[:110]}]")
    print(f"\nsaved: {dst}")


if __name__ == "__main__":
    main()
