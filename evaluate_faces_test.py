#!/usr/bin/env python3
"""
evaluate_faces.py - score face identification against a labeled test set.

The test tree is laid out as:

    test/
      jack/                   expected: jack
      jim_jack/               expected: jim, jack
      unknown/                expected: (nobody enrolled)
      difficult/              expected parsed from each filename:
                              jack_back_of_head.JPG -> jack
                              jill_jack_distant.JPG -> jill, jack

Usage:
    python evaluate_faces.py "bigger test/test" --face-db faces-family.pkl
    python evaluate_faces.py "bigger test/test" --face-db faces-family.pkl --model cnn

Prints per-photo results, then precision/recall style totals.
No LLM is involved; this only exercises face detection + matching.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from describe_images import load_face_dbs, identify_faces, DEFAULT_FACE_DB

KNOWN_NAMES = {"jack", "jim", "jill"}
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def expected_for(photo: Path) -> set[str]:
    """Expected names for a test photo, from folder name or filename."""
    group = photo.parent.name.lower()
    if group == "unknown":
        return set()
    if group == "difficult":
        return {
            tok for tok in photo.stem.lower().split("_")
            if tok in KNOWN_NAMES
        }
    # Category folders: jim_jack, jim_jill_jack, jack, ...
    return {tok for tok in group.split("_") if tok in KNOWN_NAMES}


def score_photo(photo: Path, db: dict, tolerance: float, model: str,
                upsample: int) -> dict:
    faces = identify_faces(photo, db, tolerance, model=model, upsample=upsample)
    if faces is None:
        faces = []
    got = {f["name"] for f in faces if f["name"] != "unknown"}
    expected = expected_for(photo)
    return {
        "photo": photo,
        "expected": expected,
        "got": got,
        "faces": faces,
        "tp": len(expected & got),
        "fn": len(expected - got),
        "fp": len(got - expected),
        "detected": len(faces),
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Score face identification against a labeled test tree.",
    )
    ap.add_argument("test_dir", help="Directory of labeled test subfolders")
    ap.add_argument("--face-db", default=None, action="append",
                    help="Face DB path (repeatable), same as describe_images "
                         f"(default: {DEFAULT_FACE_DB})")
    ap.add_argument("--tolerance", type=float, default=0.5)
    ap.add_argument("--model", choices=["hog", "cnn"], default="hog")
    ap.add_argument("--upsample", type=int, default=1)
    args = ap.parse_args()

    db_paths = [Path(p).expanduser() for p in (args.face_db or [str(DEFAULT_FACE_DB)])]

    root = Path(args.test_dir).expanduser().resolve()
    if not root.is_dir():
        print(f"ERROR: not a directory: {root}")
        return 1

    db = load_face_dbs(db_paths)
    if db is None:
        print(f"ERROR: could not load face DB(s): {db_paths}")
        return 1

    photos = sorted(
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in IMG_EXT
    )
    if not photos:
        print(f"No images found under {root}")
        return 1

    print()
    print("Face Identification Evaluation")
    print("==============================")
    print(f"Directory : {root}")
    print(f"Face DBs  : {', '.join(str(p) for p in db_paths)}")
    print(f"Model     : {args.model}  tolerance={args.tolerance}  "
          f"upsample={args.upsample}")
    print(f"Photos    : {len(photos)}")
    print()

    results = []
    for i, photo in enumerate(photos, 1):
        r = score_photo(photo, db, args.tolerance, args.model, args.upsample)
        results.append(r)
        rel = photo.relative_to(root)

        if r["fn"] == 0 and r["fp"] == 0 and r["expected"]:
            mark = "OK  "
        elif not r["expected"] and r["fp"] == 0:
            mark = "OK  "          # unknown folder, correctly no names
        elif r["fp"] > 0:
            mark = "FP  "          # named someone who should not be named
        else:
            mark = "MISS"

        exp = ",".join(sorted(r["expected"])) or "-"
        got = ",".join(sorted(r["got"])) or "-"
        detail = " ".join(
            f"{f['name']}({f['distance']:.2f})" for f in r["faces"]
        ) or "no faces"
        print(f"[{i:2}/{len(photos)}] {mark} {rel}")
        print(f"          expected={exp}  got={got}  faces: {detail}")

    # -----------------------------------------------------------------
    # Totals
    # -----------------------------------------------------------------
    tp = sum(r["tp"] for r in results)
    fn = sum(r["fn"] for r in results)
    fp = sum(r["fp"] for r in results)

    with_expected = [r for r in results if r["expected"]]
    perfect = [r for r in with_expected
               if r["fn"] == 0 and r["fp"] == 0]
    unknown_photos = [r for r in results if not r["expected"]]
    unknown_clean = [r for r in unknown_photos if r["fp"] == 0]

    # Faces in photos where some expected name was absent entirely
    missed_details = []
    for r in results:
        for name in sorted(r["expected"] - r["got"]):
            missed_details.append(f"  {r['photo'].relative_to(root)}  missing: {name}")

    fp_details = []
    for r in results:
        for name in sorted(r["got"] - r["expected"]):
            fp_details.append(f"  {r['photo'].relative_to(root)}  unexpected: {name}")

    recall = tp / (tp + fn) if (tp + fn) else 1.0
    precision = tp / (tp + fp) if (tp + fp) else 1.0

    print()
    print("Totals")
    print("======")
    print(f"Photos with expected people : {len(with_expected)}")
    print(f"  fully correct (all names, no extras) : {len(perfect)} "
          f"({100 * len(perfect) / max(len(with_expected), 1):.0f}%)")
    print(f"Photos expecting nobody     : {len(unknown_photos)}")
    print(f"  clean (no false names)    : {len(unknown_clean)}")
    print()
    print(f"Name recall     : {tp}/{tp + fn} = {100 * recall:.1f}%  "
          f"(expected names correctly found)")
    print(f"Name precision  : {tp}/{tp + fp} = {100 * precision:.1f}%  "
          f"(found names that were correct)")
    print()

    if fp_details:
        print(f"False positives ({fp}):")
        print("\n".join(fp_details))
        print()
    if missed_details:
        print(f"Missed names ({fn}):")
        print("\n".join(missed_details))
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
