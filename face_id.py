#!/usr/bin/env python3
"""
face_id.py - find faces in a photo and name the ones you've enrolled.

SETUP
    pip install face_recognition pillow numpy
    (face_recognition needs dlib; on Windows/Mac the easiest route is
     conda install -c conda-forge dlib, or pip install dlib-bin)

FOLDER LAYOUT FOR ENROLLMENT
    known/
      Chris/  1.jpg 2.jpg 3.jpg ...
      Jim/    ...
      Sally/  ...
      Sue/    ...
    Use 5-10 photos per person: different angles, lighting, years, glasses on/off.
    Each reference photo should contain ONLY that person (or at least have
    them as the single face). Photos with multiple faces are skipped.

USAGE
    python face_id.py enroll known/ --db family.pkl
    python face_id.py identify portrait.jpg --db family.pkl
    python face_id.py identify portrait.jpg --db family.pkl --annotate out.jpg --pretty

OUTPUT (stdout, JSON)
    [
      {"name": "Chris",   "x1": 812, "y1": 240, "x2": 1010, "y2": 470, "distance": 0.31},
      {"name": "unknown", "x1": 120, "y1": 260, "x2": 300,  "y2": 480, "distance": 0.71}
    ]
    distance: lower = more similar (0.6 is the library's default cutoff).
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

import face_recognition
import numpy as np
from PIL import Image, ImageDraw, ImageOps

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_image(path):
    """Load as RGB numpy array, honoring EXIF rotation (phone/scanner photos)."""
    img = Image.open(path)
    img = ImageOps.exif_transpose(img).convert("RGB")
    return np.array(img)


def enroll(known_dir, db_path, model="hog"):
    known_dir = Path(known_dir)
    names, encodings = [], []
    for person_dir in sorted(p for p in known_dir.iterdir() if p.is_dir()):
        count = 0
        for f in sorted(person_dir.iterdir()):
            if f.suffix.lower() not in IMG_EXT:
                continue
            img = load_image(f)
            boxes = face_recognition.face_locations(img, model=model)
            if len(boxes) != 1:
                print(f"  skip {f} ({len(boxes)} faces found, need exactly 1)", file=sys.stderr)
                continue
            enc = face_recognition.face_encodings(img, boxes, num_jitters=5)[0]
            names.append(person_dir.name)
            encodings.append(enc)
            count += 1
        print(f"{person_dir.name}: {count} reference photos", file=sys.stderr)

    if not names:
        sys.exit("No usable reference faces found.")
    with open(db_path, "wb") as fh:
        pickle.dump({"names": names, "encodings": np.array(encodings)}, fh)
    print(f"Saved {len(names)} encodings to {db_path}", file=sys.stderr)


def identify(photo, db, tolerance=0.5, model="hog", upsample=1):
    # Accept a path to a pickle, or an already-loaded {"names", "encodings"}
    # dict so a caller can merge several face DBs before matching.
    if not isinstance(db, dict):
        with open(db, "rb") as fh:
            db = pickle.load(fh)
    known_names = db["names"]
    known_enc = db["encodings"]

    img = load_image(photo)
    # For big professional portraits with small faces, raise upsample (slower)
    # or use model="cnn" (much better, best with a GPU).
    boxes = face_recognition.face_locations(img, number_of_times_to_upsample=upsample, model=model)
    encs = face_recognition.face_encodings(img, boxes)

    results = []
    for (top, right, bottom, left), enc in zip(boxes, encs):
        dists = face_recognition.face_distance(known_enc, enc)
        # Best match per PERSON (min distance over that person's reference photos)
        per_person = {}
        for n, d in zip(known_names, dists):
            per_person[n] = min(d, per_person.get(n, 9.9))
        best_name = min(per_person, key=per_person.get)
        best_dist = per_person[best_name]
        name = best_name if best_dist <= tolerance else "unknown"
        results.append({
            "name": name,
            "x1": int(left), "y1": int(top), "x2": int(right), "y2": int(bottom),
            "distance": round(float(best_dist), 3),
        })

    # Don't let two faces claim the same person: keep the closer match,
    # demote the other to unknown.
    claimed = {}
    for r in results:
        if r["name"] == "unknown":
            continue
        other = claimed.get(r["name"])
        if other is None:
            claimed[r["name"]] = r
        elif r["distance"] < other["distance"]:
            other["name"] = "unknown"
            claimed[r["name"]] = r
        else:
            r["name"] = "unknown"

    # Sort left-to-right for stable output
    results.sort(key=lambda r: r["x1"])
    return results, img


def annotate(img_arr, results, out_path):
    img = Image.fromarray(img_arr)
    draw = ImageDraw.Draw(img)
    w = max(2, img.width // 400)
    for r in results:
        color = "red" if r["name"] == "unknown" else "lime"
        draw.rectangle([r["x1"], r["y1"], r["x2"], r["y2"]], outline=color, width=w)
        draw.text((r["x1"] + 4, r["y1"] + 4), r["name"], fill=color)
    img.save(out_path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("enroll", help="build a face database from known/<Name>/*.jpg")
    e.add_argument("known_dir")
    e.add_argument("--db", default="faces.pkl")
    e.add_argument("--model", choices=["hog", "cnn"], default="hog")

    i = sub.add_parser("identify", help="detect + name faces in a photo")
    i.add_argument("photo")
    i.add_argument("--db", default="faces.pkl")
    i.add_argument("--tolerance", type=float, default=0.5,
                   help="max distance to accept a match (lower = stricter). default 0.5")
    i.add_argument("--model", choices=["hog", "cnn"], default="hog")
    i.add_argument("--upsample", type=int, default=1)
    i.add_argument("--annotate", help="also write an image with boxes drawn")
    i.add_argument("--pretty", action="store_true", help="indent the JSON")

    args = ap.parse_args()
    if args.cmd == "enroll":
        enroll(args.known_dir, args.db, args.model)
    else:
        results, img = identify(args.photo, args.db, args.tolerance, args.model, args.upsample)
        print(json.dumps(results, indent=2 if args.pretty else None))
        if args.annotate:
            annotate(img, results, args.annotate)


if __name__ == "__main__":
    main()
