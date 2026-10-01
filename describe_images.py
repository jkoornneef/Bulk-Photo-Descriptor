"""
describe_images.py

Create searchable text descriptions for photographs using a local
Ollama vision model.

For:
    IMG_1234.jpg

creates:
    IMG_1234.txt

Existing .txt files are skipped unless --overwrite is specified.

Examples:

    # Process one directory and all subdirectories
    python describe_images.py "D:\\Photos"

    # Process only the specified directory
    python describe_images.py "D:\\Photos" --no-recursive

    # Use a different vision model
    python describe_images.py "D:\\Photos" --model qwen3-vl

    # See what would be processed without calling Ollama
    python describe_images.py "D:\\Photos" --dry-run

    # Regenerate descriptions
    python describe_images.py "D:\\Photos" --overwrite

Find photos that were tagged as rejects. This reads the .txt files a
previous run already wrote. No model is loaded and no network call is
made, so it is fast enough for a whole library.

    # Blurry photos
    python describe_images.py "D:\\Photos" --find-blurry

    # Subjectless photos
    python describe_images.py "D:\\Photos" --find-subjectless

    # Both kinds, all scores
    python describe_images.py "D:\\Photos" --find-blurry --find-subjectless

    # Pass 1 - the obvious garbage, 9 and 10, for bulk deleting
    python describe_images.py "D:\\Photos" \
        --find-blurry --find-subjectless --min-confidence 9

    # Pass 2 - the borderline ones, 7 and 8, to review one at a time
    python describe_images.py "D:\\Photos" \
        --find-blurry --find-subjectless \
        --min-confidence 7 --max-confidence 8

Mark the rejects in their filenames so image software can sort or filter
on the name. Only the name changes; the pixels are not touched.

    # Preview first - nothing on disk changes
    python describe_images.py "D:\\Photos" \
        --find-blurry --find-subjectless --min-confidence 9 \
        --rename-tagged --dry-run

    # IMG00023.jpg -> IMG00023.blurry.jpg
    python describe_images.py "D:\\Photos" \
        --find-blurry --find-subjectless --min-confidence 9 \
        --rename-tagged

    # Undo - strips .blurry / .nosubject back out
    python describe_images.py "D:\\Photos" --revert-renames
    
    
    python describe_images.py "D:\\Photos" --model llava:7b
    python describe_images.py "D:\\Photos" --model gemma4:e4b
    python describe_images.py "D:\\Photos" --model qwen3.5:latest
    python describe_images.py "D:\\Photos" --model gemma4:12b
"""

from __future__ import annotations

import argparse
import base64
import os
import re
import sys
import time
from pathlib import Path
from typing import NamedTuple

import requests


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_OLLAMA_URL = "http://localhost:11434/api/chat"
DEFAULT_OLLAMA_BASE = "http://localhost:11434"
DEFAULT_MODEL = "llava:13b"

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".gif",
    ".bmp",
    ".tif",
    ".tiff",
}

PROMPT = """Describe this photograph in detail for use in a personal
photo-search system.

The description should make the photograph searchable by concepts,
objects, places, activities, people, events, and visual characteristics.

Include, when applicable:

- Main subject and important objects
- Mountains, landscapes, buildings, beaches, water, snow, forests, etc.
- Vehicles, boats, airplanes, and identifiable aircraft
- Signs, logos, and other visible text
- Activities taking place
- Recognizable landmarks or locations
- Weather, season, and general environment
- Colors or distinctive visual characteristics
- Types of animals
- Types of vehicles or aircraft when reasonably identifiable
- Any other distinctive details that would help someone find this photo

If text is visible in the photograph, transcribe it.

Do not invent information that cannot reasonably be determined from
the image. If a location, person, aircraft, or other object is uncertain,
describe it as uncertain rather than guessing.

---
PHOTO QUALITY TAGS
---

This is a family photo library: candid snapshots from point-and-shoot
cameras and phones, plus some scanned film. Many are old, grainy,
soft, dim or slightly shaky, and that is normal and perfectly fine.
Judge the photo as a family keepsake, not as portfolio work.

Almost every photograph here is worth keeping. Only flag a photo
that is genuinely UNUSABLE - nothing in it can be made out or
salvaged, and nobody would want it back. Not ugly, not plain, not
technically imperfect, and not boring.

If the photograph is a KEEPER - you can tell what it shows, even if
the grain is heavy, the light was poor or it is not perfectly sharp -
output NO TAG AT ALL. Most photographs are keepers, so the normal
and expected case is to begin your response directly with the
description. Do not tag a photograph you would keep.

Only when you would genuinely throw this photograph away, begin your
response with a tag line instead:

**Blurry Photo**-<N>
    The image is out of focus, shows camera shake or motion blur, or
    the subject is unrecognisable because of blur. Judge sharpness on
    the SUBJECT - the thing the photographer was aiming at. A part of
    the frame that is out of focus while the subject is sharp is not
    blur and must never be tagged.

**Subjectless Photo**-<N>
    The image contains no clear, identifiable subject of focus. Nothing
    in the frame can be named or recognised as the thing the
    photographer meant to capture.

Include only the tags that apply. Both may apply at once.

<N> is how sure you are that the photograph should be rejected:

    10  certain reject, keep nothing
     9  very likely reject
     8  likely reject
     7  probably reject

Never write a score below 7. Anything less certain than "probably
reject" is a keeper, and a keeper gets no tag. Use whole numbers only.
Do not write the words "confidence", "rating" or "score" in the tag line.

When in doubt, do not tag it. A missed tag costs far less than a good
photograph being wrongly buried.

Do not tag a photograph when:

- It is sharp and something in it is identifiable, however plain the
  subject is. An empty beach, a forest, a mountainscape, a sunset, a
  cityscape, a wall or a close-up of sky are all keepers.
- It is a candid snapshot of ordinary life - children playing outside,
  a family around a table, a birthday cake, pets, a garden - and you
  can make out who or what is in it. A little softness, a cloudy day,
  wind in the hair or a face caught mid-laugh does not make it
  unusable.
- It looks like a film scan. Heavy grain, colour cast, low contrast,
  a soft corner, dust specks or a slightly crooked scan are all part
  of an old photograph, not defects. Grain is never a reason to tag.
- It was taken in poor light or at high ISO / pushed film and is
  noisy and dim but still readable. Noise and softness together are
  not enough on their own - tag only if you genuinely cannot tell
  what the photo is of.
- Depth of field is doing its job and the subject is sharp. A soft or
  blurred background, creamy bokeh, a wide-open aperture, a close-up
  with the scene behind it out of focus, a blurred foreground, a
  blurred edge of the frame, or a distant subject seen through glass
  or foliage are all deliberate photography, not camera blur. Rate
  the photo on the subject, never on how much of the frame is out of
  focus. If you can describe the subject clearly, it is sharp, so it
  is not a Blurry Photo.
- Stars, the Milky Way, an aurora or the moon are visible and
  reasonably sharp - even when they are small and scattered across the
  frame because the camera was hand-held, or noisy and dim from a long
  exposure. A night sky IS the subject. Only tag it when the sky
  itself is unusable: a black frame, the lens cap still being on,
  heavy star trailing, or so little light that nothing at all can be
  identified.
- The subject is small, distant, unusual, partially cropped or simply
  hard to recognise at first glance.
- Part of the subject is soft but the rest of it is sharp, or the
  whole frame is a little soft, so you can still tell what it is.

### DO FLAG these

These are the genuine rejects - the whole image is spoiled, not just
part of it:

- The entire frame is smeared or streaked with camera shake, so that
  everything in it is a motion "woosh" and no edge anywhere is sharp.
- A finger, thumb or part of a hand covers part of the frame or the
  lens, especially if the frame is black, dark or smeared behind it.
- A shot fired by accident while the camera was being put away or
  while the photographer was walking - typically the ground, a
  ceiling, a car floor or a carpet filling the frame with nothing
  recognisable in it. Tag it Subjectless, and Blurry too if it is
  smeared.
- The lens cap was still on, or the frame is completely black, white
  or a single flat colour with nothing in it.
- A zoomed or cropped detail so tight that it is an unrecognisable
  patch of texture - a blown-out white frame, a black frame, a smear
  of colour.
- A night sky that is nothing but heavy star trailing, or a scene so
  dark that no subject can be found at all.

The test for all of these: could you crop or edit this photo into
something worth keeping? If not, tag it. If yes, it is a keeper.

Write the tag line first, on its own line, in exactly this form:

**Blurry Photo**-8 **Subjectless Photo**-9

Then a blank line, then the description.

If you do tag a photograph, still describe whatever can reasonably be
determined from it and say plainly why it was tagged.

---
END QUALITY TAGS
---

Write a concise but reasonably detailed natural-language description.
Do not use JSON. Do not use markdown headings. A quality tag line at the
top, when one is warranted, is the only bold text permitted.
"""


# ---------------------------------------------------------------------------
# Ollama
# ---------------------------------------------------------------------------

def _build_prompt(image_path: Path) -> str:
    """Build the full prompt including context and path metadata."""
    prompt = PROMPT

    context_dir = image_path.parent
    context_content = read_context_file(context_dir)

    if context_content:
        prompt += (
            f"\n\n### CONTEXT INFORMATION:\n{context_content}\n"
            "\nThis context applies to the following image and is "
            "verified factual and may be referred to:"
        )

    prompt += (
        f"\n\nThe original filename is: {image_path.name}\n"
        f"\n\nThe image resides in directory structure: {image_path.parent}\n"
        "The filename and/or file path may contain useful context "
        "(trip, location, event, date), but do not assume that "
        "information is true unless the image supports it."
    )

    return prompt


def _check_star_failure(description: str) -> None:
    """Raise if the response is the Qwen 3B star-character failure mode."""
    non_space = description.replace(" ", "").replace("\n", "")
    if len(non_space) >= 10 and set(non_space) == {"*"}:
        raise RuntimeError(
            f"Model returned {len(non_space)} '*' characters "
            "instead of a description."
        )


def _read_image_b64(image_path: Path) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


# ---------------------------------------------------------------------------
# Photo quality tags
# ---------------------------------------------------------------------------
#
# The model is asked to label each photograph with one or more quality tags
# placed at the very top of the description, each carrying a confidence
# score from 0 to 10:
#
#     **Blurry Photo**-7 **Subjectless Photo**-3
#
# The tags give you a searchable string for finding photographs that are
# out of focus or that have no identifiable subject. Because local models
# are inconsistent about layout and punctuation, the response is normalized
# so that a search for "Subjectless Photo" reliably finds every match.

# Emitted in this order whenever more than one tag applies.
QUALITY_TAGS = ("Blurry Photo", "Subjectless Photo")

# Tags below this score are treated as "this is a keeper" and removed from
# the description entirely. The prompt only ever asks the model for a score
# of 7 or above, but a local model will sometimes talk itself into a low
# score on a perfectly good photograph. Dropping those tags keeps a search
# for "Subjectless Photo" a short list of genuine rejects rather than a
# list that quietly grows to include keepers.
QUALITY_TAG_MIN_SCORE = 7

# Score assumed when the model names a tag but omits the number. A tag is
# named only when the photo is being rejected, so this defaults to the
# threshold rather than to a neutral middle value.
DEFAULT_QUALITY_SCORE = QUALITY_TAG_MIN_SCORE

# An un-bolded tag is only trusted near the top of the response. Further
# down, a bare "blurry" is far more likely to be ordinary prose.
TAG_SEARCH_WINDOW = 250

_TAG_KEYWORDS = {
    "blurry": "Blurry Photo",
    "subjectless": "Subjectless Photo",
}

# Matches a tag with a score, tolerating the many ways a model tends to
# write it: "**Subjectless Photo**-8", "Blurry Photo: 7", "blurry 3", etc.
_TAG_RE = re.compile(
    r"\*{0,2}\s*(blurry|subjectless)\b\s*"
    r"(?:photo|image|picture|shot|frame|is|rating|confidence|level|score)?"
    r"[\*\s:\-–—=]*"
    r"(\d{1,2})(?:[ \t]*/[ \t]*10\b)?",
    re.IGNORECASE,
)

# Matches a tag written without a score, e.g. "**Blurry Photo**". This is
# deliberately strict: either fully bolded, or occupying a whole line. A
# looser pattern would swallow ordinary prose such as "blurry because of
# the long exposure", which must never become a tag.
_UNTAGGED_TAG_RE = re.compile(
    r"\*\*(blurry|subjectless)\s*(?:photo|image|picture|shot)?\s*\*\*"
    r"|^[ \t]*(?:[-*][ \t]*)?(blurry|subjectless)[ \t]*"
    r"(?:photo|image|picture|shot)?[ \t]*\.?[ \t]*$",
    re.IGNORECASE | re.MULTILINE,
)

# Lines left empty behind by a removed tag, e.g. "**Blurry Photo**-7" or a
# bare "- ".
_HEADER_JUNK_RE = re.compile(r"^[\s*\-–—#\.…:|]*$")


def _is_tag_candidate(description: str, match: re.Match) -> bool:
    """Only trust a tag that is bolded, first, or alone on its line.

    Local models often use the words "blurry" and "subjectless" in ordinary
    sentences. Requiring one of these three shapes keeps a descriptive
    sentence from being mistaken for a tag.
    """
    start = match.start()

    if description[start:start + 2] == "**":
        return True

    if start <= TAG_SEARCH_WINDOW:
        line_start = description.rfind("\n", 0, start) + 1
        line_end = description.find("\n", match.end())
        if line_end == -1:
            line_end = len(description)
        remainder = description[line_start:start] + description[match.end():line_end]
        if _HEADER_JUNK_RE.match(remainder):
            return True

    return False


def _remove_spans(text: str, spans: list[tuple[int, int]]) -> str:
    """Remove the given character ranges from text."""
    pieces = []
    cursor = 0
    for start, end in sorted(spans):
        if start < cursor:
            continue
        pieces.append(text[cursor:start])
        cursor = end
    pieces.append(text[cursor:])
    return "".join(pieces)


def _strip_junk_header(body: str) -> str:
    """Drop leading lines left empty once the tag line is removed."""
    lines = body.splitlines()
    index = 0
    while index < len(lines) and _HEADER_JUNK_RE.match(lines[index]):
        index += 1
    return "\n".join(lines[index:])


class QualityResult(NamedTuple):
    """The outcome of normalizing a model's description.

    ``text`` is what gets written to the .txt file. ``tags`` holds only the
    quality tags that survived the threshold, so an empty dict means the
    photograph was judged a keeper.
    """

    text: str
    tags: dict[str, int]


def format_tag_header(tags: dict[str, int]) -> str:
    """Render tags as the canonical header line, e.g. ``**Blurry Photo**-8``."""
    return " ".join(
        f"**{name}**-{tags[name]}"
        for name in QUALITY_TAGS
        if name in tags
    )


def _collect_tag_matches(
    description: str,
) -> tuple[dict[str, int], list[tuple[int, int]]]:
    """Find every quality tag in a description.

    Returns the tags with their scores, plus the character spans they
    occupy so the caller can lift them out of the text. Shared by the
    writer and the finder so the two can never disagree about the format.
    """
    found: dict[str, int] = {}
    spans: list[tuple[int, int]] = []

    # Tags that carry an explicit score. These take precedence.
    for match in _TAG_RE.finditer(description):
        if not _is_tag_candidate(description, match):
            continue
        name = _TAG_KEYWORDS[match.group(1).lower()]
        found.setdefault(name, min(int(match.group(2)), 10))
        spans.append(match.span())

    # Tags written without a score fall back to the default rating. This
    # also runs when scored tags were found, so a response that scores one
    # tag and leaves the other bare still yields both.
    for match in _UNTAGGED_TAG_RE.finditer(description):
        if not _is_tag_candidate(description, match):
            continue
        keyword = match.group(1) or match.group(2)
        name = _TAG_KEYWORDS[keyword.lower()]
        if name in found:
            continue
        found[name] = DEFAULT_QUALITY_SCORE
        spans.append(match.span())

    return found, spans


def extract_quality_tags(description: str) -> dict[str, int]:
    """Return the quality tags written into a description.

    No threshold is applied here. Descriptions written by this tool already
    have sub-threshold tags removed, so anything found in a ``.txt`` file
    is a genuine reject; filtering by score is left to the caller.
    """
    return _collect_tag_matches(description)[0]


def normalize_quality_tags(description: str) -> QualityResult:
    """Anchor the quality tags to the top of a rejected description.

    A tag line such as ``**Blurry Photo**-8 **Subjectless Photo**-9`` is
    kept only for photographs the model rates at ``QUALITY_TAG_MIN_SCORE``
    or above, and is always rewritten to that canonical form and placed at
    the very top, no matter how the model actually formatted its response.

    Any tag below the threshold is treated as a keeper: it is stripped from
    the text and not re-added. Descriptions with no tags are returned
    unchanged apart from surrounding whitespace.
    """
    found, spans = _collect_tag_matches(description)

    body = _strip_junk_header(_remove_spans(description, spans))

    # A tag only counts as "bad" once it clears the threshold. Drop the rest
    # so a keeper never carries a quality tag, and so a search for a tag
    # returns only genuine rejects.
    found = {
        name: score
        for name, score in found.items()
        if score >= QUALITY_TAG_MIN_SCORE
    }

    if not body.strip():
        # The tags were the entire response. Leave the model's own text
        # alone rather than returning an empty file.
        return QualityResult(description.strip(), {})

    if not found:
        # Every tag was below the threshold, so this is a keeper. Return
        # the description with the tag line removed and no tag re-added.
        return QualityResult(body.strip(), {})

    header = format_tag_header(found)

    return QualityResult(f"{header}\n\n{body.strip()}", found)


# ---------------------------------------------------------------------------
# Ollama backend
# ---------------------------------------------------------------------------

def describe_image_ollama(
    image_path: Path,
    model: str,
    api_url: str,
    timeout: int,
    retries: int,
) -> str:
    """Send an image to Ollama and return the generated description."""

    image_data = _read_image_b64(image_path)
    prompt = _build_prompt(image_path)

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "images": [image_data],
                "content": prompt,
            }
        ],
        "stream": False,
    }

    last_error = None

    for attempt in range(1, retries + 1):
        try:
            response = requests.post(api_url, json=payload, timeout=timeout)
            response.raise_for_status()
            data = response.json()

            description = (
                data.get("message", {})
                .get("content", "")
                .strip()
            )

            if not description:
                raise RuntimeError("Ollama returned an empty response.")

            _check_star_failure(description)
            return description

        except Exception as exc:
            last_error = exc
            if attempt < retries:
                wait_seconds = min(5 * attempt, 30)
                print(
                    f"    Ollama error: {exc}"
                    f"\n    Retrying in {wait_seconds} seconds..."
                )
                time.sleep(wait_seconds)

    raise RuntimeError(
        f"Ollama failed after {retries} attempts: {last_error}"
    )


# ---------------------------------------------------------------------------
# OpenAI-compatible backend  (OpenAI, OpenRouter, vLLM, etc.)
# ---------------------------------------------------------------------------

def describe_image_openai(
    image_path: Path,
    model: str,
    api_url: str,
    api_key: str,
    timeout: int,
    retries: int,
) -> str:
    """Send an image to an OpenAI-compatible API and return the description."""

    image_data = _read_image_b64(image_path)
    prompt = _build_prompt(image_path)

    # Build the data URI (OpenAI expects this format).
    suffix = image_path.suffix.lower().lstrip(".")
    mime = {
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "png": "image/png",
        "webp": "image/webp",
        "gif": "image/gif",
        "bmp": "image/bmp",
        "tif": "image/tiff",
        "tiff": "image/tiff",
    }.get(suffix, "image/jpeg")

    data_uri = f"data:{mime};base64,{image_data}"

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": data_uri},
                    },
                ],
            }
        ],
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    last_error = None

    for attempt in range(1, retries + 1):
        try:
            response = requests.post(
                api_url,
                json=payload,
                headers=headers,
                timeout=timeout,
            )
            response.raise_for_status()
            data = response.json()

            description = (
                data["choices"][0]["message"]["content"].strip()
            )

            if not description:
                raise RuntimeError(
                    "OpenAI-compatible API returned an empty response."
                )

            _check_star_failure(description)
            return description

        except Exception as exc:
            last_error = exc
            if attempt < retries:
                wait_seconds = min(5 * attempt, 30)
                print(
                    f"    API error: {exc}"
                    f"\n    Retrying in {wait_seconds} seconds..."
                )
                time.sleep(wait_seconds)

    raise RuntimeError(
        f"API failed after {retries} attempts: {last_error}"
    )


# ---------------------------------------------------------------------------
# Unified entry point
# ---------------------------------------------------------------------------

def describe_image(
    image_path: Path,
    model: str,
    api_url: str,
    api_key: str | None,
    timeout: int,
    retries: int,
) -> str:
    """Route to Ollama or OpenAI-compatible backend."""
    if api_key:
        return describe_image_openai(
            image_path, model, api_url, api_key, timeout, retries,
        )
    return describe_image_ollama(
        image_path, model, api_url, timeout, retries,
    )

def unload_model(model: str, ollama_base_url: str):
    """Ask Ollama to unload the model from memory."""

    try:
        requests.post(
            f"{ollama_base_url}/api/generate",
            json={
                "model": model,
                "keep_alive": 0,
            },
            timeout=30,
        )
    except Exception:
        pass
        
        
# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def find_images(root: Path, recursive: bool):
    """Yield image files under root."""

    if recursive:
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                yield path
    else:
        for path in root.iterdir():
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                yield path

def read_context_file(directory):
    context_files = [directory / 'context.md', directory / 'context.txt']
    for file in context_files:
        if file.exists():
            with open(file, 'r') as f:
                return f.read()
    return None

# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------

def process_images(args):
    root = Path(args.directory).expanduser().resolve()

    if not root.exists():
        print(f"ERROR: Directory does not exist:")
        print(f"  {root}")
        return 1

    if not root.is_dir():
        print(f"ERROR: Not a directory:")
        print(f"  {root}")
        return 1

    images = list(find_images(root, args.recursive))

    if not images:
        print("No images found.")
        return 0

    backend = "OpenAI" if args.api_key else "Ollama"
    print()
    print("Photo Description Utility")
    print("=========================")
    print(f"Directory : {root}")
    print(f"Recursive : {args.recursive}")
    print(f"Model     : {args.model}")
    print(f"Backend   : {backend}")
    print(f"API URL   : {args.api_url}")
    print(f"Images    : {len(images)}")
    print(f"Tag at    : {QUALITY_TAG_MIN_SCORE}+ (bad photos only)")
    print()

    processed = 0
    skipped = 0
    failed = 0

    for index, image_path in enumerate(images, start=1):

        txt_path = image_path.with_suffix(".txt")

        relative_path = image_path.relative_to(root)

        print(f"[{index}/{len(images)}] {relative_path}")

        # ---------------------------------------------------------------
        # Existing description
        # ---------------------------------------------------------------

        if txt_path.exists() and not args.overwrite:
            print("    SKIP - description already exists")
            skipped += 1
            continue

        # ---------------------------------------------------------------
        # Dry run
        # ---------------------------------------------------------------

        if args.dry_run:
            if txt_path.exists():
                print("    WOULD REPLACE:", txt_path.name)
            else:
                print("    WOULD CREATE :", txt_path.name)

            processed += 1
            continue

        # ---------------------------------------------------------------
        # Generate description
        # ---------------------------------------------------------------

        try:
            print("    Sending to model...")

            description = describe_image(
                image_path=image_path,
                model=args.model,
                api_url=args.api_url,
                api_key=args.api_key,
                timeout=args.timeout,
                retries=args.retries,
            )

            # Anchor any quality tags to the top of the file in a
            # consistent, searchable format.
            result = normalize_quality_tags(description)
            description = result.text

            # -----------------------------------------------------------
            # Write to temporary file first.
            #
            # This prevents an interrupted write from leaving behind an
            # empty .txt file that would cause the image to be skipped
            # on the next run.
            # -----------------------------------------------------------

            temp_path = txt_path.with_suffix(".txt.tmp")

            temp_path.write_text(
                description + "\n",
                encoding="utf-8",
            )

            temp_path.replace(txt_path)

            print(f"    CREATED: {txt_path.name}")

            # Echo any quality warning onto the second line so a long run
            # can be scanned for photos worth reviewing.
            if result.tags:
                print(f"    WARNING: {format_tag_header(result.tags)}")

            processed += 1

        except KeyboardInterrupt:
            print()
            print("Interrupted by user.")
            print("Already completed descriptions have been preserved.")
            return 130

        except Exception as exc:
            print(f"    ERROR: {exc}")
            failed += 1

    # -------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------

    print()
    print("Finished")
    print("========")
    print(f"Processed : {processed}")
    print(f"Skipped   : {skipped}")
    print(f"Failed    : {failed}")

    return 1 if failed else 0


# ---------------------------------------------------------------------------
# Find / rename
# ---------------------------------------------------------------------------
#
# Reads the .txt descriptions written by a previous run and works out which
# images carry a quality tag. No LLM is involved, so this is fast enough to
# run over a whole library in seconds.

# Appended to the filename stem to mark a reject, so the mark survives
# whichever image software the user reviews and deletes with. Chosen over
# "--blurry" because a dotted stem groups neatly in Explorer and leaves the
# real extension last.
MARKERS = {
    "Blurry Photo": ".blurry",
    "Subjectless Photo": ".nosubject",
}


def collect_tagged_images(
    root: Path,
    recursive: bool,
    wanted: list[str],
    min_score: int,
    max_score: int,
) -> tuple[list[tuple[Path, dict[str, int]]], int]:
    """Yield every image whose description matches the requested filters.

    Returns the matches as ``(image_path, matching_tags)`` pairs, plus a
    count of how many tagged images fell outside the score band so the
    caller can report it.
    """
    matches: list[tuple[Path, dict[str, int]]] = []
    tagged_any = 0

    for image_path in find_images(root, recursive):
        txt_path = image_path.with_suffix(".txt")

        if not txt_path.exists():
            # No description yet, so nothing to judge it on.
            continue

        try:
            description = txt_path.read_text(encoding="utf-8")
        except OSError as exc:
            print(f"    ERROR: {exc}")
            continue

        tags = extract_quality_tags(description)

        if not tags:
            # A keeper, or a description written before tagging existed.
            continue

        tagged_any += 1

        # Does any tag the user asked about fall inside the requested band?
        hits = {
            name: score
            for name, score in tags.items()
            if (not wanted or name in wanted)
            and min_score <= score <= max_score
        }

        if hits:
            matches.append((image_path, hits))

    return matches, tagged_any


def _resolve_root(directory: str) -> Path | None:
    """Resolve and validate a directory, printing an error if it is bad."""
    root = Path(directory).expanduser().resolve()

    if not root.exists():
        print(f"ERROR: Directory does not exist:")
        print(f"  {root}")
        return None

    if not root.is_dir():
        print(f"ERROR: Not a directory:")
        print(f"  {root}")
        return None

    return root


def _wanted_tags(args) -> list[str]:
    """Which tags the user asked about. Empty means "either one"."""
    wanted: list[str] = []
    if args.find_blurry:
        wanted.append("Blurry Photo")
    if args.find_subjectless:
        wanted.append("Subjectless Photo")
    return wanted


def _band_description(min_score: int, max_score: int) -> str:
    return f"{min_score}-{max_score}" if max_score < 10 else f"{min_score}+"


def find_tagged_images(args):
    """Print the image paths whose descriptions carry a quality tag."""

    root = _resolve_root(args.directory)

    if root is None:
        return 1

    images = list(find_images(root, args.recursive))

    if not images:
        print("No images found.")
        return 0

    wanted = _wanted_tags(args)
    min_score = args.min_confidence
    max_score = args.max_confidence

    tag_desc = ", ".join(wanted) if wanted else "any tag"
    band_desc = _band_description(min_score, max_score)

    print()
    print("Find Tagged Photos")
    print("=================")
    print(f"Directory : {root}")
    print(f"Recursive : {args.recursive}")
    print(f"Looking for: {tag_desc}")
    print(f"Score band: {band_desc}")
    print(f"Images    : {len(images)}")
    print()

    matches, tagged_any = collect_tagged_images(
        root, args.recursive, wanted, min_score, max_score,
    )

    for image_path, hits in matches:
        print(f"{image_path}  {format_tag_header(hits)}")

    # -------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------

    print()
    print("Finished")
    print("========")
    print(f"Matched   : {len(matches)}")
    print(f"Tagged    : {tagged_any} total, {tagged_any - len(matches)} outside {band_desc}")
    print(f"Scanned   : {len(images)} images")

    return 0


# ---------------------------------------------------------------------------
# Rename
# ---------------------------------------------------------------------------
#
# Marks a rejected photo by inserting a marker into its filename so the
# mark survives the image software used to review it. The description .txt
# is renamed alongside the photo so the two stay paired, which keeps
# --find-blurry and friends working afterwards.

def _marker_names_for(tags: dict[str, int]) -> str:
    """Build the marker suffix for a set of tags, in canonical order."""
    return "".join(
        MARKERS[name] for name in QUALITY_TAGS if name in tags
    )


def _strip_markers(stem: str) -> tuple[str, list[str]]:
    """Remove any markers from a filename stem.

    Returns the cleaned stem and the names of the markers removed, which is
    empty when the name carries none.
    """
    cleaned = stem
    removed: list[str] = []

    # Longest markers are checked first so ".nosubject" is never mistaken
    # for a trailing ".subject" on a partially written name.
    for name in sorted(QUALITY_TAGS, key=lambda n: -len(MARKERS[n])):
        marker = MARKERS[name]
        if cleaned.lower().endswith(marker):
            cleaned = cleaned[: -len(marker)]
            removed.append(name)

    return cleaned, removed


def _sidecar_target(image_path: Path, new_stem: str) -> Path | None:
    """Where this image's .txt should live after a rename, if it has one."""
    txt_path = image_path.with_suffix(".txt")

    if not txt_path.exists():
        return None

    return txt_path.with_name(new_stem + txt_path.suffix)


def _move(source: Path, target: Path, dry_run: bool) -> bool:
    """Rename a file, refusing to clobber an existing target."""
    if source == target:
        return False

    if target.exists():
        print(f"    SKIP - {target.name} already exists")
        return False

    if not dry_run:
        source.rename(target)

    return True


def rename_tagged_images(args):
    """Append a marker to the filename of every rejected photo."""

    root = _resolve_root(args.directory)

    if root is None:
        return 1

    images = list(find_images(root, args.recursive))

    if not images:
        print("No images found.")
        return 0

    wanted = _wanted_tags(args)
    min_score = args.min_confidence
    max_score = args.max_confidence

    tag_desc = ", ".join(wanted) if wanted else "any tag"
    band_desc = _band_description(min_score, max_score)

    print()
    print("Rename Tagged Photos")
    print("====================")
    print(f"Directory : {root}")
    print(f"Looking for: {tag_desc}")
    print(f"Score band: {band_desc}")
    print(f"Dry run   : {args.dry_run}")
    print(f"Images    : {len(images)}")
    print()

    matches, tagged_any = collect_tagged_images(
        root, args.recursive, wanted, min_score, max_score,
    )

    renamed = 0
    skipped = 0

    for image_path, hits in matches:
        marker = _marker_names_for(hits)
        new_stem = image_path.stem + marker

        # Already carries one of these markers - renaming again would
        # produce a name like IMG_0001.blurry.blurry.jpg.
        _, existing = _strip_markers(image_path.stem)

        if any(name in hits for name in existing):
            skipped += 1
            print(f"    SKIP - {image_path.name} is already marked")
            continue

        target = image_path.with_name(new_stem + image_path.suffix)

        if not _move(image_path, target, args.dry_run):
            skipped += 1
            continue

        # Keep the description with the photo.
        sidecar = _sidecar_target(image_path, new_stem)
        if sidecar is not None:
            _move(image_path.with_suffix(".txt"), sidecar, args.dry_run)

        renamed += 1

        prefix = "WOULD RENAME" if args.dry_run else "RENAMED"
        print(f"    {prefix}: {image_path.name} -> {target.name}")

    # -------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------

    print()
    print("Finished")
    print("========")
    print(f"{'Would rename' if args.dry_run else 'Renamed'} : {renamed}")
    print(f"Skipped   : {skipped}")
    print(f"Tagged    : {tagged_any} total, {tagged_any - len(matches)} outside {band_desc}")
    print(f"Scanned   : {len(images)} images")

    if renamed and not args.dry_run:
        print()
        print("Undo with:")
        print(f"  python describe_images.py \"{args.directory}\" --revert-renames")

    return 0


def revert_renamed_images(args):
    """Strip quality markers back out of the filenames."""

    root = _resolve_root(args.directory)

    if root is None:
        return 1

    images = list(find_images(root, args.recursive))

    if not images:
        print("No images found.")
        return 0

    print()
    print("Revert Renamed Photos")
    print("=====================")
    print(f"Directory : {root}")
    print(f"Recursive : {args.recursive}")
    print(f"Dry run   : {args.dry_run}")
    print(f"Images    : {len(images)}")
    print()

    reverted = 0
    skipped = 0

    for image_path in images:
        new_stem, removed = _strip_markers(image_path.stem)

        if not removed:
            continue

        target = image_path.with_name(new_stem + image_path.suffix)

        if not _move(image_path, target, args.dry_run):
            skipped += 1
            continue

        sidecar = _sidecar_target(image_path, new_stem)
        if sidecar is not None:
            _move(image_path.with_suffix(".txt"), sidecar, args.dry_run)

        reverted += 1

        prefix = "WOULD REVERT" if args.dry_run else "REVERTED"
        print(f"    {prefix}: {image_path.name} -> {target.name}")

    print()
    print("Finished")
    print("========")
    print(f"{'Would revert' if args.dry_run else 'Reverted'} : {reverted}")
    print(f"Skipped   : {skipped}")
    print(f"Scanned   : {len(images)} images")

    if reverted:
        print()
        print("Note: any file whose name ended in .blurry or .nosubject before")
        print("this run has also been renamed. Check git or your backup if")
        print("that name was original.")

    return 0


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Create text descriptions of images using a local Ollama "
            "vision model or an OpenAI-compatible API."
        )
    )

    parser.add_argument(
        "directory",
        help="Directory containing photographs",
    )

    parser.add_argument(
        "--recursive",
        dest="recursive",
        action="store_true",
        default=True,
        help="Process subdirectories (default)",
    )

    parser.add_argument(
        "--no-recursive",
        dest="recursive",
        action="store_false",
        help="Only process images directly in the specified directory",
    )

    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Vision model (default: {DEFAULT_MODEL})",
    )

    parser.add_argument(
        "--api-url",
        default=DEFAULT_OLLAMA_URL,
        help=(
            f"API endpoint URL (default: {DEFAULT_OLLAMA_URL}). "
            "Use OpenAI's URL for cloud APIs: "
            "https://api.openai.com/v1/chat/completions"
        ),
    )

    parser.add_argument(
        "--api-key",
        default=os.environ.get("OPENAI_API_KEY"),
        help=(
            "API key for cloud providers. Also reads from "
            "OPENAI_API_KEY env var. When set, uses OpenAI-compatible "
            "request format."
        ),
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Request timeout in seconds (default: 300)",
    )

    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Number of attempts for failed requests (default: 3)",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate descriptions even when .txt already exists",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be processed without calling the LLM",
    )

    find_group = parser.add_argument_group(
        "find",
        "Search existing descriptions for quality tags. "
        "No LLM is called.",
    )

    find_group.add_argument(
        "--find-blurry",
        action="store_true",
        help="Print images tagged **Blurry Photo**",
    )

    find_group.add_argument(
        "--find-subjectless",
        action="store_true",
        help="Print images tagged **Subjectless Photo**",
    )

    find_group.add_argument(
        "--min-confidence",
        type=int,
        default=QUALITY_TAG_MIN_SCORE,
        metavar="N",
        help=(
            "Only report tags scoring N or higher, 0-10 "
            f"(default: {QUALITY_TAG_MIN_SCORE})"
        ),
    )

    find_group.add_argument(
        "--max-confidence",
        type=int,
        default=10,
        metavar="N",
        help=(
            "Only report tags scoring N or lower, 0-10 (default: 10). "
            "Use with --min-confidence to review one score band at a time"
        ),
    )

    find_group.add_argument(
        "--rename-tagged",
        action="store_true",
        help=(
            "Rename matching photos to mark them, e.g. "
            "IMG_0001.jpg -> IMG_0001.blurry.jpg. The description .txt is "
            "renamed to match. Revert with --revert-renames"
        ),
    )

    find_group.add_argument(
        "--revert-renames",
        action="store_true",
        help="Strip .blurry / .nosubject markers back out of the filenames",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    tag_selection = args.find_blurry or args.find_subjectless

    # --revert-renames works purely off the filenames, so it needs no tag
    # selection and no score band.
    reverting = args.revert_renames

    if reverting and args.rename_tagged:
        print(
            "ERROR: --rename-tagged and --revert-renames cannot be combined. "
            "Rename, then revert in a separate run."
        )
        return 1

    searching = tag_selection or args.rename_tagged

    # Validate before touching the filesystem.
    if not 0 <= args.min_confidence <= 10:
        print(
            f"ERROR: --min-confidence must be between 0 and 10, "
            f"got {args.min_confidence}."
        )
        return 1

    if not 0 <= args.max_confidence <= 10:
        print(
            f"ERROR: --max-confidence must be between 0 and 10, "
            f"got {args.max_confidence}."
        )
        return 1

    if args.min_confidence > args.max_confidence:
        print(
            f"ERROR: --min-confidence ({args.min_confidence}) is above "
            f"--max-confidence ({args.max_confidence}); that matches "
            "nothing."
        )
        return 1

    if args.rename_tagged and not tag_selection:
        print(
            "ERROR: --rename-tagged needs a tag to look for. "
            "Add --find-blurry and/or --find-subjectless."
        )
        return 1

    band_given = (
        args.min_confidence != QUALITY_TAG_MIN_SCORE
        or args.max_confidence != 10
    )

    if not searching and band_given:
        print(
            "ERROR: --min-confidence and --max-confidence only apply to a "
            "search. Add --find-blurry and/or --find-subjectless."
        )
        return 1

    try:
        import requests  # noqa: F401
    except ImportError:
        print("ERROR: The 'requests' package is required.")
        print()
        print("Install it with:")
        print("    python -m pip install requests")
        return 1

    # Reads existing .txt files and filenames only. No model is loaded and
    # no network call is made, so these are near-instant.
    if reverting:
        return revert_renamed_images(args)

    if args.rename_tagged:
        return rename_tagged_images(args)

    if tag_selection:
        return find_tagged_images(args)

    return process_images(args)


if __name__ == "__main__":
    sys.exit(main())

