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
    
    
    
    python describe_images.py "D:\\Photos" --model llava:7b
    python describe_images.py "D:\\Photos" --model gemma4:e4b
    python describe_images.py "D:\\Photos" --model qwen3.5:latest
    python describe_images.py "D:\\Photos" --model gemma4:12b
"""

from __future__ import annotations

import argparse
import base64
import os
import sys
import time
from pathlib import Path

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

Write a concise but reasonably detailed natural-language description.
Do not use JSON or markdown headings.
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

    return parser.parse_args()


def main():
    args = parse_args()

    try:
        import requests  # noqa: F401
    except ImportError:
        print("ERROR: The 'requests' package is required.")
        print()
        print("Install it with:")
        print("    python -m pip install requests")
        return 1

    return process_images(args)


if __name__ == "__main__":
    sys.exit(main())

