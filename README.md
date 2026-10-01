# Bulk-Photo-Descriptor

A simple Python script that batch processes images in a directory, using an LLM to generate descriptions for each image.

Supports both local inference via [Ollama](https://ollama.com/) and cloud vision APIs with an OpenAI-compatible interface (OpenAI, OpenRouter, vLLM, etc.). By default it runs locally with Ollama for full privacy.

> **Note:** This project was developed as part of the research and prototyping for photo ingestion in an upcoming project — [CoraCortex](https://github.com/jkoornneef/CoraCortex) — a modular, local-first AI framework for personal knowledge and automation.

The script iterates through all images in a specified directory and generates descriptions for each image using a local LLM.
It passes the path of the image to the prompt as it may contain contextual information useful for generating accurate descriptions.
For example:

```
c:/photos/2013/March Vacation/Walt Disney World/Water Park/IMG00023.jpg
```

Each directory can also have a context file named `context.txt` or `context.md` that provides additional information for generating image descriptions.

```
In 2013 we enjoyed our march vacation at Walt Disney World, specifically the Typhoon Lagoon water park.
Our oldest son, Jack, was particularly excited about the water slides and spent most of his time there.
Our youngest daughter, Jill, loved the lazy river and spent a lot of time floating around and splashing.
```

Result from LLM:

```
The image depicts a relaxed scene at a waterpark, possibly at Walt Disney World. The focus is on
three individuals who appear to be on a family vacation. The woman, is seated on a blue and
yellow inflatable inner tube, with her legs extended and hands resting on her knees. She is wearing
a black top and sunglasses.

Next to her, the boy, Jack, is also on an inner tube. He is wearing a blue shirt and has his legs
folded underneath him. He is looking towards the camera with a slight smile.

The girl, Jill, is sitting on the edge of the pool adjacent to the inner tubes. She is wearing a pink
shirt and appears to be watching the others with a smile.

The environment is a sunny day, with palm trees in the background, suggesting a warm and tropical climate.
The pool is surrounded by a fence, indicating it's a controlled area within the waterpark. There are no
visible texts, logos, or other identifiable landmarks in the image.
```

## Features

- Iterates through all images in a specified directory
- Calls a local LLM (via Ollama) or cloud API (OpenAI-compatible) to generate image descriptions
- Supports multiple vision models with different speed/quality tradeoffs
- Tags photos it considers rejects as blurry and/or subjectless, with a
  7-10 confidence; good photos get no tag
- Find tagged photos later without calling the LLM again
- Mark rejects in the filename so image software can sort or filter on it,
  with a one-command undo
- Dry-run mode to preview operations
- Optional recursive directory processing
- Overwrite mode to regenerate existing descriptions

## Photo Quality Tags

Photos the model considers a reject get a quality tag line at the very top
of the description, giving you a searchable string for finding them:

```
**Blurry Photo**-8 **Subjectless Photo**-9
```

**Good photos get no tag at all.** The tag line appears only when the model
judges the photograph bad enough to discard — a score of **7 or above**. Any
lower score is treated as a keeper: the tag is stripped from the file
entirely, so the description is written but never shows up in a
`Subjectless Photo` search. Most photos in a normal library are keepers, so
the default output is an untagged description.

The two tags are independent — either, both, or neither:

| Tag | Meaning |
|---|---|
| `**Blurry Photo**` | The **subject** is out of focus, or the photo shows camera shake, motion blur, or the subject is unrecognisable because of blur. Sharpness is judged on the subject; a deliberately blurred background from a shallow depth of field is not blur |
| `**Subjectless Photo**` | Nothing in the frame can be named or recognised as the thing the photographer meant to capture |

The number is how sure the model is that the photo is a reject:

| Score | Meaning |
|---|---|
| `10` | Certain reject |
| `8`–`9` | Likely reject |
| `7` | Probably reject |
| `1`–`6` | Keeper — tag is removed |

A high score is a strong signal the photo can be reviewed last or skipped.
The threshold is `QUALITY_TAG_MIN_SCORE` near the top of
`describe_images.py`; raise it to 8 or 9 to be stricter, lower it to 6 to
catch more.

### What is deliberately not a reject

This is a family library — candid snapshots from point-and-shoot cameras and
phones, plus scanned film. Much of it is old, grainy, soft or dim, and that
is normal. The prompt therefore asks the model for photos that are genuinely
**unusable**, not imperfect ones, and it is told to stay quiet when in doubt
(a missed tag costs far less than a good photo being wrongly buried).

The model is asked one question it must be able to answer **yes** to:

> Could you crop or edit this photo into something worth keeping? If not, it is
> a reject.

Deliberately kept as keepers:

- An empty landscape is still a subject. A beach, forest, mountainscape,
  sunset, cityscape or blank wall is a keeper.
- **Candid snapshots of ordinary life.** Children playing outside on a cloudy
  day, a family around a table, pets, a garden — if you can make out who or
  what is in it, it is a keeper. Wind, mid-laugh faces and a little softness
  do not make it unusable.
- **Film scans.** Heavy grain, colour cast, low contrast, dust specks, a soft
  corner or a crooked scan are part of an old photograph, not defects. Grain
  is never a reason to tag.
- **Low light or high-ISO / pushed film** that is noisy and dim but still
  readable. Noise plus softness alone is not enough; tag only if the content
  genuinely cannot be made out.
- **Shallow depth of field.** A blurred background, bokeh, or a soft
  foreground is deliberate photography. Sharpness is judged on the subject —
  if the subject is sharp, the photo is not a `**Blurry Photo**` no matter how
  much of the frame is out of focus.
- A **night sky is a subject**. Stars, the Milky Way, an aurora or the moon
  that are reasonably sharp count even when small and scattered from a
  hand-held camera — this is the case that motivated the confidence score.
  A sky is only rejected when it's genuinely unusable: a black frame, the
  lens cap still on, heavy star trailing, or no identifiable light at all.
- A small, distant, unusual or partially cropped subject is still a subject.

What does get flagged is a photo whose **whole** frame is spoiled:

- The entire frame smeared by camera shake, with no sharp edge anywhere.
- A thumb or finger over the lens, especially over a black or smeared frame.
- A shot fired by accident mid-stride or while pocketing the camera — the
  ground, a ceiling or a car floor filling the frame with nothing recognisable.
  Tagged `**Subjectless Photo**`, plus `**Blurry Photo**` if it's smeared.
- The lens cap still on, a fully black/white frame, or a crop so tight it is an
  unrecognisable patch of texture.
- A night sky that is nothing but heavy star trailing.

### Searching

Because tags are anchored to the first line and normalized to one exact
format, they can be searched directly:

```bash
# every rejected photo
grep -rl "Subjectless Photo" ~/photos

# only high-confidence rejects
grep -rl "Subjectless Photo.**-[89]" ~/photos
grep -rl "Subjectless Photo.**-10" ~/photos

# photos that got no tag at all (keepers)
grep -rL "Photo\*\*-" ~/photos
```

Every tag is rewritten to the canonical form regardless of how the model
originally wrote it (a tag at the bottom of the response, written as
`Blurry Photo: 8`, or scored out of `10` all come out identical), so a search
never misses a match because of formatting drift.

Descriptions written before this feature was added have no tag line. Re-run
with `--overwrite` to add them.

### Spotting rejects while it runs

When a photo is written with a quality tag, the warning is echoed on the
console directly under the filename, so a long run can be scanned or piped
to `grep WARNING` without opening a single file:

```
[1/4] 2013/March Vacation/IMG00023.jpg
    Sending to model...
    CREATED: IMG00023.txt
    WARNING: **Blurry Photo**-8 **Subjectless Photo**-9
[2/4] 2013/March Vacation/IMG00024.jpg
    Sending to model...
    CREATED: IMG00024.txt
```

Photos that are keepers print no warning line at all. The configured
threshold is also shown in the run header:

```
Tag at    : 7+ (bad photos only)
```

To review rejects from a finished run:

```bash
python describe_images.py "D:\Photos" | grep -B2 WARNING
```

### Finding tagged photos

Once descriptions exist, three switches search them for tags. These read
only the `.txt` files — no model is loaded and no network call is made, so
a whole library takes seconds.

```bash
# photos tagged as blurry
python describe_images.py "D:\Photos" --find-blurry

# photos tagged as subjectless
python describe_images.py "D:\Photos" --find-subjectless

# either kind, all scores
python describe_images.py "D:\Photos" --find-blurry --find-subjectless
```

Each match prints the full path of the image, with the tags that matched:

```
D:\Photos\2013\March Vacation\IMG00023.jpg  **Blurry Photo**-8 **Subjectless Photo**-9
D:\Photos\2014\Beach\IMG00041.jpg  **Subjectless Photo**-10
```

A photo tagged with both appears under either search. A summary follows the
results:

```
Matched   : 2
Tagged    : 5 total, 3 outside 9+
Scanned   : 412 images
```

#### Reviewing in two passes

The scores run 7-10, which splits naturally into two passes. Use
`--min-confidence` with `--max-confidence` to select a band; the two bands
below never overlap, so nothing is reviewed twice or missed between them.

**Pass 1 — the obvious garbage (9-10).** These are the lens-cap-on,
finger-over-the-lens, total-smear photos. Bulk delete them in Explorer:

```bash
python describe_images.py "D:\Photos" \
    --find-blurry --find-subjectless --min-confidence 9
```

**Pass 2 — the borderline ones (7-8).** These are worth a human eye, and are
where your hand-held star fields will land. Look at them one at a time and
keep anything you'd miss:

```bash
python describe_images.py "D:\Photos" \
    --find-blurry --find-subjectless \
    --min-confidence 7 --max-confidence 8
```

Because each line is a bare path, pass 1 output pastes straight into an
Explorer address bar or search box. But if you review in dedicated image
software, it's easier to let the script mark the files for you — see
[Marking photos with `--rename-tagged`](#marking-photos-with---rename-tagged)
below.

For pass 2, a plain list is usually easier to work through top to bottom:

```bash
python describe_images.py "D:\Photos" \
    --find-blurry --find-subjectless \
    --min-confidence 7 --max-confidence 8
```

Or count a band without listing it:

```bash
python describe_images.py "D:\Photos" --find-blurry | find /c /v ""
```

### Marking photos with `--rename-tagged`

If your review workflow happens in image software rather than Explorer, you
don't have to hunt the files down one by one. `--rename-tagged` marks each
reject in its own filename, so you can point your software at the folder and
sort or filter on the name:

```bash
# preview first - nothing is touched
python describe_images.py "D:\Photos" \
    --find-blurry --find-subjectless --min-confidence 9 \
    --rename-tagged --dry-run

# then for real
python describe_images.py "D:\Photos" \
    --find-blurry --find-subjectless --min-confidence 9 \
    --rename-tagged
```

```
    RENAMED: IMG00023.jpg -> IMG00023.blurry.jpg
    RENAMED: IMG00041.jpg -> IMG00041.nosubject.jpg
    RENAMED: IMG00052.jpg -> IMG00052.blurry.nosubject.jpg
```

Only the images are renamed — the pixels are never touched. The description
`.txt` is renamed to match (`IMG00023.blurry.jpg` + `IMG00023.blurry.txt`), so
the pair stays together and `--find-blurry` keeps working afterwards.

A photo with both tags gets both markers. A photo matched on only one tag
gets only that marker, even if its other tag scored higher — matching
follows the band you asked for, not the worst score overall.

**Undo** by stripping the markers back out:

```bash
python describe_images.py "D:\Photos" --revert-renames
```

```
    REVERTED: IMG00023.blurry.jpg -> IMG00023.jpg
```

This works purely off filenames, so it needs no tag or score filter.

Safety details:

- **`--dry-run` works here too.** Preview any rename first and nothing on
  disk changes.
- **Never overwrites.** If `IMG00023.blurry.jpg` already exists, that file
  is skipped with a message and both files are left alone.
- **Never stacks markers.** Running twice does not produce
  `IMG00023.blurry.blurry.jpg`; an already-marked file is skipped.
- **Undo it as many times as you like** — a name with no marker is left
  alone.

The one risk worth knowing: if you had a photo genuinely named
`something.nosubject.jpg` *before* running `--revert-renames`, it will also
be renamed to `something.jpg`. Reverting cannot tell a marker it added from
one you always had. Use `--dry-run` and read the list before reverting if
your library has such names.

Notes:

- Scores below 7 never got a tag written in the first place, so
  `--min-confidence` can only narrow within 7-10. There is nothing to find
  below 7.
- A photo with two tags at different scores (say `**Blurry Photo**-8
  **Subjectless Photo**-7`) matches a band if *either* tag falls in it, so
  the printed tag list is what tells you which band it landed in.
- Photos with no `.txt` file yet are skipped. `Scanned : N` is the
  denominator, so you can see how many were missed.

## Requirements

- Python 3.x
- `requests` library
- One of:
  - [Ollama](https://ollama.com/) for local inference (default)
  - An API key for a cloud provider with an OpenAI-compatible vision API

## Installation

Install the required Python library:

```bash
python -m pip install requests
```

Install and start Ollama with a vision model:

```bash
ollama pull llava:7b
```

## Usage

Basic usage:

```bash
python describe_images.py "path/to/photos"
```

### Options

| Option | Description |
|---|---|
| `--dry-run` | Preview what would be processed without making LLM calls |
| `--no-recursive` | Only process images in the specified directory (don't include subdirectories) |
| `--model` | Vision model to use (default: `llava:13b`) |
| `--api-url` | API endpoint URL (default: Ollama's local URL) |
| `--api-key` | API key for cloud providers; also reads `OPENAI_API_KEY` env var |
| `--timeout` | Request timeout in seconds (default: 300) |
| `--retries` | Number of attempts for failed requests (default: 3) |
| `--overwrite` | Regenerate descriptions for images that already have a `.txt` file |
| `--find-blurry` | Print the image paths tagged `**Blurry Photo**` |
| `--find-subjectless` | Print the image paths tagged `**Subjectless Photo**` |
| `--min-confidence N` | Only report tags scoring N or higher (default: 7) |
| `--max-confidence N` | Only report tags scoring N or lower (default: 10) |
| `--rename-tagged` | Rename matching photos to mark them (`--dry-run` supported) |
| `--revert-renames` | Strip `.blurry` / `.nosubject` markers from filenames |

### Examples

Preview without processing:

```bash
python describe_images.py "D:\Photos" --dry-run --no-recursive
```

Use a different local model for better descriptions:

```bash
python describe_images.py "D:\Photos" --model qwen2.5vl:7b
```

Use a cloud API (OpenAI) instead of local Ollama:

```bash
python describe_images.py "D:\Photos" \
    --api-url https://api.openai.com/v1/chat/completions \
    --api-key sk-your-key-here \
    --model gpt-4o
```

Or set the key via environment variable:

```bash
set OPENAI_API_KEY=sk-your-key-here
python describe_images.py "D:\Photos" \
    --api-url https://api.openai.com/v1/chat/completions \
    --model gpt-4o
```

## Model Recommendations

| Model | Speed | Quality | Notes |
|---|---|---|---|
| `llava:7b` | ⚡ Fastest | Good | Recommended for large batches |
| `llava:13b` | ⚡ Fast | ⚡ Excellent | Good balance |
| `qwen2.5vl:7b` | 🐢 Slow | ⚡⚡ Best | ~2x slower than llava:13b, highest quality descriptions |
| `qwen3-vl` | ❌ Times out | — | Not recommended for RTX 3060 (12GB) |

## License

LGPL-2.1
