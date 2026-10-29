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

The earlier example was able to name the children, Jack and Jill, because they were the only children in the photo and the LLM could substitute boy and girl for their actual names based on the context provided.  I added a face recognition module to reliably identify and name people in photos when a face database is available.

Using gemma4:e4b and the face recognition module, the system can now fairly reliably identify and name individuals in photos, provided their faces have been enrolled in the face database.

Below is a description from a photo of a man and 2 adult children in a lounge on a ship.

```
A family of three—Jill, Jim, and Jack—is posing for a photograph indoors. They are standing behind a low, artificial rock or stone ledge. Jill, on the left, is wearing a black top and blue jeans. Jim, in the center, is wearing a dark blue short-sleeved shirt over a lighter grey shirt and blue jeans. Jack, on the right, is wearing a royal blue t-shirt and black shorts.

The setting is a modern, spacious interior lobby or atrium, characterized by a large, curved, illuminated ceiling fixture. Behind the group, the room is heavily decorated with lush, tropical foliage and large potted plants, including various types of palms and broad-leafed greenery. The background wall consists of expansive glass windows that offer a scenic view of a body of water (likely the ocean or a large bay) and distant urban buildings or resort structures. The floor is light-colored wood or tile. To the right, visible through the window and positioned in the foreground of the seating area, are several rows of uniform, light green, slatted lounge chairs or furniture groupings. A small green exit sign is visible on the wall between the foliage and the window view. The overall environment suggests a warm, tropical location with modern architectural design.
```

## Features

- Iterates through all images in a specified directory
- Calls a local LLM (via Ollama) or cloud API (OpenAI-compatible) to generate image descriptions
- Supports multiple vision models with different speed/quality tradeoffs
- Face recognition: enroll family faces once with `--enroll`, then descriptions
  use real names ("Phoebe is cooking") instead of "a woman is cooking"
- Score face identification against a labeled test set (`evaluate_faces.py`)
- Tags photos with a blurry and/or subjectless quality concern on a 5-10
  scale; the default reject search is 8+
- Find tagged photos later without calling the LLM again
- Mark rejects in the filename so image software can sort or filter on it,
  with a one-command undo
- Dry-run mode to preview operations
- Optional recursive directory processing
- Overwrite mode to regenerate existing descriptions

## Face Identification

### Default face database

One default pair, everywhere (Python and the `.bat` wrappers alike):

| Piece | Default |
|---|---|
| Enrollment folder convention | **`./faces/`** — one subfolder per person |
| Database file | **`faces.pkl`**, next to `describe_images.py` |

```bash
python describe_images.py --enroll faces/    # reads ./faces/*, writes faces.pkl
python describe_images.py "D:\Photos"        # loads faces.pkl automatically
```

Anything else is an **override**: pass your own folder and/or `--face-db`
(e.g. `my_friends/` → `my_friends.pkl`, a school-play cast, several DBs
merged). There is no second hidden default.

Two modes, plus an override for one-off sets of faces:

| Mode | When | How |
|---|---|---|
| **Off** | No face DB, or explicitly | `--no-face-id` (or simply never enroll) |
| **Family** (default) | `faces.pkl` exists next to the script | one-time `--enroll faces/`, then plain describe runs |
| **Override** | A special photo set with extra/other faces | separate `--enroll` into another `--face-db`, used only for that run |

Model choice stays yours (`--model`); these modes only control face naming.

### Mode 1 - no face detection

```bash
python describe_images.py "D:\Photos" --no-face-id
```

```
Face ID   : off (no face DB, or --no-face-id)
```

### Mode 2 - family faces (default enrollment)

Organize reference photos into one folder per person — 5-10 photos each,
across different angles, lighting, years, glasses on/off. Each reference
photo must contain **exactly one face** (multi-face photos are skipped):

```
faces/
  Phoebe/  1.jpg 2.jpg 3.jpg ...
  John/    ...
```

```bash
python describe_images.py --enroll faces/
```

This writes **`faces.pkl`** next to `describe_images.py`. Every later describe
run with no `--face-db` picks it up automatically:

```
Phoebe: 14 reference photos
John: 7 reference photos
Saved 21 encodings to L:\...\faces.pkl
```

### Describe with names

```bash
python describe_images.py "D:\Photos"
```

```
Face ID   : L:\...\faces.pkl (tolerance 0.55, model hog)

[1/3] party.jpg
    Faces: John (0.13), Phoebe (0.11)
    Sending to model...
    CREATED: party.txt
```

### Override enrollment - temporary faces

For a photo set that contains people who should **not** join the permanent
family DB (a children's school play, a one-off party), enroll them
separately and point that run at the extra DB:

```bash
# one-time: learn the play cast into their own file
python describe_images.py --enroll known-play/ --face-db faces-play.pkl

# describe only the play photos with the play cast
python describe_images.py "D:\Photos\school play" --face-db faces-play.pkl
```

`--face-db` replaces the default for that run — the family DB is not
loaded, so ordinary photos elsewhere are unaffected (they keep using
`faces.pkl` in their own run).

If the special photos also contain family members (parents in the
audience), **merge** by passing both DBs:

```bash
python describe_images.py "D:\Photos\school play" \
    --face-db faces.pkl --face-db faces-play.pkl
```

```
Face ID   : L:\...\faces.pkl, L:\...\faces-play.pkl (tolerance 0.55, model hog)
```

Names from every listed DB are combined before matching. Because describe
skips images that already have a `.txt`, you can process a special
subdirectory first with its override, then run the whole library with the
default family DB.

For each photo, faces are detected, matched against the enrolled people,
and the result is passed to the model as JSON with the bounding boxes and
instructions to use those names:

```json
[
  {"name": "John",   "x1": 324, "y1": 263, "x2": 510, "y2": 449, "distance": 0.129},
  {"name": "Phoebe", "x1": 613, "y1": 910, "x2": 835, "y2": 1133, "distance": 0.114},
  {"name": "unknown", "x1": 464, "y1": 390, "x2": 687, "y2": 613, "distance": 0.775}
]
```

`distance` is the match distance — lower is more confident. Faces over the
tolerance come back as `"unknown"` and are described generically (no
invented names). If two faces both match the same person, only the closer
one keeps the name.

Example descriptions:

| Detection | Description |
|---|---|
| John + Phoebe | "The upper section shows **John**... the lower section is a portrait of **Phoebe**" |
| Phoebe + unknown | "**Phoebe** and an unknown young man are posing together..." |

### Face options

| Option | Description |
|---|---|
| `--enroll KNOWN_DIR` | Learn faces from `KNOWN_DIR/<Name>/*.jpg`, write the DB, exit |
| `--face-db PATH` | Repeatable. Describe: omit for family default, one path to replace, several to merge. `--enroll`: at most one destination |
| `--face-tolerance T` | Max match distance, lower = stricter (default: 0.55). Lower to ~0.5 if strangers get named; raise further only with care |
| `--face-model hog\|cnn` | `hog` is fast (default); `cnn` is slower but finds angled/profile faces (use `--face-model cnn` for group or candid shots) |
| `--face-upsample N` | Upsample before detection (default: 1); try 2 for small faces in high-res images |
| `--no-face-id` | Skip face identification even when a face DB exists |

If you switch `--face-model`, re-run `--enroll` with the same model.

### Which vision model to use

Face names only appear in the description if the vision model actually
follows the naming instruction. Smaller models often ignore it:

| Model | Uses enrolled names? |
|---|---|
| `gemma4:e4b`, `gemma4:12b`, `qwen3.5:latest` | Yes |
| `llava:7b`, `llava:13b`, `qwen2.5vl:3b` | Usually no |

### Requirements

Face identification needs the `face_recognition` package plus `pillow` and
`numpy` — all listed in `requirements.txt`:

```bash
python -m pip install -r requirements.txt
# or:  pip install face_recognition pillow numpy
```

See [Installation](#installation) for dlib build issues on Windows/macOS.
Without these packages describe runs still work — they print a warning and
skip face naming.

## Photo Quality Tags

Photos the model considers a reject get a quality tag line at the very top
of the description, giving you a searchable string for finding them:

```
**Blurry Photo**-8 **Subjectless Photo**-9
```

**Good photos get no tag at all.** The tag line appears only when the model
judges the photograph as having a quality concern — a score of **5 or above**.
The default reject search is **8+**, while scores 5-7 are retained for a
professional-quality review. Most photos in a family library should be kept,
so use `--min-confidence 8` for the normal deletion pass or
`--min-confidence 5` when professional standards are wanted.

The two tags are independent — either, both, or neither:

| Tag | Meaning |
|---|---|
| `**Blurry Photo**` | The photo has meaningful out-of-focus, camera-shake, or motion blur. Slight softness in one person or a recognizable scene is at most a borderline 5-6; 8+ means the whole image is not useful as a family photo |
| `**Subjectless Photo**` | Nothing in the frame can be named or recognised as the thing the photographer meant to capture |

The number is how sure the model is that the photo is a reject:

| Score | Meaning |
|---|---|
| `10` | Unusable; nothing useful can be made |
| `8`–`9` | Whole image is genuinely unusable; default delete/review |
| `7` | Serious defect, but a recognizable family photo may be kept |
| `5`–`6` | Borderline/professional concern; usually keep in a family library |
| `1`–`4` | No meaningful quality problem; no tag |

A high score is a strong signal the photo can be reviewed last or skipped.
The writer preserves scores from `QUALITY_TAG_MIN_SCORE` (5) upward, while
the default search threshold is `DEFAULT_MIN_CONFIDENCE` (8) near the top of
`describe_images.py`. Use `--min-confidence 5` for the stricter professional
workflow.

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

The scores run 5-10, which supports a family-library pass and a stricter
professional pass. Use
`--min-confidence` with `--max-confidence` to select a band; the two bands
below never overlap, so nothing is reviewed twice or missed between them.

**Pass 1 — the obvious garbage (8-10).** These are the lens-cap-on,
finger-over-the-lens, total-smear photos. Bulk delete them in Explorer:

```bash
python describe_images.py "D:\Photos" \
    --find-blurry --find-subjectless --min-confidence 8
```

**Professional pass — borderline concerns (5-7).** These are recognizable
photos that a professional may reject but a family may want to keep:

```bash
python describe_images.py "D:\Photos" \
    --find-blurry --find-subjectless \
    --min-confidence 5 --max-confidence 7
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
- One of:
  - [Ollama](https://ollama.com/) for local inference (default)
  - An API key for a cloud provider with an OpenAI-compatible vision API

Python packages (see `requirements.txt`):

| Package | Needed for |
|---|---|
| `requests` | Calling Ollama or a cloud API (always) |
| `face_recognition`, `pillow`, `numpy` | Face identification (`--enroll`, face naming) — optional; without them describe runs still work and just skip names |

## Installation

Install the required Python libraries:

```bash
python -m pip install -r requirements.txt
```

Or just the minimum for describing without faces:

```bash
python -m pip install requests
```

Face identification needs dlib, which is pulled in by `face_recognition`.
If `pip install face_recognition` fails to build dlib, install it first:

```bash
# Windows / macOS - prebuilt wheel
pip install dlib-bin

# or via conda (any OS)
conda install -c conda-forge dlib
pip install face_recognition
```

If you see a misleading *"Please install face_recognition_models"* error on
Python 3.12+ with setuptools ≥ 81, it is a `pkg_resources` issue: install
`setuptools<81`, or use the patched `face_recognition_models` import that
resolves model paths without `pkg_resources`.

Install and start Ollama with a vision model:

```bash
ollama pull llava:7b
```

For face naming, also pull a model that follows name instructions
(`gemma4:e4b` is a good choice — see
[Model Recommendations](#model-recommendations)):

```bash
ollama pull gemma4:e4b
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
| `--enroll KNOWN_DIR` | Learn faces from `KNOWN_DIR/<Name>/*.jpg` into the face DB, then exit |
| `--face-db PATH` | Repeatable: omit for family default, one path to replace, several to merge (also the `--enroll` destination) |
| `--face-tolerance T` | Max face match distance, lower = stricter (default: 0.55) |
| `--face-model hog\|cnn` | Face detector: `hog` fast (default), `cnn` better for angled faces |
| `--face-upsample N` | Upsample images before face detection (default: 1) |
| `--no-face-id` | Skip face identification even when a face DB exists |
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

## Convenience scripts (.bat)

Windows users can run these short `.bat` files instead of typing the full
Python command. **They are only wrappers** — every script has an equivalent
`python ...` invocation shown alongside it, which works the same on macOS
and Linux. Face defaults are the same as plain Python: folder `./faces`,
database `faces.pkl`.

| Script | What it does |
|---|---|
| `processphotos.bat` | Describe a folder with the default vision model |
| `processfaces.bat` | Build a face database (`.pkl`) from a folder of known people |
| `processphotos_faces.bat` | Describe a folder with face names enabled |
| `renamebadphotos.bat` | Mark rejected photos in their filenames |
| `revertbadphotos.bat` | Strip those marks back out |

### `processfaces.bat` — enroll faces

```bat
processfaces.bat                        REM .\faces  ->  faces.pkl
processfaces.bat my_friends             REM my_friends -> my_friends.pkl
processfaces.bat my_friends friends.pkl REM custom folder and output
```

No arguments: enroll **`.\faces`** into **`faces.pkl`** — the same defaults
as `python describe_images.py --enroll faces/`. The output name follows the
folder (`my_friends` → `my_friends.pkl`) unless you pass a second argument,
so several groups can sit side by side.

Equivalent on any OS:

```bash
# default: enroll ./faces into faces.pkl
python describe_images.py --enroll faces/

# named override (any name you like)
python describe_images.py --enroll my_friends --face-db my_friends.pkl
```

### `processphotos_faces.bat` — describe with face names

```bat
processphotos_faces.bat "D:\Photos"                              REM faces.pkl
processphotos_faces.bat "D:\Photos\school play" friends.pkl      REM override
processphotos_faces.bat "D:\Photos\school play" faces.pkl friends.pkl  REM merge
```

Extra arguments after the photo directory are treated as face databases
and merged. With no face DB arguments, none is passed — the run uses
`describe_images.py`'s own default (`faces.pkl`). Model defaults to
`gemma4:e4b` (a model that reliably follows the face-name instructions —
see [Which vision model to use](#which-vision-model-to-use)).

Equivalent on any OS:

```bash
# exactly what processphotos_faces.bat runs with no extra arguments
python describe_images.py "path/to/photos" --model gemma4:e4b

# explicit face DB (override)
python describe_images.py "path/to/photos" --model gemma4:e4b \
    --face-db friends.pkl

# merge several face DBs
python describe_images.py "path/to/photos" --model gemma4:e4b \
    --face-db faces.pkl --face-db friends.pkl
```

### The other scripts

```bat
processphotos.bat "D:\Photos"
```

```bash
python describe_images.py "path/to/photos" --model llava:13b
```

```bat
renamebadphotos.bat "D:\Photos"
revertbadphotos.bat "D:\Photos"
```

```bash
python describe_images.py "path/to/photos" \
    --find-blurry --find-subjectless --rename-tagged
python describe_images.py "path/to/photos" --revert-renames
```

## Model Recommendations

| Model | Speed | Quality | Notes |
|---|---|---|---|
| `llava:7b` | ⚡ Fastest | Good | Large batches **without** face naming |
| `llava:13b` | ⚡ Fast | ⚡ Excellent | Good balance **without** face naming |
| `gemma4:e4b` | ⚡ Fast | ⚡ Excellent | **Use with a face DB** — follows name instructions reliably |
| `gemma4:12b` | 🐢 Slow | ⚡⚡ Best | **Use with a face DB** when quality matters more than speed |
| `qwen3.5:latest` | 🐢 Slow | ⚡⚡ Best | **Use with a face DB** — also follows name instructions |
| `qwen2.5vl:7b` | 🐢 Slow | ⚡⚡ Best | ~2x slower than llava:13b; **ignores face names** |
| `qwen3-vl` | ❌ Times out | — | Not recommended for RTX 3060 (12GB) |

**When a face database is loaded** (a `Face ID:` line appears in the run
header, or a `--face-db` was passed), pick **`gemma4:e4b`** (default of
`processphotos_faces.bat`), **`gemma4:12b`**, or **`qwen3.5:latest`**.
The llava and qwen2.5vl models typically receive the face JSON but do not
put the names into the description — you get "a woman" instead of "Phoebe"
even when detection succeeded. See
[Which vision model to use](#which-vision-model-to-use).

With no face DB, any model in the table is fine; choose on speed vs.
description quality alone.

## License

LGPL-2.1
