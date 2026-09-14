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
- Dry-run mode to preview operations
- Optional recursive directory processing
- Overwrite mode to regenerate existing descriptions

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
