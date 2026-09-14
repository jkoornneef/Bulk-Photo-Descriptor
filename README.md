# Bulk-Photo-Descriptor
A simple Python script that batch processes images in a directory, using a local LLM to generate descriptions for each image.

This is a quick script that is meant to run against a local instance of Ollama for generating image descriptions.
Goal is to be private and run entirely on your local machine.

The script iterates through all images in a specified directory and generates descriptions for each image using a local LLM.
It passes the path of the image to the prompt as it may contain contextual information useful for generating accurate descriptions.
i.e. c:/photos/2013/March Vacation/Walt Disney World/Water Park/IMG00023.jpg

Each directory can also have a context file named ```context.txt``` or ```context.md``` that provides additional information for generating image descriptions.
```
In 2013 we enjoyed our march vacation at Walt Disney World, specifically the Typhoon Lagoon water park.
Our oldest son, Jack, was particularly excited about the water slides and spent most of his time there.
Our youngest daughter, Jill, loved the lazy river and spent a lot of time floating around and splashing.
```

Result from LLM
```
The image depicts a relaxed scene at a waterpark, possibly at Walt Disney World. The focus is on three individuals who appear to be on a family vacation. The woman, is seated on a blue and yellow inflatable inner tube, with her legs extended and hands resting on her knees. She is wearing a black top and sunglasses.

Next to her, the boy, Jack, is also on an inner tube. He is wearing a blue shirt and has his legs folded underneath him. He is looking towards the camera with a slight smile.

The girl, Jill, is sitting on the edge of the pool adjacent to the inner tubes. She is wearing a pink shirt and appears to be watching the others with a smile.

The environment is a sunny day, with palm trees in the background, suggesting a warm and tropical climate. The pool is surrounded by a fence, indicating it's a controlled area within the waterpark. There are no visible texts, logos, or other identifiable landmarks in the image.
```



### Features

- Iterates through all images in a specified directory
- Calls a local LLM (via Ollama) to generate image descriptions
- Supports multiple vision models with different speed/quality tradeoffs
- Dry-run mode to preview operations
- Optional recursive directory processing

### Requirements
Python 3.x
Ollama (for running local LLMs)
```requests```
library

### Installation
Install the required Python library:
```python -m pip install requests```
Install and start Ollama with a vision model:
```ollama pull llava:7b```

### Usage
Basic usage:
```python describe_images.py "path/to/photos"```

### Options
```--dry-run```
- Preview what would be processed without making changes

```--no-recursive```
- Only process images in the specified directory (don't include subdirectories)

```--model```
- Specify which vision model to use (default:llava:7b)

#### Examples
- Preview without processing:
    - ```python describe_images.py "D:\Photos" --dry-run --no-recursive```

- Use a different model for better descriptions:
    - ```python describe_images.py "D:\Photos" --model qwen2.5vl:7b```

- Use a larger model for balanced speed/quality:
    - ```python describe_images.py "D:\Photos" --model llava:13b```

### Model Recommendations

Model Recommendations
Model	Speed	Quality	Notes
llava:7b
⚡ Fastest	Good	Recommended for large batches
llava:13b
⚡⚡ Fast	⚡⚡ Excellent	Good balance
qwen2.5vl:7b
⚡⚡ Fast	⚡⚡⚡ Best	Highest quality descriptions
qwen3-vl
❌ Times out	—	Not recommended for RTX 3060 (12GB)
License
LGPL-2.1