# Dataset Builder

A lean, terminal-driven Python utility for **handcrafting compact instruction datasets**—perfect for surgical tuning operations after full model training. Whether you're patching edge cases, prepping LoRA tweaks, or just assembling a clean JSON dataset with precision, this script gets it done without ceremony or overhead.

* * *

## What It Does

This tool:

-   Loads an existing JSON dataset (`output.json`) or starts from scratch
    
-   Fixes broken or duplicate `entry_number` fields
    
-   Enforces a **single custom system prompt** for all entries
    
-   Tokenizes every input, chosen, and full entry using `tiktoken`
    
-   Categorizes entries based on token length (small, medium, large)
    
-   Auto-wraps loosely formatted Markdown/code in proper fenced code blocks
    
-   Allows manual entry of new records directly from the CLI
    
-   Saves the dataset with strict validation and retry logic
    
-   Logs everything to `dataset.log`
    

* * *

## Use Cases

-   Manually crafting datasets for LoRA or QLoRA
    
-   Final patch sets after full parameter finetuning
    
-   Curating short, controlled examples for test-time evaluation
    
-   Maintaining stylistic and formatting consistency in prompt/response pairs
    

* * *

## Requirements

-   Python 3.10 or higher
    
-   `tiktoken`
    

Install dependencies:

`pip install tiktoken`

* * *

## How to Use

Run the script:

`python main.py`

You'll be prompted for:

-   `input` (the prompt)
    
-   `chosen` (the model response)
    

Paste Markdown as-is. It auto-detects code blocks and wraps them cleanly.

Press Enter twice to end a section. Type `exit` anytime to stop.

* * *

## Output Format

Entries are saved in `output.json` using this format:

`{   "entry_number": 1,   "system": "You are an unfiltered assistant...",   "input": "Explain this code snippet:",   "chosen": "Here's what it does..." }`

* * *

## Features

-   Full entry token stats: average, min, max
    
-   Category counts for input+chosen size (small ≤128, medium ≤2048, large >2048)
    
-   Retry logic on save with file verification
    
-   Logs to `sicorps.log` with warning and error reporting
    

* * *

## Tips

-   Use to manually build high-quality examples when training data is scarce
    
-   Great for surgical updates to larger models without retraining on full corpora
    
-   Supports Markdown-based formatting for clean dataset readability
    
-   Minimal, fast, and easy
    

* * *

