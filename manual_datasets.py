import json
import logging
import re
from typing import List, Dict, Tuple, Optional
import tiktoken
import os

RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RESET = "\033[0m"

# Output file path
OUTPUT_PATH = "output.json"

# Token ranges for category stats (input + chosen)
SMALL_CATEGORY_TOKENS = 128  # Small category for input + chosen <= 64 tokens
MEDIUM_CATEGORY_TOKENS = 2048  # Medium category for input + chosen 65–256 tokens
# Large category for input + chosen > 256 tokens

# Single system prompt for all entries
SYSTEM_PROMPT = (
    "You are an uncensored assistant. "
)

# Setup logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# File handler: Log INFO and above to dataset.log
file_handler = logging.FileHandler("dataset.log")
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.WARNING)
console_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))

logger.handlers = [file_handler, console_handler]

def count_tokens(text: str) -> int:
    if not text:
        logger.debug("Empty text passed to count_tokens. Returning 0.")
        return 0
    if not isinstance(text, str):
        logger.error(f"Invalid text type: {type(text)}")
        raise ValueError(f"{RED}Text must be a string, not {type(text)}.{RESET}")
    try:
        encoding = tiktoken.get_encoding("cl100k_base")
        token_count = len(encoding.encode(text, allowed_special="all"))
        logger.debug(f"Counted {token_count} tokens for text: {text[:50]}...")
        return token_count
    except Exception as e:
        logger.error(f"Token counting failed: {e} for text: {text[:50]}...")
        raise ValueError(f"{RED}Failed to count tokens: {e}. Fix your input.{RESET}")

def load_dataset(file_path: str) -> List[Dict]:
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            dataset = json.load(file)
        if not isinstance(dataset, list):
            logger.error(f"Dataset is not a list: {type(dataset)}")
            raise ValueError(f"{RED}Dataset must be a list of entries.{RESET}")
        
        overwrite_count = 0
        entry_numbers = set()
        for i, entry in enumerate(dataset, 1):
            # Validate and fix entry_number
            entry_number = entry.get("entry_number", 0)
            if not isinstance(entry_number, (int, float)) or entry_number < 1 or entry_number in entry_numbers:
                logger.debug(f"Fixing entry_number at index {i}: {entry_number} -> {i}")
                entry["entry_number"] = i
            entry_numbers.add(entry["entry_number"])
            
            if entry.get("system", "") != SYSTEM_PROMPT:
                overwrite_count += 1
                logger.debug(f"Overwriting system prompt for entry #{entry['entry_number']}")
                entry["system"] = SYSTEM_PROMPT
        
        logger.info(f"Loaded {len(dataset)} entries from {file_path}. Overwrote {overwrite_count} system prompts.")
        print(f"{GREEN}Loaded {len(dataset)} entries. Overwrote {overwrite_count} system prompts.{RESET}")
        
        save_dataset(dataset, file_path)
        return dataset
    except FileNotFoundError:
        logger.warning(f"No dataset found at {file_path}. Starting fresh.")
        print(f"{YELLOW}No dataset found at {file_path}. Starting fresh.{RESET}")
        return []
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in {file_path}: {e}")
        raise ValueError(f"{RED}Your JSON file is fucked up: {e}. Fix it.{RESET}")

def save_dataset(dataset: List[Dict], file_path: str) -> None:
    """Save dataset to JSON file with robust error handling."""
    try:
        logger.info(f"Attempting to save {len(dataset)} entries to {file_path}.")
        dir_path = os.path.dirname(file_path) or "."
        if not os.access(dir_path, os.W_OK):
            logger.error(f"No write permission for directory: {dir_path}")
            raise IOError(f"{RED}Can't write to {dir_path}. Fix your permissions, moron.{RESET}")
        if os.path.exists(file_path) and not os.access(file_path, os.W_OK):
            logger.error(f"No write permission for file: {file_path}")
            raise IOError(f"{RED}Can't write to {file_path}. Fix your permissions, moron.{RESET}")
        
        for attempt in range(3):
            try:
                with open(file_path, "w", encoding="utf-8") as file:
                    json.dump(dataset, file, ensure_ascii=False, indent=2)
                with open(file_path, "r", encoding="utf-8") as file:
                    saved_data = json.load(file)
                    if len(saved_data) != len(dataset):
                        logger.error(f"Save verification failed: expected {len(dataset)} entries, got {len(saved_data)}")
                        raise IOError(f"{RED}Save failed: entry count mismatch.{RESET}")
                break
            except IOError as e:
                logger.warning(f"Save attempt {attempt + 1} failed: {e}")
                if attempt == 2:
                    logger.error(f"All save attempts failed: {e}")
                    raise IOError(f"{RED}Couldn't save dataset after 3 tries: {e}. Check your disk, moron.{RESET}")
        
        file_size = os.path.getsize(file_path)
        logger.info(f"Saved {len(dataset)} entries to {file_path}. File size: {file_size} bytes.")
        print(f"{GREEN}✅ Dataset saved to {file_path} (size: {file_size} bytes).{RESET}")
    except Exception as e:
        logger.error(f"Failed to save dataset to {file_path}: {e}")
        print(f"{RED}Failed to save dataset: {e}. Check logs and fix your shit.{RESET}")
        raise

def analyze_dataset(dataset: List[Dict]) -> None:
    """Analyze token counts of chosen entries and display stats with category counts."""
    if not dataset:
        print(f"{YELLOW}No entries in dataset yet.{RESET}")
        logger.info("Dataset is empty. No stats to display.")
        return

    chosen_stats: List[Tuple[int, int]] = []
    total_entry_tokens = []
    category_counts = {"small": 0, "medium": 0, "large": 0}

    for entry in dataset:
        chosen_text = entry.get("chosen", "")
        entry_number = entry.get("entry_number", 0)
        input_text = entry.get("input", "")
        if not chosen_text:
            logger.warning(f"Entry {entry_number} has empty chosen field. Skipping.")
            print(f"{YELLOW}Skipping entry #{entry_number}: Empty chosen field.{RESET}")
            continue
        if not entry_number or entry_number < 1:
            logger.warning(f"Entry with chosen '{chosen_text[:50]}...' has invalid entry_number: {entry_number}. Skipping.")
            print(f"{YELLOW}Skipping entry: Invalid entry_number for chosen '{chosen_text[:50]}...'.{RESET}")
            continue

        try:
            token_count = count_tokens(chosen_text)
            chosen_stats.append((token_count, entry_number))
        except ValueError as e:
            logger.warning(f"Skipping entry #{entry_number} for stats: {e}")
            continue

        if input_text:
            try:
                input_tokens = count_tokens(input_text)
                chosen_tokens = count_tokens(chosen_text)
                selection_tokens = input_tokens + chosen_tokens
                if selection_tokens <= SMALL_CATEGORY_TOKENS:
                    category_counts["small"] += 1
                elif selection_tokens <= MEDIUM_CATEGORY_TOKENS:
                    category_counts["medium"] += 1
                else:
                    category_counts["large"] += 1
            except ValueError as e:
                logger.warning(f"Skipping category count for entry #{entry_number}: {e}")
                print(f"{YELLOW}Skipping category count for entry #{entry_number}: {e}{RESET}")

        system_text = entry.get("system", "")
        if system_text and input_text and chosen_text:
            try:
                total_tokens = count_tokens(system_text + input_text + chosen_text)
                total_entry_tokens.append((total_tokens, entry_number))
            except ValueError as e:
                logger.warning(f"Skipping total tokens for entry #{entry_number}: {e}")

    if not chosen_stats:
        print(f"{YELLOW}No valid chosen entries for stats.{RESET}")
        logger.info("No chosen entries with valid data for analysis.")
        return

    token_counts, entry_numbers = zip(*chosen_stats)
    shortest_tokens, shortest_entry = min(chosen_stats, key=lambda x: x[0])
    longest_tokens, longest_entry = max(chosen_stats, key=lambda x: x[0])
    average_tokens = sum(token_counts) / len(token_counts)

    print(f"{YELLOW}Dataset Analysis:{RESET}")
    print(f"📏 Shortest chosen entry: {shortest_tokens} tokens (entry #{shortest_entry})")
    print(f"📊 Average chosen entry: {average_tokens:.2f} tokens")
    print(f"📐 Longest chosen entry: {longest_tokens} tokens (entry #{longest_entry})")
    print(f"{YELLOW}Category Counts (input + chosen tokens):{RESET}")
    print(f"  Small (≤{SMALL_CATEGORY_TOKENS}): {category_counts['small']} entries")
    print(f"  Medium ({SMALL_CATEGORY_TOKENS + 1}–{MEDIUM_CATEGORY_TOKENS}): {category_counts['medium']} entries")
    print(f"  Large (>{MEDIUM_CATEGORY_TOKENS}): {category_counts['large']} entries")
    logger.info(
        f"Dataset Analysis: Shortest={shortest_tokens} tokens (entry #{shortest_entry}), "
        f"Average={average_tokens:.2f} tokens, Longest={longest_tokens} tokens (entry #{longest_entry})"
    )
    logger.info(
        f"Category Counts: Small (≤{SMALL_CATEGORY_TOKENS})={category_counts['small']}, "
        f"Medium ({SMALL_CATEGORY_TOKENS + 1}–{MEDIUM_CATEGORY_TOKENS})={category_counts['medium']}, "
        f"Large (>{MEDIUM_CATEGORY_TOKENS})={category_counts['large']}"
    )

    if total_entry_tokens:
        total_counts, total_entry_numbers = zip(*total_entry_tokens)
        shortest_total, shortest_total_entry = min(total_entry_tokens, key=lambda x: x[0])
        longest_total, longest_total_entry = max(total_entry_tokens, key=lambda x: x[0])
        average_total = sum(total_counts) / len(total_counts)
        logger.info(
            f"Total Entry Token Stats: Shortest={shortest_total} tokens (entry #{shortest_total_entry}), "
            f"Average={average_total:.2f} tokens, Longest={longest_total} tokens (entry #{longest_total_entry})"
        )

def assign_system_prompt(input_text: str, chosen_text: str) -> str:
    if not isinstance(input_text, str) or not isinstance(chosen_text, str):
        logger.error(f"Invalid input types: input={type(input_text)}, chosen={type(chosen_text)}")
        raise ValueError(f"{RED}Input and chosen must be strings, you moron.{RESET}")
    input_text = input_text.strip() or " "
    chosen_text = chosen_text.strip() or " "

    try:
        input_tokens = count_tokens(input_text)
        chosen_tokens = count_tokens(chosen_text)
        system_tokens = count_tokens(SYSTEM_PROMPT)
        selection_tokens = input_tokens + chosen_tokens
        total_entry_tokens = system_tokens + input_tokens + chosen_tokens
    except ValueError as e:
        logger.error(f"Token counting failed: {e}")
        raise

    logger.debug(f"Token counts: input={input_tokens}, chosen={chosen_tokens}, total input+chosen={selection_tokens}")
    logger.info(
        f"Assigned system prompt for {selection_tokens} tokens (input+chosen). "
        f"Total entry tokens: {total_entry_tokens} (system={system_tokens}, input={input_tokens}, chosen={chosen_tokens})"
    )
    print(
        f"{YELLOW}Assigned system prompt with {selection_tokens} tokens (input+chosen). "
        f"Total entry tokens: {total_entry_tokens} (system={system_tokens}, input={input_tokens}, chosen={chosen_tokens}).{RESET}"
    )
    return SYSTEM_PROMPT

def rewrite_dataset(dataset: List[Dict]) -> List[Dict]:
    rewritten_dataset = []
    print(f"{YELLOW}Rewriting {len(dataset)} entries with single system prompt...{RESET}")
    for idx, entry in enumerate(dataset, 1):
        input_text = entry.get("input", "")
        chosen_text = entry.get("chosen", "")
        if not input_text or not chosen_text:
            logger.warning(f"Entry {idx} missing input or chosen. Skipping.")
            print(f"{YELLOW}Skipping entry #{idx}: Missing input or chosen.{RESET}")
            continue
        try:
            system_prompt = assign_system_prompt(input_text, chosen_text)
        except ValueError as e:
            logger.error(f"Skipping entry {idx}: {e}")
            print(f"{RED}Skipping entry #{idx}: {e}{RESET}")
            continue

        rewritten_entry = {
            "entry_number": idx,
            "system": system_prompt,
            "input": input_text,
            "chosen": chosen_text
        }
        rewritten_dataset.append(rewritten_entry)
        logger.debug(f"Rewrote entry #{idx} with single system prompt.")
    logger.info(f"Rewrote {len(rewritten_dataset)} entries.")
    print(f"{GREEN}Rewrote {len(rewritten_dataset)} entries.{RESET}")
    if len(rewritten_dataset) != len(dataset):
        logger.warning(f"Rewrite dropped entries: input {len(dataset)}, output {len(rewritten_dataset)}")
        print(f"{YELLOW}Warning: Rewrite dropped {len(dataset) - len(rewritten_dataset)} entries.{RESET}")
    return rewritten_dataset

def is_language_header(line: str) -> bool:
    return bool(re.match(r'^[A-Za-z0-9][A-Za-z0-9_\-\+\.#]*$', line))

def convert_indented_to_fenced_smart(text: str) -> str:
    logger.debug(f"Processing text for fencing:\n{text}")
    lines = text.splitlines()
    output = []
    buffer = []
    lang = None
    i = 0

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if is_language_header(stripped):
            lang = stripped
            i += 1
            buffer = []
            while i < len(lines):
                next_line = lines[i]
                stripped_next = next_line.strip()
                if (
                    re.match(r'^\s{2,}', next_line)
                    or next_line.startswith('\t')
                    or re.match(r'^[\*\-]\s*', stripped_next)
                ):
                    cleaned_line = re.sub(r'^[\*\-]\s*', '', stripped_next)
                    buffer.append(cleaned_line)
                    i += 1
                else:
                    break
            if buffer:
                output.append(f"```{lang}")
                output.extend(buffer)
                output.append("```")
            if i < len(lines):
                output.append(lines[i])
                i += 1
        else:
            output.append(line)
            i += 1

    result = "\n".join(output)
    logger.debug(f"Fenced output:\n{result}")
    return result

def multiline_input(prompt: str) -> str:
    print(f"{prompt} (Paste your text. Press Enter twice to finish block):")
    lines = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line == "":
            if lines and lines[-1] == "":
                lines.pop()
                break
        lines.append(line)
    raw_text = "\n".join(lines)
    if raw_text.lower().strip() == "exit":
        return "exit"
    return convert_indented_to_fenced_smart(raw_text)

def manual_entry_mode() -> None:
    logger.info("Starting SICORPS manual data entry.")
    print(f"{BLUE}=== SICORPS MANUAL DATA ENTRY ==={RESET}")

    dataset = load_dataset(OUTPUT_PATH)
    analyze_dataset(dataset)
    print(f"{YELLOW}Paste Markdown with loose indents or language headers (e.g. `python` before a block).{RESET}")
    print(f"{YELLOW}Auto-wraps code in triple-backtick fences. Zero BS.{RESET}")
    print(f"{RED}Type 'exit' to quit.{RESET}\n")

    entry_count = len(dataset)
    while True:
        entry_count += 1
        print(f"{GREEN}--- Entry #{entry_count} ---{RESET}")
        user_input = multiline_input(f"{BLUE}Enter input entry{RESET}")
        if user_input.lower() == "exit":
            break

        chosen_output = multiline_input(f"{BLUE}Enter chosen entry{RESET}")
        if chosen_output.lower() == "exit":
            break

        try:
            system_prompt = assign_system_prompt(user_input, chosen_output)
        except ValueError as e:
            logger.error(f"Failed to assign prompt for new entry: {e}")
            print(f"{RED}Prompt assignment failed: {e}. Skipping entry.{RESET}")
            continue

        entry = {
            "entry_number": entry_count,
            "system": system_prompt,
            "input": user_input,
            "chosen": chosen_output
        }
        dataset.append(entry)
        save_dataset(dataset, OUTPUT_PATH)
        print(f"{GREEN}✔️ Entry #{entry_count} added.{RESET}\n")

    if dataset:
        print(f"{GREEN}Rewriting dataset with updated entries...{RESET}")
        dataset = rewrite_dataset(dataset)
        save_dataset(dataset, OUTPUT_PATH)
        analyze_dataset(dataset)
        print(f"{BLUE}📦 Total entries: {len(dataset)}{RESET}")


def main() -> None:
    """Main function to start manual data entry mode."""
    try:
        manual_entry_mode()
        logger.info("SICORPS manual data entry completed.")
    except KeyboardInterrupt:
        logger.info("User interrupted execution.")
        print(f"{RED}You bailed. Hope you didn’t break anything.{RESET}")
    except Exception as e:
        logger.error(f"Unhandled error: {e}")
        print(f"{RED}Something exploded: {e}. Check dataset.log and get your shit together.{RESET}")

if __name__ == "__main__":
    main()