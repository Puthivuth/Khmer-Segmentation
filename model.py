"""Reusable Khmer longest-match segmentation model."""

import csv
import unicodedata
from pathlib import Path
from typing import Any

# Find the folder containing this Python file.
PROJECT_DIR = Path(__file__).resolve().parent
# Use the project's Khmer dictionary unless another file is provided.
DEFAULT_DICTIONARY_FILE = PROJECT_DIR / "data" / "khmer_dictionary.csv"
# These are the POS tags that the application understands.
POS_TAGS = ("កិ.", "ន.", "គុ.", "ឈ.", "ឧ.", "សព.", "បុ.", "អា.")
# Increase this when the model's rules or data format change.
MODEL_VERSION = 3
# Store a short description of the rules used by the model.
RULES = {
    "matching": "dynamic_programming_longest_dictionary_matches",
    "numbers": "Arabic_and_Khmer_digits",
    "english": "ASCII_letters_and_reference_quotes",
    "normalization": "NFC_and_េី_to_ើ",
    "punctuation": "skip_unicode_punctuation_and_whitespace",
    "unknown_text": "skip_characters_not_in_dictionary",
    "pos": "dictionary_candidates_or_context_uncertain",
}


def build_trie(dictionary: list[dict[str, str]]) -> dict[str, Any]:
    """Build a character trie so dictionary words can be matched quickly."""
    # The root node contains the first character of every dictionary word.
    root: dict[str, Any] = {}
    # Count valid dictionary rows added to the trie.
    word_count = 0

    # Add each dictionary row to the character trie.
    for row in dictionary:
        # Read and trim the word; use an empty string when the column is missing.
        word = (row.get("word") or "").strip()
        # Ignore dictionary rows without a usable word.
        if not word:
            continue
        # Start at the trie root for the current word.
        node = root
        # Create one trie node for each character in the word.
        for character in word:
            node = node.setdefault(character, {})
        # Mark this node as the end of a word and store its dictionary data.
        node.setdefault("entries", []).append({
            "word": word,
            "pos": (row.get("pos") or "").strip(),
        })
        # Count this successfully added dictionary word.
        word_count += 1

    # Return both the trie and metadata needed by the rest of the application.
    return {
        "model_version": MODEL_VERSION,
        "rules": RULES,
        "trie": root,
        "word_count": word_count,
    }


def load_model(
    dictionary_file: Path = DEFAULT_DICTIONARY_FILE,
) -> dict[str, Any]:
    """Load the CSV dictionary and return the segmentation model."""
    # Open the dictionary with UTF-8 support, including an optional BOM.
    with dictionary_file.open(encoding="utf-8-sig", newline="") as file:
        # Convert each CSV row into a dictionary.
        dictionary = list(csv.DictReader(file))
    # Build the searchable trie from the loaded rows.
    return build_trie(dictionary)


def is_separator(character: str) -> bool:
    """Return True for whitespace and Unicode punctuation."""
    # Skip spaces, line breaks, commas, periods, and other punctuation.
    return character.isspace() or unicodedata.category(character).startswith("P")


def is_number(character: str) -> bool:
    """Recognize Arabic and Khmer digits."""
    # isdigit handles Arabic digits; the string explicitly includes Khmer digits.
    return character.isdigit() or character in "០១២៣៤៥៦៧៨៩"


def is_english(character: str) -> bool:
    """Recognize ASCII letters and quote/bracket characters."""
    # Keep English text and common quote/bracket characters together.
    return (character.isascii() and character.isalpha()) or character in "()“ ” ‘ ’ ' \""


def normalize_khmer(text: str) -> str:
    """Normalize equivalent Khmer Unicode spellings before matching."""
    # NFC combines equivalent Unicode sequences into a consistent form.
    text = unicodedata.normalize("NFC", text)
    # Convert this common Khmer spelling to the dictionary's normalized form.
    return text.replace("េី", "ើ")


def longest_match(
    text: str,
    start: int,
    trie: dict[str, Any],
) -> tuple[dict[str, str], int] | None:
    """Return the longest dictionary word beginning at ``start``."""
    # Walk through the trie while characters continue to match a dictionary word.
    node = trie
    index = start
    longest: tuple[dict[str, str], int] | None = None

    while index < len(text) and text[index] in node:
        node = node[text[index]]
        index += 1
        if "entries" in node:
            # Keep the latest complete word because it is the longest so far.
            positions = [entry["pos"] for entry in node["entries"] if entry["pos"]]
            longest = (
                {
                    "word": node["entries"][0]["word"],
                    "pos": ", ".join(dict.fromkeys(positions)),
                },
                index,
            )

    return longest


def dictionary_matches(
    text: str,
    start: int,
    trie: dict[str, Any],
) -> list[tuple[dict[str, str], int]]:
    """Return every dictionary word beginning at ``start``."""
    # Begin searching from the trie root at the requested text position.
    node = trie
    # Store each matching word and the index immediately after that word.
    matches: list[tuple[dict[str, str], int]] = []
    # The current character position while walking through the trie.
    index = start

    # Continue while the next text character exists in the current trie node.
    while index < len(text) and text[index] in node:
        # Move to the node for the current character.
        node = node[text[index]]
        # Advance to the next character.
        index += 1
        # A node with entries represents a complete dictionary word.
        if "entries" in node:
            # Keep only non-empty POS values and remove duplicate tags.
            positions = [entry["pos"] for entry in node["entries"] if entry["pos"]]
            # Save this match; longer matches will be found by continuing the loop.
            matches.append((
                {
                    "word": node["entries"][0]["word"],
                    "pos": ", ".join(dict.fromkeys(positions)),
                },
                index,
            ))

    # Return all dictionary words that start at the requested position.
    return matches


def segment_khmer_run(text: str, trie: dict[str, Any]) -> list[dict[str, str]]:
    """Find the best dictionary-based segmentation for one Khmer text run."""
    # best[position] stores the highest-scoring segmentation from that position.
    best: dict[int, tuple[float, list[dict[str, str]]]] = {len(text): (0, [])}

    # Process the text from right to left so later results are already available.
    for index in range(len(text) - 1, -1, -1):
        # Store possible segmentations beginning at this character.
        options = []
        # Check the longest dictionary word first.
        longest = longest_match(text, index, trie)
        matches = dictionary_matches(text, index, trie)
        if longest:
            matches = [longest] + [match for match in matches if match[1] != longest[1]]

        # Try every dictionary word that starts at this position.
        for entry, end in matches:
            # Calculate the matched word length.
            length = end - index
            # Squaring favors longer dictionary words over short fragments.
            score = length * length
            # Only use a match if the remaining text can also be segmented.
            if end in best:
                # Combine this word's score with the best score for the remainder.
                options.append((score + best[end][0], [{
                    "word": entry["word"],
                    "dictionary_pos": entry["pos"],
                    "pos": entry["pos"] or "POS_NOT_AVAILABLE",
                    "status": "FOUND",
                }] + best[end][1]))

        # If no dictionary word matches, skip this character.
        if not matches and index + 1 in best:
            options.append(best[index + 1])

        # Save the highest-scoring option for this starting position.
        if options:
            best[index] = max(options, key=lambda option: option[0])

    # Return the best segmentation for the complete text run.
    return best.get(0, (0, []))[1]


def pos_candidates(raw_pos: str) -> list[str]:
    """Extract supported POS tags from the dictionary POS text."""
    # Keep only POS tags recognized by this project.
    return [tag for tag in POS_TAGS if tag in raw_pos]


def apply_context_pos(tokens: list[dict[str, str]]) -> list[dict[str, str]]:
    """Convert dictionary POS candidates into the user-facing POS label."""
    # Resolve the final POS label for every token found in the dictionary.
    for token in tokens:
        # Unknown and non-Khmer tokens already have their own POS labels.
        if token["status"] != "FOUND":
            continue
        # Extract supported POS candidates from the raw dictionary value.
        candidates = pos_candidates(token["dictionary_pos"])
        # No recognized tag means the dictionary has no usable POS information.
        if not candidates:
            token["pos"] = "POS_NOT_AVAILABLE"
        # A single candidate can be displayed directly.
        elif len(candidates) == 1:
            token["pos"] = candidates[0]
        # Multiple candidates require context that this model does not infer.
        else:
            token["pos"] = "POS_CONTEXT_UNCERTAIN"
    # Return the same list after updating its POS values.
    return tokens


def segment(text: str, model: dict[str, Any] | None = None) -> list[dict[str, str]]:
    """Segment Khmer text while preserving numbers and non-Khmer text."""
    # Load the model automatically when the caller does not provide one.
    model = model or load_model()
    # Normalize the input before looking up dictionary words.
    text = normalize_khmer(text)
    # Get the trie used to find dictionary matches.
    trie = model["trie"]
    # Accumulate the final token dictionaries here.
    tokens = []
    # Track the current character position in the input.
    index = 0

    # Scan the input from left to right.
    while index < len(text):
        # Read the character at the current position.
        character = text[index]
        # Ignore whitespace and punctuation between tokens.
        if is_separator(character):
            index += 1
            continue

        # Numbers and English text are kept as one non-Khmer token.
        if is_number(character) or is_english(character):
            # Remember where this non-Khmer token starts.
            start = index
            # Continue until the non-Khmer token ends.
            while index < len(text) and (
                is_number(text[index]) or is_english(text[index]) or text[index] == " "
            ):
                index += 1
            # Remove spaces at the edges of the collected token.
            token = text[start:index].strip()
            # Add the token only when it contains actual content.
            if token:
                tokens.append({
                    "word": token,
                    "dictionary_pos": "",
                    "pos": "N/A",
                    "status": "NON_KHMER_OR_NUMBER",
                })
            continue

        # Collect one uninterrupted Khmer run for dictionary segmentation.
        start = index
        while (
            index < len(text)
            and not is_separator(text[index])
            and not is_number(text[index])
            and not is_english(text[index])
        ):
            # Advance until a separator, number, or English character appears.
            index += 1

        # Avoid an infinite loop if the current character was not classified.
        if start == index:
            index += 1
            continue
        # Segment the Khmer run and add its tokens to the result.
        tokens.extend(segment_khmer_run(text[start:index], trie))

    # Apply final POS labels after all tokens have been collected.
    return apply_context_pos(tokens)
