"""Mask likely identifiers before logs and preserve one finite label vocabulary."""

import re
import unicodedata

EMAIL = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
NUMBER = re.compile(r"(?<!\w)(?:\d[ -]?){8,19}(?!\w)")


def mask(text: str) -> str:
    value = EMAIL.sub("[EMAIL]", text)
    return NUMBER.sub("[NUMBER]", value)


def normalize(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def valid_label(label: str, labels: list[str]) -> str:
    if label not in labels:
        raise ValueError("Model returned a label outside the fixed vocabulary")
    return label
