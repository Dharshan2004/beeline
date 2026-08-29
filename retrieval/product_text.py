from __future__ import annotations

import re
from typing import Iterable


TEXT_TEMPLATE_VERSION = 1

# Ordered deliberately: the fields a customer is most likely to describe come
# first, so truncation at the model's sequence limit drops the least useful
# evidence rather than an arbitrary tail.
TEXT_FIELDS: tuple[str, ...] = (
    "title",
    "store",
    "categories",
    "features",
    "details",
    "description",
)

_WHITESPACE_RE = re.compile(r"\s+")


def _flatten(value: object) -> Iterable[str]:
    if value is None:
        return ()
    if isinstance(value, dict):
        parts: list[str] = []
        for key in sorted(value, key=str):
            item = value[key]
            if item in (None, "", [], {}):
                continue
            parts.extend(f"{key}: {piece}" for piece in _flatten(item))
        return parts
    if isinstance(value, (list, tuple)):
        parts = []
        for item in value:
            parts.extend(_flatten(item))
        return parts
    if isinstance(value, bool):
        return (str(value).lower(),)
    text = str(value).strip()
    return (text,) if text else ()


def product_text(product: dict) -> str:
    """Render one catalog product as the deterministic string that gets embedded.

    Dictionary keys are sorted so that a catalog row producing the same content
    in a different key order still yields the same vector. The result is
    whitespace-normalized because the tokenizer treats runs of whitespace
    inconsistently across platforms.
    """
    sections: list[str] = []
    for field in TEXT_FIELDS:
        values = [piece for piece in _flatten(product.get(field)) if piece]
        if not values:
            continue
        sections.append(f"{field}: " + " | ".join(values))
    joined = "\n".join(sections)
    return _WHITESPACE_RE.sub(" ", joined).strip()
