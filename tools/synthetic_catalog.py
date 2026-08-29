"""Generate a catalog-shaped stand-in for scale measurements.

The frozen 50,000-product `data/catalog.jsonl` is distributed as a GitHub Release
asset. This generator exists only so build time, artifact size, load time, and
peak memory can be measured at the right scale before that file is available. It
is a development aid, never an input to a scored run: a real build must use the
frozen catalog, and the artifact's catalog checksum is what keeps the two apart.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


GARMENTS = (
    "running shoe", "ankle boot", "winter scarf", "denim jacket", "cotton t-shirt",
    "wool sweater", "leather belt", "athletic sock", "rain parka", "linen shirt",
    "yoga legging", "chino trouser", "puffer vest", "canvas sneaker", "silk blouse",
    "fleece hoodie", "sun hat", "swim short", "cardigan", "work glove",
)
MATERIALS = ("cotton", "polyester", "nylon", "leather", "wool", "spandex", "silk", "rayon")
COLORS = ("black", "white", "blue", "red", "pink", "green", "brown", "gray", "purple", "yellow")
DEPARTMENTS = ("womens", "mens", "unisex", "girls", "boys")
FITS = ("relaxed", "slim", "regular", "oversized", "athletic")
USES = ("hiking", "running", "office wear", "everyday wear", "travel", "cold weather", "the gym")
STORES = ("Northline", "Harborfield", "Cedar & Pine", "Atlas Basics", "Marlow Supply")


def _row(rng: random.Random, index: int) -> dict:
    garment = rng.choice(GARMENTS)
    material = rng.choice(MATERIALS)
    color = rng.choice(COLORS)
    department = rng.choice(DEPARTMENTS)
    fit = rng.choice(FITS)
    use = rng.choice(USES)
    return {
        "parent_asin": f"S{index:09d}",
        "title": f"{department.capitalize()} {color} {material} {garment}",
        "categories": ["Clothing", "Shoes & Jewelry", garment.split()[-1].capitalize()],
        "features": [
            f"{material} construction",
            f"{fit} fit",
            f"suitable for {use}",
            f"machine washable in {color}",
        ],
        "details": {
            "department": department,
            "material": material,
            "color": color,
            "fit_type": fit,
        },
        "description": [
            f"A {fit} {color} {garment} made from {material}, designed for {use}.",
            f"Sized for {department} wear with reinforced stitching and a durable finish.",
        ],
        "store": rng.choice(STORES),
        "price": round(rng.uniform(9.99, 199.99), 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="data/synthetic_catalog.jsonl")
    arguments = parser.parse_args()

    rng = random.Random(arguments.seed)
    destination = Path(arguments.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for index in range(arguments.rows):
            handle.write(json.dumps(_row(rng, index)) + "\n")
    print(json.dumps({"rows": arguments.rows, "output": str(destination)}, indent=2))


if __name__ == "__main__":
    main()
