"""Generate original synthetic reviews; never reads or copies Amazon review text."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import random

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    rng = random.Random(42)
    titles = ["Harbour Lights", "Winter Orchestra", "The Quiet Planet", "Last Train Home",
              "Museum at Midnight", "A Paper Kingdom", "Northern Summer", "The Second Map"]
    observations = [
        "The dialogue is clear and the music stays below the voices in busy scenes.",
        "The picture is sharp in daylight scenes but shadows lose a little detail.",
        "The subtitles follow the dialogue well and the menu is simple to navigate.",
        "The middle section moves slowly before the final scenes bring the story together.",
        "The cast makes ordinary conversations feel natural and the setting is convincing.",
        "The bonus interview discusses the editing choices and lasts about twenty minutes.",
        "The disc played without interruptions and the packaging arrived in good condition.",
        "The characters have interesting motives although the ending feels rather sudden.",
        "The soundtrack has a wide range and quiet passages benefit from a higher volume.",
        "The restored image keeps visible grain instead of smoothing away every texture.",
        "The episode selection works well and each chapter starts at a sensible point.",
        "The camera often stays still which gives the actors room to carry each scene.",
    ]
    judgements = {
        1: "I found the story frustrating and would struggle to recommend this release.",
        2: "There are a few good moments but the overall experience disappointed me.",
        3: "This is a reasonable evening of viewing with strengths and clear weaknesses.",
        4: "I enjoyed this release and would watch the main feature again with friends.",
        5: "This was a wonderful viewing experience and an easy addition to my collection.",
    }
    rows = []
    for item, name in enumerate(titles):
        for _ in range(128):
            detailed = rng.random() < 0.35
            stars = rng.choices([1, 2, 3, 4, 5], weights=[1, 1, 2, 3 + item % 3, 3])[0]
            sentences = rng.sample(observations, k=rng.randint(3, 5) if detailed else rng.randint(1, 2))
            text = " ".join(sentences + [judgements[stars]])
            votes = rng.choices([0, 1, 2, 3, 5, 8, 13, 21, 34],
                                weights=[15, 10, 15, 15, 15, 12, 8, 6, 4] if detailed
                                else [72, 10, 7, 4, 3, 2, 1, 0.7, 0.3])[0]
            rows.append({"title": f"Viewing notes on {name}", "text": text, "rating": stars,
                         "helpful_vote": votes, "verified_purchase": rng.random() < 0.8,
                         "timestamp": 1577836800000 + rng.randrange(1200) * 86400000,
                         "parent_asin": f"DEMO_MOVIE_{item + 1:02d}"})
    rng.shuffle(rows)
    folder = ROOT / "datasets" / "sample"
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / "reviews.jsonl"
    target.write_text("".join(json.dumps(r, ensure_ascii=True) + "\n" for r in rows), encoding="utf-8", newline="\n")
    manifest = {"kind": "original synthetic movie reviews; NOT Amazon review excerpts",
                "generator": "scripts/build_demo_sample.py", "seed": 42, "rows": len(rows),
                "items": len(titles), "vote_counts": dict(sorted(Counter(r['helpful_vote'] for r in rows).items())),
                "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                "schema_source": "https://amazon-reviews-2023.github.io/main.html",
                "text_provenance": "Original sentence templates in the generator; no real users or products."}
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(rows)} synthetic reviews ({target.stat().st_size:,} bytes).")


if __name__ == "__main__":
    main()
