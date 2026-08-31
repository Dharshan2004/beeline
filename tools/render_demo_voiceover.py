"""Generate local per-scene narration for the Team TechBros demo video."""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


NARRATION = [
    "We are Team TechBros. We built a conversational shopping copilot that is fast, reproducible, and valid without the network.",
    "We began with an official baseline that treated every customer message as a new search. Earlier needs disappeared, final ranking was weak, and the target usually arrived too late to support a real purchase decision.",
    "Our key architectural decision was to separate language interpretation from state correctness. A model, or the local interpreter, proposes a complete Turn Plan. Deterministic code validates every transition against one revision. Either the plan commits atomically, or none of it does.",
    "Retrieval uses three complementary routes. Structured evidence enforces active constraints. BM25 over FTS5 preserves exact catalog language and prior dialog terms. Embedded Qdrant supplies semantic recall. We normalize and fuse those signals with frozen weights, enforce hard eligibility, narrow to fifty candidates, and use a bounded local cross-encoder to produce a catalog-valid top ten.",
    "The largest useful gain came from better shopping behavior, not simply a larger model. We accumulated conversational evidence, selected questions by information value, and understood stated budgets. Together those changes moved the development score from zero point five five one eight to zero point seven three nine two at zero point seven zero seven seconds p ninety five.",
    "For TikTok Shop, retrieval quality is only half the product problem. The credible recommendation must arrive while purchase intent is still alive. Asking another question can improve precision, but it also delays the buy moment. Deeper or connected ranking can improve order, but added latency creates friction. So we balance coverage, first-rank quality, and turns to conversion.",
    "Here is a released sample run through the repository's real demo helper. The target is absent for three turns. The agent asks about color, use case, and then feature, each selected from the live candidate pool. On turn four, the customer reveals the distinguishing Triple Moon Pentagram feature. Only then does the target enter at rank one. The system converts with no connected-model tokens.",
    "One successful session makes the behavior visible. Robustness makes the engineering credible. The local system scores zero point seven five five six on all two hundred released sessions, zero point seven two five zero under paraphrase, and zero point seven one eight two on novel targets. All two hundred eighteen tests pass, and optional semantic ranking remains additive and fail-open.",
    "Team TechBros. High quality, seconds per turn, and valid without the network.",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="artifacts/team-techbros-demo/voiceover")
    parser.add_argument("--voice", default="Samantha")
    parser.add_argument("--rate", default="170")
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for index, narration in enumerate(NARRATION, start=1):
        aiff = output / f"scene-{index:02d}.aiff"
        subprocess.run(
            ["/usr/bin/say", "-v", args.voice, "-r", args.rate, "-o", str(aiff), narration],
            check=True,
        )
        print(aiff)


if __name__ == "__main__":
    main()
