"""Render audience-facing stills for the Team TechBros Track 4 demo video.

The session frame is a faithful visual rendering of the output produced by:
    .venv/bin/python -m tools.demo_session --sample public_0001

No evaluator or agent is invoked by this renderer.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


WIDTH, HEIGHT = 1920, 1080
BG = "#07100d"
PANEL = "#0e1915"
PANEL_2 = "#13211c"
WHITE = "#eef6f0"
MUTED = "#8d9b93"
LINE = "#26372f"
LIME = "#d8ff63"
MINT = "#5cf2bb"
ORANGE = "#ff936a"
REPO = "github.com/Dharshan2004/techjam-2026-track-4-shopping-copilot"

FONT_REGULAR = "/System/Library/Fonts/Helvetica.ttc"
FONT_BOLD = "/System/Library/Fonts/Helvetica.ttc"
FONT_MONO = "/System/Library/Fonts/SFNSMono.ttf"


def font(size: int, *, mono: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONT_MONO if mono else FONT_REGULAR, size)


def canvas(section: str, index: str) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(image)
    draw.text((78, 52), "TEAM TECHBROS", font=font(22, mono=True), fill=LIME)
    draw.text((78, 88), section.upper(), font=font(15, mono=True), fill=MUTED)
    draw.text((1738, 52), index, font=font(17, mono=True), fill=MUTED)
    draw.line((78, 126, 1842, 126), fill=LINE, width=2)
    draw.text((78, 1022), REPO, font=font(14, mono=True), fill="#67766e")
    return image, draw


def rounded(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], *, fill=PANEL, outline=LINE, radius=26, width=2) -> None:
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def center_text(draw: ImageDraw.ImageDraw, y: int, text: str, size: int, fill=WHITE, *, mono=False) -> None:
    f = font(size, mono=mono)
    box = draw.textbbox((0, 0), text, font=f)
    draw.text(((WIDTH - (box[2] - box[0])) / 2, y), text, font=f, fill=fill)


def metric(draw: ImageDraw.ImageDraw, x: int, y: int, label: str, value: str, detail: str, accent=LIME) -> None:
    rounded(draw, (x, y, x + 390, y + 190), fill=PANEL)
    draw.text((x + 28, y + 26), label.upper(), font=font(15, mono=True), fill=MUTED)
    draw.text((x + 28, y + 64), value, font=font(58), fill=accent)
    draw.text((x + 28, y + 143), detail, font=font(15, mono=True), fill="#73827a")


def slide_title() -> Image.Image:
    image, draw = canvas("Track 4 · Conversational Shopping", "01 / 09")
    draw.text((116, 258), "SHOPPING", font=font(112), fill=WHITE)
    draw.text((116, 364), "COPILOT", font=font(112), fill=LIME)
    draw.text((122, 510), "A fast, reproducible agent that remembers intent,", font=font(31), fill=MUTED)
    draw.text((122, 553), "retrieves locally, and fails open.", font=font(31), fill=MUTED)
    rounded(draw, (122, 674, 680, 770), fill=PANEL_2, outline="#385143")
    draw.text((154, 700), "50,000 PRODUCTS", font=font(20, mono=True), fill=MINT)
    draw.text((154, 735), "10 TURNS MAXIMUM", font=font(17, mono=True), fill=WHITE)
    draw.text((1320, 282), "0.7556", font=font(104), fill=LIME)
    draw.text((1330, 394), "LOCAL · FULL 200", font=font(17, mono=True), fill=MUTED)
    draw.line((1330, 445, 1750, 445), fill=LINE, width=3)
    draw.text((1330, 482), "218 / 218 tests", font=font(24), fill=WHITE)
    draw.text((1330, 530), "valid without network", font=font(18), fill=MINT)
    return image


def slide_problem() -> Image.Image:
    image, draw = canvas("The starting point", "02 / 09")
    draw.text((92, 188), "A search engine is not yet", font=font(54), fill=WHITE)
    draw.text((92, 248), "a shopping conversation.", font=font(54), fill=LIME)
    metric(draw, 92, 382, "Official baseline", "0.1067", "TechnicalScore", ORANGE)
    metric(draw, 510, 382, "Hit@10", "0.125", "one target in eight", ORANGE)
    metric(draw, 928, 382, "MRR", "0.068", "weak final ordering", ORANGE)
    metric(draw, 1346, 382, "MTTC", "9.81", "nearly all ten turns", ORANGE)
    rounded(draw, (92, 650, 1760, 914), fill="#0b1512")
    messages = [
        ("TURN 1", "“for hiking”", "forgotten"),
        ("TURN 2", "“waterproof”", "forgotten"),
        ("TURN 3", "“under $100”", "only this message reaches retrieval"),
    ]
    for row, (turn, message, status) in enumerate(messages):
        y = 688 + row * 70
        draw.text((128, y), turn, font=font(15, mono=True), fill=MUTED)
        draw.text((300, y - 5), message, font=font(28), fill=WHITE)
        draw.line((760, y + 16, 1120, y + 16), fill=LINE, width=2)
        draw.text((1160, y), status, font=font(16, mono=True), fill=MINT if row == 2 else ORANGE)
    return image


def slide_architecture() -> Image.Image:
    image, draw = canvas("Validated control plane", "03 / 09")
    draw.text((90, 180), "Meaning is proposed.", font=font(55), fill=WHITE)
    draw.text((90, 242), "State is validated.", font=font(55), fill=LIME)
    nodes = [
        (90, "01", "TURN PLAN", "Model or local\ninterpreter"),
        (500, "02", "ATOMIC VALIDATOR", "All transitions\nor none"),
        (910, "03", "CONSTRAINT STATE", "Revisioned and\ndeterministic"),
        (1320, "04", "RETRIEVE + RANK", "Catalog-valid\ntop ten"),
    ]
    for x, number, title, body in nodes:
        rounded(draw, (x, 430, x + 330, 700), fill=PANEL)
        draw.text((x + 28, 458), number, font=font(17, mono=True), fill=LIME)
        draw.text((x + 28, 540), title, font=font(19, mono=True), fill=WHITE)
        draw.multiline_text((x + 28, 590), body, font=font(23), fill=MUTED, spacing=8)
    for x in (436, 846, 1256):
        draw.line((x, 566, x + 48, 566), fill=MINT, width=3)
        draw.polygon([(x + 48, 558), (x + 64, 566), (x + 48, 574)], fill=MINT)
    rounded(draw, (330, 790, 1590, 896), fill="#0b1713", outline="#2a4b3d")
    center_text(draw, 812, "LOCAL AND CONNECTED PATHS SHARE THE SAME INVARIANTS", 18, MINT, mono=True)
    return image


def slide_retrieval() -> Image.Image:
    image, draw = canvas("Retrieval architecture", "04 / 09")
    draw.text((90, 178), "Three routes. One fused pool.", font=font(54), fill=WHITE)
    draw.text((90, 240), "One bounded final ranker.", font=font(54), fill=LIME)
    routes = [
        (90, "STRUCTURED", "0.02", "active constraints\nand eligibility"),
        (490, "BM25 / FTS5", "0.64", "current message +\nprior dialog terms"),
        (890, "DENSE / QDRANT", "0.34", "semantic query +\nactive hard evidence"),
    ]
    for x, name, weight, description in routes:
        rounded(draw, (x, 386, x + 340, 638), fill=PANEL)
        draw.text((x + 26, 418), name, font=font(17, mono=True), fill=MINT)
        draw.text((x + 26, 462), weight, font=font(50), fill=WHITE)
        draw.text((x + 26, 536), "FUSION WEIGHT", font=font(12, mono=True), fill=MUTED)
        draw.multiline_text((x + 170, 462), description, font=font(18), fill=MUTED, spacing=7)
    draw.line((430, 512, 470, 512), fill=LINE, width=3)
    draw.line((830, 512, 870, 512), fill=LINE, width=3)
    rounded(draw, (1320, 386, 1772, 638), fill="#122119", outline="#4a6241")
    draw.text((1350, 418), "NORMALIZE + FUSE", font=font(17, mono=True), fill=LIME)
    draw.text((1350, 474), "100", font=font(56), fill=WHITE)
    draw.text((1480, 492), "wide candidates", font=font(17), fill=MUTED)
    draw.text((1350, 568), "hard eligibility · popularity bands", font=font(15, mono=True), fill=MINT)
    steps = [
        (130, "50", "frozen candidate depth"),
        (580, "LOCAL CROSS-ENCODER", "MiniLM-L6 · 1.5 s deadline"),
        (1150, "TOP 10", "budget-aware · catalog-valid"),
    ]
    for x, headline, caption in steps:
        draw.text((x, 760), headline, font=font(31 if len(headline) > 4 else 54), fill=WHITE)
        draw.text((x, 822), caption, font=font(15, mono=True), fill=MUTED)
    draw.line((360, 797, 530, 797), fill=LIME, width=3)
    draw.polygon([(530, 789), (546, 797), (530, 805)], fill=LIME)
    draw.line((1010, 797, 1100, 797), fill=LIME, width=3)
    draw.polygon([(1100, 789), (1116, 797), (1100, 805)], fill=LIME)
    return image


def slide_improvements() -> Image.Image:
    image, draw = canvas("Measured iteration", "05 / 09")
    draw.text((90, 178), "The biggest gain came from", font=font(53), fill=WHITE)
    draw.text((90, 238), "better shopping behavior.", font=font(53), fill=LIME)
    rounded(draw, (90, 360, 672, 860), fill=PANEL)
    draw.text((126, 398), "160-SESSION DEVELOPMENT SPLIT", font=font(14, mono=True), fill=MUTED)
    draw.text((126, 492), "0.5518", font=font(58), fill="#78867e")
    draw.text((365, 510), "→", font=font(36), fill=MINT)
    draw.text((440, 480), "0.7392", font=font(72), fill=LIME)
    draw.text((126, 605), "+0.1873 TechnicalScore", font=font(22, mono=True), fill=MINT)
    draw.line((126, 680, 630, 680), fill=LINE, width=2)
    draw.text((126, 720), "p95 complete turn", font=font(15, mono=True), fill=MUTED)
    draw.text((126, 760), "0.707 s", font=font(36), fill=WHITE)
    improvements = [
        (760, "01", "DIALOG MEMORY", "Useful evidence accumulates across turns."),
        (760, "02", "INFORMATION-VALUE QUESTIONS", "Ask what best divides the live candidate pool."),
        (760, "03", "BUDGET UNDERSTANDING", "Prefer in-range products without dropping unknown prices."),
    ]
    for idx, (x, number, heading, body) in enumerate(improvements):
        y = 360 + idx * 170
        rounded(draw, (x, y, 1768, y + 142), fill=PANEL_2)
        draw.text((x + 28, y + 31), number, font=font(17, mono=True), fill=LIME)
        draw.text((x + 100, y + 25), heading, font=font(19, mono=True), fill=WHITE)
        draw.text((x + 100, y + 68), body, font=font(20), fill=MUTED)
    return image


def slide_conversion_tradeoff() -> Image.Image:
    image, draw = canvas("TikTok Shop conversion lens", "06 / 09")
    draw.text((90, 178), "The best answer is the one that arrives", font=font(51), fill=WHITE)
    draw.text((90, 237), "while purchase intent is still alive.", font=font(51), fill=LIME)
    tradeoffs = [
        (90, "HIT@10", "Show a credible option", "Coverage earns the chance to convert."),
        (650, "MRR", "Put the best option first", "Higher rank reduces comparison friction."),
        (1210, "MTTC", "Reach it in fewer turns", "Every unnecessary question delays the buy moment."),
    ]
    for x, metric_name, heading, body in tradeoffs:
        rounded(draw, (x, 380, x + 500, 620), fill=PANEL)
        draw.text((x + 28, 412), metric_name, font=font(17, mono=True), fill=MINT)
        draw.text((x + 28, 463), heading, font=font(27), fill=WHITE)
        draw.text((x + 28, 525), body, font=font(18), fill=MUTED)
    rounded(draw, (90, 694, 1710, 890), fill=PANEL_2, outline="#405746")
    draw.text((126, 730), "THE OPERATING TRADEOFF", font=font(15, mono=True), fill=LIME)
    draw.text((126, 775), "Ask only when the answer can materially shrink the candidate pool.", font=font(26), fill=WHITE)
    draw.text((126, 821), "Use bounded local ranking by default; spend connected latency only on uncertain heads.", font=font(22), fill=MUTED)
    draw.text((1440, 740), "0.72 s", font=font(42), fill=MINT)
    draw.text((1440, 798), "LOCAL p95", font=font(13, mono=True), fill=MUTED)
    draw.text((1580, 740), "4.59 s", font=font(42), fill=LIME)
    draw.text((1580, 798), "MINI-RANKED p95", font=font(13, mono=True), fill=MUTED)
    return image


def terminal_session() -> Image.Image:
    image, draw = canvas("Live agent session · public_0001", "07 / 09")
    draw.text((90, 166), "Four turns. The target appears only after distinguishing evidence.", font=font(36), fill=WHITE)
    rounded(draw, (90, 246, 1790, 948), fill="#09110f", outline="#31443b", radius=18)
    draw.ellipse((120, 274, 138, 292), fill="#ff6b61")
    draw.ellipse((150, 274, 168, 292), fill="#f7c653")
    draw.ellipse((180, 274, 198, 292), fill="#5ddf82")
    draw.text((810, 269), "tools.demo_session · public_0001", font=font(14, mono=True), fill="#6e7f76")
    draw.line((90, 318, 1790, 318), fill="#273831", width=2)
    draw.text((132, 350), "Session public_0001 · buying", font=font(23, mono=True), fill=WHITE)
    draw.text((132, 389), "Hidden target: B09PYB7B6Z · Triple Moon Pentagram necklace", font=font(20, mono=True), fill="#718078")
    turns = [
        ("TURN 1", "Jewelry necklaces · alloy", "target absent", "asks color · eliminates ≥ 44"),
        ("TURN 2", "no additional color preference", "target absent", "asks use case · eliminates ≥ 48"),
        ("TURN 3", "no additional use-case preference", "target absent", "asks feature · eliminates ≥ 43"),
        ("TURN 4", "Triple Moon Pentagram symbol", "target rank 1", "converted"),
    ]
    for row, (turn, customer, result, decision) in enumerate(turns):
        y = 458 + row * 100
        draw.text((132, y), turn, font=font(18, mono=True), fill=LIME if row == 3 else WHITE)
        draw.text((300, y), customer, font=font(20, mono=True), fill="#71dff2")
        draw.text((1020, y), result, font=font(19, mono=True), fill=MINT if row == 3 else "#718078")
        draw.text((300, y + 39), decision, font=font(17, mono=True), fill="#f4cf62" if row < 3 else LIME)
        if row < 3:
            draw.line((132, y + 78, 1645, y + 78), fill="#1f3029", width=1)
    draw.text((1280, 851), "TARGET FOUND", font=font(16, mono=True), fill=MINT)
    draw.text((1280, 887), "TURN 4 · RANK 1", font=font(30, mono=True), fill=LIME)
    return image


def slide_results() -> Image.Image:
    image, draw = canvas("Robustness evidence", "08 / 09")
    draw.text((90, 178), "One session is the demo.", font=font(53), fill=WHITE)
    draw.text((90, 238), "Robustness is the evidence.", font=font(53), fill=LIME)
    metric(draw, 90, 390, "Exact · full 200", "0.7556", "released sessions")
    metric(draw, 510, 390, "Paraphrased", "0.7250", "same intent · new wording", MINT)
    metric(draw, 930, 390, "Novel targets", "0.7182", "absent from public labels", MINT)
    metric(draw, 1350, 390, "Test suite", "218/218", "passing", WHITE)
    rounded(draw, (90, 696, 1770, 872), fill=PANEL_2)
    draw.text((128, 735), "LOCAL-ONLY", font=font(17, mono=True), fill=MINT)
    draw.text((128, 780), "Correct output without network, hosted APIs, or runtime downloads.", font=font(29), fill=WHITE)
    draw.text((128, 831), "Optional semantic ranking is additive and fail-open.", font=font(20), fill=MUTED)
    return image


def slide_close() -> Image.Image:
    image, draw = canvas("Team TechBros", "09 / 09")
    center_text(draw, 230, "FAST. REPRODUCIBLE.", 70, WHITE)
    center_text(draw, 315, "VALID WITHOUT THE NETWORK.", 70, LIME)
    center_text(draw, 500, "0.7556", 126, LIME)
    center_text(draw, 646, "LOCAL · FULL 200", 20, MUTED, mono=True)
    rounded(draw, (420, 760, 1500, 850), fill=PANEL_2, outline="#385143")
    center_text(draw, 790, REPO, 18, MINT, mono=True)
    return image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="artifacts/team-techbros-demo")
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    slides = [
        slide_title(),
        slide_problem(),
        slide_architecture(),
        slide_retrieval(),
        slide_improvements(),
        slide_conversion_tradeoff(),
        terminal_session(),
        slide_results(),
        slide_close(),
    ]
    for index, slide in enumerate(slides, start=1):
        slide.save(output / f"slide-{index:02d}.png", optimize=True)
    slides[6].save(output / "demo-session-public_0001.png", optimize=True)
    print(output.resolve())


if __name__ == "__main__":
    main()
