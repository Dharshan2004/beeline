# Demo video storyboard prototype

Throwaway UI prototype answering: **What should the three-minute Track 4 demo video look like?**

It uses mock conversations and the supplied aggregate metrics. It does not import, execute, or contact the real shopping Agent, evaluator, models, or APIs.

Run it with one command from the repository root:

```bash
python3 -m http.server 4173 -d prototypes/demo-video-storyboard
```

Then open [http://localhost:4173/?variant=A](http://localhost:4173/?variant=A).

Variants:

- `?variant=A` — cinematic engineering journey
- `?variant=B` — director's evidence cut with the complete voiceover
- `?variant=C` — metric/latency Pareto proof reel

Use the bottom arrows or keyboard left/right arrows to switch variants. Use the player controls to scrub the three-minute timeline; the playback-speed button cycles through 1×, 2×, and 4×.

The evidence-backed script and claim boundaries are in [`NARRATIVE.md`](NARRATIVE.md).

Delete this directory after choosing a treatment and transferring the useful decisions into the real video plan.
