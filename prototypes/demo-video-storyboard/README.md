# Demo video storyboard prototype

Throwaway UI prototype answering: **What should the three-minute Track 4 Beeline demo look like?**

Its three session excerpts were recorded by running the real `tools.demo_session` helper in forced-offline mode. The prototype page itself does not import, execute, or contact the Agent, evaluator, models, or APIs.

Run it with one command from the repository root:

```bash
python3 -m http.server 4173 -d prototypes/demo-video-storyboard
```

Then open [http://localhost:4173/?variant=A](http://localhost:4173/?variant=A).

Variants:

- `?variant=A` — cinematic engineering journey
- `?variant=B` — director's evidence cut with the complete voiceover
- `?variant=C` — metric/latency Pareto proof reel

Use the bottom arrows or keyboard left/right arrows to switch variants. Use the player controls to scrub the 3:00 timeline; the playback-speed button cycles through 1×, 2×, and 4×.

The exact-timestamp delivery script is in [`../../artifacts/team-techbros-storyboard/DEMO_SCRIPT.md`](../../artifacts/team-techbros-storyboard/DEMO_SCRIPT.md).

Delete this directory after choosing a treatment and transferring the useful decisions into the real video plan.
