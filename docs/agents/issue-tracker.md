# Issue tracker: GitHub

Issues and specifications for this repository live in GitHub Issues at `Dharshan2004/techjam-2026-track-4-shopping-copilot`. Because this clone also has TikTok's repository configured as `upstream`, every `gh` issue and label command must specify `--repo Dharshan2004/techjam-2026-track-4-shopping-copilot` or run after setting that repository as the GitHub CLI default.

## Conventions

- Create issues with `gh issue create --repo Dharshan2004/techjam-2026-track-4-shopping-copilot`.
- Read issues and their discussion with `gh issue view <number> --comments --repo Dharshan2004/techjam-2026-track-4-shopping-copilot`.
- List work with `gh issue list --repo Dharshan2004/techjam-2026-track-4-shopping-copilot`, including labels and comments where relevant.
- Comment with `gh issue comment <number> --repo Dharshan2004/techjam-2026-track-4-shopping-copilot`.
- Apply or remove labels with `gh issue edit <number> --repo Dharshan2004/techjam-2026-track-4-shopping-copilot`.
- Close completed or rejected work with `gh issue close <number> --repo Dharshan2004/techjam-2026-track-4-shopping-copilot` and an explanatory comment.

## Pull requests as a triage surface

**PRs as a request surface: no.**

Pull requests are not treated as incoming feature requests by the triage workflow. This flag can be changed to `yes` later if that convention changes.

## Skill operations

- When a skill says to publish to the issue tracker, create a GitHub issue.
- When a skill says to fetch a relevant ticket, read the issue body, labels, and comments.
- When a skill creates agent-ready work, apply the label mapped to `ready-for-agent` in `docs/agents/triage-labels.md`.

## Wayfinding operations

When a wayfinding workflow is used:

- Represent the map as one GitHub issue labelled `wayfinder:map`.
- Represent decision work as child issues or, when sub-issues are unavailable, as a task list linked from the map.
- Use `wayfinder:research`, `wayfinder:prototype`, `wayfinder:grilling`, or `wayfinder:task` to identify child types.
- Prefer native GitHub issue dependencies for blocking relationships. Fall back to a `Blocked by: #<number>` line when dependencies are unavailable.
- Claim work by assigning the issue to the current GitHub user.
- Resolve work by recording the decision in a comment, closing the child issue, and linking the outcome from the map.
