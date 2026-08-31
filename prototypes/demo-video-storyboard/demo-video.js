// PROTOTYPE — Three demo-video treatments, switchable with ?variant=A|B|C.
// This uses mock content only. It never imports or invokes the real Agent/evaluator.

const METRICS = {
  baseline: { label: "Official baseline", hit: 0.125, mrr: 0.068, mttc: 9.81, score: 0.1067 },
  local: { label: "Local-only", hit: 0.9, mrr: 0.521779, mttc: 5.216667, score: 0.7222, p95: "0.72 s" },
  ranked: { label: "Mini-ranked", hit: 0.916667, mrr: 0.586435, mttc: 4.866667, score: 0.756931, p95: "4.59 s" },
};

const SCENES = [
  { start: 0, end: 12, key: "hook", eyebrow: "THE CHALLENGE", title: "50,000 products. One hidden purchase.", note: "Open with the evaluator's real constraint: ten turns." },
  { start: 12, end: 30, key: "baseline", eyebrow: "WHERE WE STARTED", title: "The baseline searched each message in isolation.", note: "Show the official 0.1067 baseline and one failure trace." },
  { start: 30, end: 50, key: "insight", eyebrow: "THE ARCHITECTURAL INSIGHT", title: "Interpretation can be probabilistic. State cannot.", note: "Introduce the validated Turn Plan and atomic state boundary." },
  { start: 50, end: 70, key: "experiments", eyebrow: "MEASURE, REJECT, FREEZE", title: "We refused to buy score with unbounded latency.", note: "Show the predeclared reranker gates and a rejected deeper model." },
  { start: 70, end: 88, key: "integrity", eyebrow: "THE HONESTY GATE", title: "A 0.9628 shortcut failed the real-shopping test.", note: "Explain why it was removed and how robustness gates replaced it." },
  { start: 88, end: 128, key: "session", eyebrow: "PLACEHOLDER · LIVE INTENT OVERRIDE", title: "Now watch the architecture handle a shopper changing direction.", note: "Replace this card with your real session capture." },
  { start: 128, end: 148, key: "connected", eyebrow: "OPTIONAL SEMANTIC RANKING", title: "Connected intelligence improves rank—without owning correctness.", note: "Show the paired quality/latency trade-off and fail-open behavior." },
  { start: 148, end: 172, key: "proof", eyebrow: "PLACEHOLDER · OFFICIAL EVALUATOR", title: "One story is a demo. Robustness is the evidence.", note: "Insert the full evaluator capture, then show exact/paraphrase/novel results." },
  { start: 172, end: 180, key: "close", eyebrow: "THE RESULT", title: "High quality. Seconds per turn. Valid without the network.", note: "Close on the engineering principle, not a leaderboard claim." },
];

const variants = {
  A: { name: "Engineering journey", render: renderCinematic },
  B: { name: "Director's evidence cut", render: renderStoryboard },
  C: { name: "Pareto proof reel", render: renderProofReel },
};

const state = {
  variant: getVariant(),
  time: 0,
  playing: false,
  speed: 1,
  lastFrame: performance.now(),
};

const app = document.querySelector("#app");

function getVariant() {
  const value = new URLSearchParams(location.search).get("variant")?.toUpperCase();
  return variants[value] ? value : "A";
}

function currentScene() {
  return SCENES.find((scene) => state.time >= scene.start && state.time < scene.end) ?? SCENES.at(-1);
}

function setVariant(key) {
  state.variant = key;
  const params = new URLSearchParams(location.search);
  params.set("variant", key);
  history.replaceState({}, "", `${location.pathname}?${params}`);
  render();
}

function cycleVariant(direction) {
  const keys = Object.keys(variants);
  const next = (keys.indexOf(state.variant) + direction + keys.length) % keys.length;
  setVariant(keys[next]);
}

function formatTime(seconds) {
  const value = Math.max(0, Math.min(180, Math.round(seconds)));
  return `${Math.floor(value / 60)}:${String(value % 60).padStart(2, "0")}`;
}

function render() {
  app.innerHTML = variants[state.variant].render(currentScene());
  bindControls();
}

function bindControls() {
  document.querySelector("[data-play]")?.addEventListener("click", () => {
    if (state.time >= 180) state.time = 0;
    state.playing = !state.playing;
    render();
  });
  document.querySelector("[data-restart]")?.addEventListener("click", () => {
    state.time = 0;
    state.playing = false;
    render();
  });
  document.querySelector("[data-speed]")?.addEventListener("click", () => {
    state.speed = state.speed === 1 ? 2 : state.speed === 2 ? 4 : 1;
    render();
  });
  document.querySelector("[data-scrub]")?.addEventListener("input", (event) => {
    state.time = Number(event.target.value);
    render();
  });
  document.querySelector("[data-prev-variant]")?.addEventListener("click", () => cycleVariant(-1));
  document.querySelector("[data-next-variant]")?.addEventListener("click", () => cycleVariant(1));
  document.querySelectorAll("[data-seek]").forEach((element) => {
    element.addEventListener("click", () => {
      state.time = Number(element.dataset.seek);
      render();
    });
  });
}

function chrome(content, options = {}) {
  const progress = (state.time / 180) * 100;
  return `
    <div class="prototype-shell ${options.shellClass ?? ""}">
      <div class="prototype-warning">
        <span>PROTOTYPE</span>
        <span>Mock data · placeholders only · no evaluator or agent execution</span>
      </div>
      ${content}
      <div class="video-controls">
        <button class="control-button" data-play aria-label="${state.playing ? "Pause" : "Play"}">${state.playing ? pauseIcon() : playIcon()}</button>
        <button class="control-button" data-restart aria-label="Restart">${restartIcon()}</button>
        <span class="timecode">${formatTime(state.time)} <i>/</i> 3:00</span>
        <input data-scrub class="scrubber" type="range" min="0" max="180" step="1" value="${state.time}" style="--progress:${progress}%" aria-label="Video position" />
        <button class="speed-button" data-speed>${state.speed}×</button>
      </div>
      <div class="variant-switcher" aria-label="Prototype variants">
        <button data-prev-variant aria-label="Previous variant">←</button>
        <span><b>${state.variant}</b> — ${variants[state.variant].name}</span>
        <button data-next-variant aria-label="Next variant">→</button>
      </div>
    </div>`;
}

function renderCinematic(scene) {
  return chrome(`
    <section class="video-stage cinematic scene-${scene.key}">
      <div class="grain"></div>
      <div class="cinematic-topline">
        <span>SHOPPING COPILOT</span>
        <span>TECHJAM · TRACK 4</span>
      </div>
      <div class="cinematic-content">
        ${sceneVisual(scene)}
      </div>
      <div class="scene-caption">
        <span>${scene.eyebrow}</span>
        <p>${scene.note}</p>
      </div>
    </section>
  `, { shellClass: "dark-shell" });
}

function renderStoryboard(scene) {
  return chrome(`
    <section class="storyboard-layout">
      <aside class="storyboard-sidebar">
        <div>
          <p class="kicker">3:00 VIDEO PLAN</p>
          <h1>Director's<br />storyboard</h1>
        </div>
        <nav class="scene-list">
          ${SCENES.map((item) => `
            <button data-seek="${item.start}" class="scene-list-item ${item.key === scene.key ? "active" : ""}">
              <span>${formatTime(item.start)}</span>
              <strong>${item.eyebrow.replace("PLACEHOLDER · ", "")}</strong>
            </button>`).join("")}
        </nav>
        <div class="director-note">
          <span>DIRECTOR NOTE</span>
          <p>${scene.note}</p>
        </div>
      </aside>
      <div class="storyboard-canvas">
        <div class="frame-label"><span>FRAME 16:9</span><span>${formatTime(scene.start)}–${formatTime(scene.end)}</span></div>
        <div class="frame-preview scene-${scene.key}">${sceneVisual(scene)}</div>
        <div class="voiceover-panel">
          <span>VOICEOVER</span>
          <p>${voiceover(scene.key)}</p>
        </div>
      </div>
    </section>
  `, { shellClass: "paper-shell" });
}

function renderProofReel(scene) {
  return chrome(`
    <section class="proof-layout">
      <header class="proof-header">
        <div><span class="signal-dot"></span> SHOPPING COPILOT / TECHNICAL CUT</div>
        <div class="proof-score">0.7556 <span>LOCAL · FULL 200</span></div>
      </header>
      <div class="proof-grid">
        <div class="proof-primary scene-${scene.key}">${sceneVisual(scene, true)}</div>
        <aside class="proof-rail">
          <p class="kicker">ROBUSTNESS BOARD</p>
          ${metricMini("Exact", 0.755552, 0.10671)}
          ${metricMini("Paraphrased", 0.725042, 0.10671)}
          ${metricMini("Novel targets", 0.718247, 0.10671)}
          <div class="fallback-card">
            <span>LOCAL FALLBACK</span>
            <strong>0 tokens</strong>
            <small>valid without network</small>
          </div>
        </aside>
      </div>
      <div class="proof-timeline">
        ${SCENES.map((item) => `<button data-seek="${item.start}" class="${item.key === scene.key ? "active" : ""}"><i></i><span>${formatTime(item.start)}</span></button>`).join("")}
      </div>
    </section>
  `, { shellClass: "proof-shell" });
}

function sceneVisual(scene, compact = false) {
  switch (scene.key) {
    case "hook":
      return `
        <div class="hook-visual">
          <p class="eyebrow">${scene.eyebrow}</p>
          <h1><span>50,000</span> products.<br />One hidden purchase.</h1>
          <div class="ten-turns">10 <small>turns<br />maximum</small></div>
        </div>`;
    case "baseline":
      return baselineVisual();
    case "insight":
      return architectureDiagram(compact);
    case "experiments":
      return experimentVisual();
    case "integrity":
      return integrityVisual();
    case "session":
      return placeholderSession(true);
    case "connected":
      return connectedVisual();
    case "proof":
      return metricsVisual();
    case "close":
      return `
        <div class="close-visual">
          <p class="eyebrow">${scene.eyebrow}</p>
          <div class="final-score"><span>0.7556</span><small>local · full 200</small></div>
          <h2>High quality. Seconds per turn.<br />Valid without the network.</h2>
          <p>github.com/your-team/your-repository</p>
        </div>`;
  }
}

function baselineVisual() {
  return `
    <div class="baseline-visual">
      <div class="baseline-copy">
        <p class="eyebrow">WHERE WE STARTED</p>
        <h2>Every turn looked like<br />a brand-new search.</h2>
        <p>The weak BM25 starter forgot earlier disclosures, could not retire stale intent, and ranked from one message at a time.</p>
      </div>
      <div class="baseline-score-block">
        <span>OFFICIAL BASELINE</span><strong>0.1067</strong><small>TechnicalScore</small>
        <dl><div><dt>Hit@10</dt><dd>0.125</dd></div><div><dt>MRR</dt><dd>0.068</dd></div><div><dt>MTTC</dt><dd>9.81</dd></div></dl>
      </div>
      <div class="forgotten-turns">
        <span>“for hiking”</span><i></i><span class="forgotten">forgotten</span>
        <span>“waterproof”</span><i></i><span class="forgotten">forgotten</span>
        <span>“under $100”</span><i></i><span class="kept">only this turn survives</span>
      </div>
    </div>`;
}

function experimentVisual() {
  return `
    <div class="experiment-visual">
      <div class="experiment-heading"><p class="eyebrow">MEASURE, REJECT, FREEZE</p><h2>Better score did not automatically mean better engineering.</h2></div>
      <div class="experiment-grid">
        <article class="experiment-card selected">
          <div><span>SELECTED</span><small>depth 50</small></div><strong>0.5072</strong><p>TechnicalScore</p>
          <dl><div><dt>p95 rerank</dt><dd>0.55 s</dd></div><div><dt>200-session wall</dt><dd>800.6 s</dd></div></dl>
        </article>
        <div class="decision-rule"><span>PREDECLARED GATES</span><i></i><b>≤1.5 s p95</b><b>≤900 s wall</b></div>
        <article class="experiment-card rejected">
          <div><span>REJECTED</span><small>depth 100</small></div><strong>0.5190</strong><p>slightly higher score</p>
          <dl><div><dt>p95 rerank</dt><dd>1.09 s</dd></div><div><dt>200-session wall</dt><dd>1,366.9 s</dd></div></dl>
        </article>
      </div>
      <p class="experiment-caption">The gate was written before the result. We kept the deepest configuration that met both quality and runtime constraints.</p>
    </div>`;
}

function integrityVisual() {
  return `
    <div class="integrity-visual">
      <div class="integrity-score"><span>METRIC-ONLY SHORTCUT</span><strong>0.9628</strong><div class="rejected-stamp">REMOVED</div></div>
      <div class="integrity-copy">
        <p class="eyebrow">THE HONESTY GATE</p><h2>It scored higher.<br />It understood less.</h2>
        <p>The route mirrored simulator wording, failed under paraphrase, and delayed useful recommendations to protect MRR. We deleted it.</p>
        <div class="gate-list"><span>NO EVALUATOR IMPORTS</span><span>PARAPHRASE TEST</span><span>NOVEL-TARGET TEST</span></div>
      </div>
    </div>`;
}

function connectedVisual() {
  return `
    <div class="connected-visual">
      <div class="connected-heading"><p class="eyebrow">OPTIONAL SEMANTIC RANKING · PAIRED 60</p><h2>Quality when connected.<br />Correctness when disconnected.</h2></div>
      <div class="pareto-chart">
        <div class="axis axis-y"><span>TechnicalScore</span></div><div class="axis axis-x"><span>p95 complete-turn latency</span></div>
        <div class="pareto-point local-point" style="--x:12%;--y:42%"><i></i><b>LOCAL</b><strong>0.7222</strong><small>0.72 s</small></div>
        <div class="pareto-point connected-point" style="--x:70%;--y:26%"><i></i><b>MINI-RANKED</b><strong>0.7569</strong><small>4.59 s</small></div>
        <svg class="pareto-line" viewBox="0 0 100 100" preserveAspectRatio="none"><path d="M17 58 C42 50 55 39 74 31" /></svg>
      </div>
      <div class="fail-open-flow"><span>timeout · malformed output · budget exhausted</span><i>→</i><strong>return local ordering</strong></div>
    </div>`;
}

function architectureDiagram(compact = false) {
  return `
    <div class="architecture-visual ${compact ? "compact" : ""}">
      <div class="architecture-heading"><p class="eyebrow">THE ARCHITECTURAL INSIGHT</p><h2>Meaning is proposed.<br />State is validated.</h2></div>
      <div class="architecture-flow">
        ${diagramNode("01", "Turn Plan", "LLM or local interpreter proposes", "message")}
        ${flowArrow("validate")}
        <div class="route-stack">
          ${diagramNode("02A", "Atomic commit", "all transitions or none", "buying")}
          ${diagramNode("02B", "Constraint State", "revisioned · deterministic", "browsing")}
        </div>
        ${flowArrow("retrieve")}
        ${diagramNode("03", "Hybrid retrieval", "Keyword · dense · category", "retrieval")}
        ${flowArrow("rank")}
        ${diagramNode("04", "Semantic rank", "Top-10 or clarify", "rank")}
      </div>
      <div class="memory-loop"><span>ONE VALIDATED BOUNDARY</span><i></i><span>connected and local paths share the same invariants</span></div>
    </div>`;
}

function diagramNode(index, title, subtitle, className) {
  return `<div class="diagram-node node-${className}"><span>${index}</span><strong>${title}</strong><small>${subtitle}</small></div>`;
}

function flowArrow(label) {
  return `<div class="flow-arrow"><span>${label}</span><svg viewBox="0 0 90 24" aria-hidden="true"><path d="M2 12h80m-8-8 8 8-8 8" /></svg></div>`;
}

function placeholderSession(showOverride) {
  return `
    <div class="session-visual">
      <div class="placeholder-ribbon">PLACEHOLDER · REPLACE WITH REAL CAPTURE</div>
      <div class="conversation-panel">
        <p class="panel-label">SIMULATED CONVERSATION</p>
        <div class="chat user"><span>SHOPPER</span><p>I need something comfortable for an outdoor wedding.</p></div>
        <div class="chat agent"><span>COPILOT</span><p>Do you have a preferred type or weather requirement?</p></div>
        ${showOverride ? `<div class="chat user highlight"><span>SHOPPER · INTENT OVERRIDE</span><p>Actually, it might rain. I need waterproof shoes.</p></div>` : `<div class="chat user"><span>SHOPPER</span><p>Low-profile shoes, preferably neutral.</p></div>`}
      </div>
      <div class="state-panel">
        <div class="state-header"><span>LIVE STATE</span><span>TURN ${showOverride ? "4" : "2"} / 10</span></div>
        <dl>
          <div><dt>intent</dt><dd>BUYING</dd></div>
          <div><dt>category</dt><dd>shoes</dd></div>
          <div class="${showOverride ? "cleared" : ""}"><dt>style</dt><dd>${showOverride ? "cleared" : "low-profile"}</dd></div>
          <div class="${showOverride ? "cleared" : ""}"><dt>color</dt><dd>${showOverride ? "cleared" : "neutral"}</dd></div>
          <div class="${showOverride ? "added" : "muted"}"><dt>feature</dt><dd>${showOverride ? "+ waterproof" : "—"}</dd></div>
        </dl>
        <div class="candidate-funnel"><span>50,000</span><i>→</i><span>184</span><i>→</i><strong>10</strong></div>
      </div>
    </div>`;
}

function metricsVisual() {
  return `
    <div class="metrics-visual">
      <div class="metrics-title"><p class="eyebrow">PLACEHOLDER · OFFICIAL EVALUATOR</p><h2>Robustness, not one lucky run.</h2></div>
      <div class="robustness-comparison">
        <article class="journey-score"><span>OFFICIAL BASELINE</span><strong>0.1067</strong><small>200 sessions</small></article>
        <div class="journey-arrow"><span>7.08×</span><i>→</i><small>technical score</small></div>
        <article class="journey-score shipped"><span>SHIPPED LOCAL</span><strong>0.7556</strong><small>exact · full 200</small></article>
      </div>
      <div class="robustness-strip">
        <div><span>EXACT</span><strong>0.7556</strong></div>
        <div><span>PARAPHRASED</span><strong>0.7250</strong><small>−0.031</small></div>
        <div><span>NOVEL TARGETS</span><strong>0.7182</strong><small>unseen public labels</small></div>
        <div class="tests-pass"><span>TEST SUITE</span><strong>218 / 218</strong><small>passing</small></div>
      </div>
      <div class="metric-footnote">Mock rendering of checked-in journal evidence · replace the placeholder with the official evaluator capture</div>
    </div>`;
}

function scoreCard(metric, className) {
  return `
    <article class="score-card ${className}">
      <span>${metric.label}</span>
      <strong>${metric.score.toFixed(4)}</strong>
      <div class="score-bar"><i style="width:${metric.score * 100}%"></i></div>
      <dl><div><dt>Hit@10</dt><dd>${metric.hit.toFixed(3)}</dd></div><div><dt>MRR</dt><dd>${metric.mrr.toFixed(3)}</dd></div><div><dt>MTTC</dt><dd>${metric.mttc.toFixed(2)}</dd></div></dl>
    </article>`;
}

function metricMini(label, ranked, local) {
  return `<div class="mini-metric"><span>${label}</span><strong>${ranked.toFixed(3)}</strong><div><i style="width:${ranked * 100}%"></i></div><small>baseline ${local.toFixed(3)}</small></div>`;
}

function voiceover(key) {
  const lines = {
    hook: "Fifty thousand products, one hidden purchase, and only ten turns to find it.",
    baseline: "The official baseline treated every message like a new search. It hit only twelve and a half percent of targets and scored 0.1067.",
    insight: "We separated probabilistic interpretation from deterministic correctness. Every local or connected plan passes one atomic validation boundary.",
    experiments: "We wrote runtime gates before benchmarking. A deeper reranker scored slightly higher, but violated the complete-run budget, so we rejected it.",
    integrity: "We even found a shortcut that scored 0.9628 by mirroring simulator wording. It failed paraphrase, so we deleted it and made robustness a release gate.",
    session: "In this intent-override session, watch the old product intent retire atomically while session-level evidence survives and immediately drives a new retrieval.",
    connected: "Optional semantic ranking improves the paired score to 0.7569 in 4.59 seconds p95. If it times out or fails, the valid local ordering returns unchanged.",
    proof: "The shipped local system scores 0.7556 on all two hundred released sessions, 0.7250 under paraphrase, and 0.7182 on novel targets.",
    close: "This is not the highest score at any cost. It is a fast, reproducible shopping agent whose correctness does not depend on the network.",
  };
  return lines[key];
}

function playIcon() { return `<svg viewBox="0 0 24 24"><path d="m8 5 11 7-11 7Z" /></svg>`; }
function pauseIcon() { return `<svg viewBox="0 0 24 24"><path d="M8 5v14M16 5v14" /></svg>`; }
function restartIcon() { return `<svg viewBox="0 0 24 24"><path d="M5 7v5h5M6 17a8 8 0 1 0 0-10l-1 1" /></svg>`; }

function tick(now) {
  const delta = (now - state.lastFrame) / 1000;
  state.lastFrame = now;
  if (state.playing) {
    const previousScene = currentScene().key;
    state.time = Math.min(180, state.time + delta * state.speed);
    if (state.time >= 180) state.playing = false;
    if (previousScene !== currentScene().key || Math.floor(state.time) !== Math.floor(state.time - delta * state.speed)) render();
    else {
      const scrubber = document.querySelector("[data-scrub]");
      const timecode = document.querySelector(".timecode");
      if (scrubber) { scrubber.value = state.time; scrubber.style.setProperty("--progress", `${(state.time / 180) * 100}%`); }
      if (timecode) timecode.innerHTML = `${formatTime(state.time)} <i>/</i> 3:00`;
    }
  }
  requestAnimationFrame(tick);
}

window.addEventListener("keydown", (event) => {
  if (["INPUT", "TEXTAREA"].includes(document.activeElement?.tagName) || document.activeElement?.isContentEditable) return;
  if (event.key === "ArrowLeft") cycleVariant(-1);
  if (event.key === "ArrowRight") cycleVariant(1);
  if (event.key === " ") {
    event.preventDefault();
    state.playing = !state.playing;
    render();
  }
});

render();
requestAnimationFrame(tick);
