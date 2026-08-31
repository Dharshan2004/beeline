// PROTOTYPE — Three demo-video treatments, switchable with ?variant=A|B|C.
// This uses mock content only. It never imports or invokes the real Agent/evaluator.

const METRICS = {
  baseline: { label: "Official baseline", hit: 0.125, mrr: 0.068, mttc: 9.81, score: 0.1067 },
  local: { label: "Local-only", hit: 0.9, mrr: 0.521779, mttc: 5.216667, score: 0.7222, p95: "0.72 s" },
  ranked: { label: "Mini-ranked", hit: 0.916667, mrr: 0.586435, mttc: 4.866667, score: 0.756931, p95: "4.59 s" },
};

const SCENES = [
  { start: 0, end: 12, key: "hook", eyebrow: "THE CHALLENGE", title: "50,000 products. One hidden purchase.", note: "Open with the evaluator's real constraint: ten turns." },
  { start: 12, end: 28, key: "baseline", eyebrow: "WHERE WE STARTED", title: "The baseline searched each message in isolation.", note: "Show the official 0.1067 baseline and one failure trace." },
  { start: 28, end: 52, key: "insight", eyebrow: "THE CONTROL PLANE", title: "Conversation becomes validated, revisioned state.", note: "Show the boundary between probabilistic interpretation and deterministic state." },
  { start: 52, end: 80, key: "retrieval", eyebrow: "THE RETRIEVAL DATA PLANE", title: "Three routes. Frozen fusion. Bounded ranking.", note: "Trace the full local candidate and ranking pipeline." },
  { start: 80, end: 100, key: "levers", eyebrow: "MEASURE, REJECT, IMPROVE", title: "We optimized shopping quality inside a runtime envelope.", note: "Connect score gains to behavior and the frozen reranker boundary." },
  { start: 100, end: 132, key: "session", eyebrow: "LIVE DEMO SESSION", title: "The target appears only after distinguishing evidence.", note: "A real four-turn run through tools.demo_session." },
  { start: 132, end: 160, key: "connected", eyebrow: "TIKTOK SHOP TRADE-OFF", title: "Conversion quality matters only while intent is alive.", note: "Show the paired quality/latency trade-off and fail-open behavior." },
  { start: 160, end: 175, key: "proof", eyebrow: "ROBUSTNESS EVIDENCE", title: "One story is a demo. Robustness is the evidence.", note: "Exact, paraphrased, novel-target, and test-suite evidence." },
  { start: 175, end: 180, key: "close", eyebrow: "THE RESULT", title: "High quality. Seconds per turn. Valid without the network.", note: "Close on the engineering principle, not a leaderboard claim." },
];

const variants = {
  A: { name: "Engineering journey", render: renderCinematic },
  B: { name: "Director's evidence cut", render: renderStoryboard },
  C: { name: "Pareto proof reel", render: renderProofReel },
};

const params = new URLSearchParams(location.search);
const CAPTURE_MODE = params.get("capture") === "1";

const state = {
  variant: getVariant(),
  time: Math.max(0, Math.min(179.9, Number(params.get("time")) || 0)),
  playing: false,
  speed: 1,
  lastFrame: performance.now(),
};

const app = document.querySelector("#app");
document.documentElement.classList.toggle("capture-mode", CAPTURE_MODE);

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
        <span>TEAM TECHBROS · SHOPPING COPILOT</span>
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
    case "retrieval":
      return retrievalVisual();
    case "levers":
      return improvementVisual();
    case "session":
      return demoSession();
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
          <p>github.com/Dharshan2004/techjam-2026-track-4-shopping-copilot</p>
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

function improvementVisual() {
  return `
    <div class="improvement-visual">
      <div class="improvement-score">
        <span>160-SESSION DEVELOPMENT SPLIT</span>
        <div class="score-jump"><small>0.5518</small><i>→</i><strong>0.7392</strong></div>
        <p>TechnicalScore <b>+0.1873</b></p>
        <div class="latency-badge"><span>p95 complete turn</span><strong>0.707 s</strong></div>
      </div>
      <div class="improvement-copy">
        <p class="eyebrow">MEASURE, REJECT, IMPROVE</p><h2>Better shopping behavior.<br />Inside a runtime envelope.</h2>
        <div class="lever-list">
          <article><span>01</span><div><strong>Conversational evidence</strong><small>Validated memory across turns.</small></div></article>
          <article><span>02</span><div><strong>Information-value questions</strong><small>Split the live candidate pool.</small></div></article>
          <article><span>03</span><div><strong>Budget-aware ordering</strong><small>Respect ranges; retain unknown prices.</small></div></article>
        </div>
        <div class="runtime-choice"><b>FROZEN RERANK DEPTH 50</b><span>0.55 s p95 · 800.6 s projected wall</span><small>depth 100 rejected: 1,366.9 s wall</small></div>
      </div>
    </div>`;
}

function retrievalVisual() {
  return `
    <div class="retrieval-visual">
      <div class="retrieval-heading"><p class="eyebrow">THE RETRIEVAL DATA PLANE</p><h2>Three routes. Frozen fusion.<br />Bounded ranking.</h2></div>
      <div class="query-context">
        <span>QUERY CONTEXT</span>
        <strong>latest message</strong><i>+</i><strong>prior dialog</strong><i>+</i><strong>active constraints</strong><i>+</i><strong>profile hint while vague</strong>
      </div>
      <div class="retrieval-routes">
        <article class="route structured"><span>ELIGIBILITY</span><strong>Structured filters</strong><small>frozen weight · 0.02</small></article>
        <article class="route lexical"><span>LEXICAL</span><strong>BM25 / FTS5</strong><small>frozen weight · 0.64</small></article>
        <article class="route dense"><span>SEMANTIC</span><strong>Dense / Qdrant Local</strong><small>frozen weight · 0.34 · fail-open</small></article>
      </div>
      <div class="pipeline-band">
        <div><span>01</span><strong>Normalize + fuse</strong><small>per-route min–max</small></div><i>→</i>
        <div><span>02</span><strong>Wide pool · 100</strong><small>hard eligibility + popularity bands</small></div><i>→</i>
        <div><span>03</span><strong>Candidate depth · 50</strong><small>frozen production boundary</small></div><i>→</i>
        <div class="rank-stage"><span>04</span><strong>MiniLM-L6 rerank</strong><small>local · 1.5 s deadline</small></div>
      </div>
      <div class="retrieval-output"><strong>catalog-valid Top 10</strong><span>or</span><strong>highest-information clarification</strong><i></i><small>budget stable partition · optional connected head only on uncertain margin · local ordering survives failure</small></div>
    </div>`;
}

function connectedVisual() {
  return `
    <div class="connected-visual shop-conversion-visual">
      <div class="connected-heading"><p class="eyebrow">A SHOPPING AGENT FOR TIKTOK SHOP</p><h2>Most users browse.<br />A few are ready to buy.</h2><p class="conversion-thesis">The job is to recognize purchase intent, reduce uncertainty, and seal the deal before attention moves on.</p></div>
      <div class="conversion-funnel">
        <div class="funnel-stage browse"><span>01</span><strong>People browsing</strong><small>discovery · entertainment · curiosity</small></div>
        <i>↓</i>
        <div class="funnel-stage intent"><span>02</span><strong>Purchase intent appears</strong><small>constraints narrow · confidence rises</small></div>
        <i>↓</i>
        <div class="funnel-stage close"><span>03</span><strong>Close quickly</strong><small>recommend while attention is still alive</small></div>
      </div>
      <div class="conversion-tradeoff">
        <div><span>FAST LOCAL DEFAULT</span><strong>0.7222</strong><small>0.72 s p95</small></div>
        <b>versus</b>
        <div class="slower"><span>CONNECTED RANKING</span><strong>0.7569</strong><small>4.59 s p95</small></div>
        <p><strong>Trade-off:</strong> a small ranking gain can lose the shopper if latency breaks the moment. Connected ranking is optional; failures return the valid local order.</p>
      </div>
    </div>`;
}

function architectureDiagram(compact = false) {
  return `
    <div class="architecture-visual ${compact ? "compact" : ""}">
      <div class="architecture-heading"><p class="eyebrow">THE CONTROL PLANE</p><h2>Probabilistic meaning.<br />Deterministic state.</h2></div>
      <div class="control-plane-flow">
        ${diagramNode("01", "Conversation input", "latest message + prior dialog", "message")}
        ${flowArrow("propose")}
        ${diagramNode("02", "Complete Turn Plan", "model or local interpreter", "buying")}
        ${flowArrow("validate")}
        ${diagramNode("03", "Atomic validator", "all transitions or none", "validator")}
        ${flowArrow("commit")}
        ${diagramNode("04", "Constraint State", "revisioned + deterministic", "browsing")}
      </div>
      <div class="state-contracts">
        <div><span>RETIRES STALE INTENT</span><strong>new evidence can replace old constraints</strong></div>
        <div><span>PRESERVES INVARIANTS</span><strong>one validated revision per turn</strong></div>
        <div><span>BUILDS RETRIEVAL QUERY</span><strong>state + dialog + current message</strong></div>
      </div>
      <div class="memory-loop"><span>ONE CORRECTNESS BOUNDARY</span><i></i><span>local and connected planners share the same state contract</span></div>
    </div>`;
}

function diagramNode(index, title, subtitle, className) {
  return `<div class="diagram-node node-${className}"><span>${index}</span><strong>${title}</strong><small>${subtitle}</small></div>`;
}

function flowArrow(label) {
  return `<div class="flow-arrow"><span>${label}</span><svg viewBox="0 0 90 24" aria-hidden="true"><path d="M2 12h80m-8-8 8 8-8 8" /></svg></div>`;
}

function demoSession() {
  return `
    <div class="session-visual">
      <div class="conversation-panel">
        <p class="panel-label">REAL HELPER RUN · PUBLIC_0001</p>
        <div class="session-turn"><span>TURN 1</span><p>Jewelry necklaces · alloy</p><small>target absent · asks color</small></div>
        <div class="session-turn"><span>TURN 2</span><p>No additional color preference</p><small>target absent · asks use case</small></div>
        <div class="session-turn"><span>TURN 3</span><p>No additional use-case preference</p><small>target absent · asks feature</small></div>
        <div class="session-turn hit"><span>TURN 4</span><p>Triple Moon Pentagram symbol</p><small>target enters at rank 1</small></div>
      </div>
      <div class="state-panel">
        <div class="state-header"><span>CONVERSION PATH</span><span>TURN 4 / 10</span></div>
        <dl>
          <div><dt>intent</dt><dd>BUYING</dd></div>
          <div><dt>category</dt><dd>necklace</dd></div>
          <div><dt>material</dt><dd>alloy</dd></div>
          <div class="muted"><dt>color</dt><dd>dismissed</dd></div>
          <div class="added"><dt>feature</dt><dd>+ triple moon</dd></div>
        </dl>
        <div class="session-result"><span>TARGET FOUND</span><strong>TURN 4 · RANK 1</strong><small>0 connected-model tokens</small></div>
      </div>
    </div>`;
}

function metricsVisual() {
  return `
    <div class="metrics-visual">
      <div class="metrics-title"><p class="eyebrow">ROBUSTNESS EVIDENCE</p><h2>Robustness, not one lucky run.</h2></div>
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
      <div class="metric-footnote">Checked-in evidence · exact, paraphrased, and novel-target conditions</div>
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
    insight: "A complete Turn Plan passes an atomic validator before it can change revisioned Constraint State. Probabilistic meaning never owns correctness.",
    retrieval: "Structured eligibility, BM25, and local dense retrieval are normalized and fused with frozen weights before a bounded local cross-encoder produces the final ordering.",
    levers: "Conversational evidence, information-value questions, and budget-aware ordering moved the development score from 0.5518 to 0.7392, while a frozen depth of fifty kept the full run inside its runtime gate.",
    session: "In this intent-override session, watch the old product intent retire atomically while session-level evidence survives and immediately drives a new retrieval.",
    connected: "On TikTok Shop, most people are browsing and only some reveal purchase intent. The agent has to reduce uncertainty and close while attention is still alive, so a small ranking gain must be weighed against latency. The fast local order remains the fail-open default.",
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
