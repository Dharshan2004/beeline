// PROTOTYPE — Three demo-video treatments, switchable with ?variant=A|B|C.
// Session excerpts were recorded from tools.demo_session; this page never invokes the Agent/evaluator.

const METRICS = {
  baseline: { label: "Official baseline", hit: 0.125, mrr: 0.068, mttc: 9.81, score: 0.1067 },
  local: { label: "Local-only", hit: 0.9, mrr: 0.521779, mttc: 5.216667, score: 0.7222, p95: "0.72 s" },
  ranked: { label: "Mini-ranked", hit: 0.916667, mrr: 0.586435, mttc: 4.866667, score: 0.756931, p95: "4.59 s" },
};

const SCENES = [
  { start: 0, end: 14, key: "hook", eyebrow: "THE PROBLEM", title: "Attention is the scarce resource.", note: "Name the product and the live-commerce thesis." },
  { start: 14, end: 31, key: "tiktok_context", eyebrow: "WHY TIKTOK SHOP", title: "Buying intent is a short-lived conversion window.", note: "Show the browse-to-buy funnel and why speed matters." },
  { start: 31, end: 46, key: "demo_one", eyebrow: "BEELINE · SESSION 1", title: "One message in. Rank one. Done.", note: "Real helper replay: public_0044." },
  { start: 46, end: 66, key: "demo_questions", eyebrow: "BEELINE · SESSION 2", title: "Every question must eliminate uncertainty.", note: "Real helper replay: public_0019." },
  { start: 66, end: 81, key: "demo_override", eyebrow: "BEELINE · SESSION 3", title: "A changed mind must not corrupt the session.", note: "Real helper replay: public_0013." },
  { start: 81, end: 105, key: "architecture", eyebrow: "THE ARCHITECTURE", title: "One correctness boundary. Two execution planes.", note: "Separate validated state from bounded retrieval and ranking." },
  { start: 105, end: 123, key: "tournament", eyebrow: "THE RANKING DIFFERENTIATOR", title: "Parallelism buys coverage, not waiting time.", note: "Four nano chunks read 48 candidates concurrently." },
  { start: 123, end: 143, key: "dev_process", eyebrow: "THE DEVELOPMENT PROCESS", title: "Measure. Gate. Reject. Freeze.", note: "Show benchmark-led decisions and rejected configurations." },
  { start: 143, end: 163, key: "proof", eyebrow: "THE PROOF", title: "The bare evaluator command measures the shipped agent.", note: "Official exact score and robustness evidence." },
  { start: 163, end: 173, key: "honesty", eyebrow: "THE HONESTY", title: "A gain ships only if it survives distribution shift.", note: "Paraphrase, novel targets, and fail-open behavior." },
  { start: 173, end: 180, key: "close", eyebrow: "BEELINE", title: "The shortest honest path from attention to purchase.", note: "Close on product value, cost, and reproducibility." },
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
        <span>Recorded helper excerpts · no live evaluator or agent execution</span>
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
        <div class="proof-score">0.8058 <span>NANO TOURNAMENT · FULL 200</span></div>
      </header>
      <div class="proof-grid">
        <div class="proof-primary scene-${scene.key}">${sceneVisual(scene, true)}</div>
        <aside class="proof-rail">
          <p class="kicker">ROBUSTNESS BOARD</p>
          ${metricMini("Exact", 0.805848, 0.10671)}
          ${metricMini("Paraphrased", 0.763434, 0.10671)}
          ${metricMini("Novel targets", 0.755807, 0.10671)}
          <div class="fallback-card">
            <span>LOCAL FALLBACK</span>
            <strong>0.756</strong>
            <small>offline · $0 · valid without network</small>
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
      return beelineHook();
    case "tiktok_context":
      return tiktokContextVisual();
    case "demo_one":
      return demoOneTurn();
    case "demo_questions":
      return demoQuestions();
    case "demo_override":
      return demoOverride();
    case "architecture":
      return shippedArchitecture();
    case "tournament":
      return tournamentVisual();
    case "dev_process":
      return developmentVisual();
    case "proof":
      return proofVisual();
    case "honesty":
      return honestyVisual();
    case "close":
      return `
        <div class="close-visual">
          <p class="eyebrow">TEAM TECHBROS · ${scene.eyebrow}</p>
          <div class="beeline-wordmark">BEELINE<i></i></div>
          <h2>The shortest honest path<br />from attention to purchase.</h2>
          <div class="close-modes"><span><b>OFFLINE</b> 0.756 · $0</span><span><b>CONNECTED</b> 0.806 · ~$0.01/session</span></div>
          <p>github.com/Dharshan2004/beeline</p>
        </div>`;
  }
}

function beelineHook() {
  return `
    <div class="beeline-hook">
      <p class="eyebrow">A SHOPPING AGENT FOR TIKTOK SHOP</p>
      <h1>Seconds.<br /><span>Not minutes.</span></h1>
      <p>Every unnecessary question is another chance for the buyer to scroll on.</p>
      <div class="beeline-thesis"><strong>BEELINE</strong><i></i><span>how few turns to the exact product?</span></div>
    </div>`;
}

function tiktokContextVisual() {
  return `
    <div class="tiktok-context-visual">
      <div class="tiktok-copy">
        <p class="eyebrow">WHY THIS MATTERS FOR TIKTOK SHOP</p>
        <h2>Attention creates traffic.<br /><span>Intent creates revenue.</span></h2>
        <p>Most people are browsing. A smaller group reveals buying intent—and that intent can disappear with the next swipe.</p>
      </div>
      <div class="intent-window">
        <div class="audience-stage"><span>ALL VIEWERS</span><strong>browse</strong><small>discover · compare · entertain</small></div>
        <i>→</i>
        <div class="audience-stage shoppers"><span>BUYING SIGNAL</span><strong>intent</strong><small>requirements become concrete</small></div>
        <i>→</i>
        <div class="audience-stage conversion"><span>BEELINE WINDOW</span><strong>convert</strong><small>before attention moves on</small></div>
      </div>
      <div class="attention-clock">
        <div><span>EVERY EXTRA TURN</span><strong>adds uncertainty + attrition</strong></div>
        <div><span>EVERY EXTRA SECOND</span><strong>risks losing the moment</strong></div>
        <p>Beeline optimizes the path from <b>“I might buy”</b> to <b>the exact product</b>.</p>
      </div>
    </div>`;
}

function terminalHeader(sample, scenario) {
  return `<div class="terminal-header"><span>REAL HELPER REPLAY</span><code>python -m tools.demo_session --sample ${sample}</code><b>${scenario}</b></div>`;
}

function demoOneTurn() {
  return `
    <div class="demo-terminal one-turn-demo">
      ${terminalHeader("public_0044", "BUYING")}
      <div class="terminal-body">
        <div class="terminal-turn"><span>TURN 1 · CUSTOMER</span><p>“I'm looking for Men Jammers. A key requirement is: fabric.”</p></div>
        <div class="terminal-rank"><span>TOP RESULT</span><strong>#1 · K898 Men's Swimming Jammer</strong><small>B09BQ4G5BD · target</small></div>
        <div class="terminal-outcome"><strong>CONVERTED ON TURN 1 · RANK 1</strong><span>0 connected-model tokens</span></div>
      </div>
      <div class="demo-callout"><span>01</span><strong>One message in.</strong><small>The beeline is done.</small></div>
    </div>`;
}

function demoQuestions() {
  return `
    <div class="demo-terminal question-demo">
      ${terminalHeader("public_0019", "BROWSING")}
      <div class="question-trace">
        <article><span>TURN 1</span><p>Outdoor & Work Rain · still exploring</p><strong>ASK material</strong><small>50 candidates → 6 groups · eliminates ≥46</small></article>
        <article><span>TURN 2</span><p>no material preference</p><strong>ASK color</strong><small>dismissed material is never repeated</small></article>
        <article><span>TURN 3</span><p>no color preference</p><strong>ASK feature</strong><small>50 candidates → 8 groups</small></article>
        <article class="hit"><span>TURN 4</span><p>Rubber sole · 5.5-inch shaft</p><strong>TARGET #1</strong><small>Asgard Women's Ankle Rain Boots</small></article>
      </div>
      <div class="policy-callout"><strong>QUESTION POLICY</strong><span>Ask only when an attribute splits the live pool.</span><span>Never re-ask an answered or dismissed attribute.</span></div>
    </div>`;
}

function demoOverride() {
  return `
    <div class="demo-terminal override-demo">
      ${terminalHeader("public_0013", "INTENT OVERRIDE")}
      <div class="override-path">
        <div><span>TURN 1–3</span><strong>Shoes · slippers · textile</strong><small>target remains rank 1</small></div>
        <i>→</i>
        <div class="override-message"><span>TURN 4</span><strong>“Actually, ignore my earlier preference.”</strong><small>What I need is: Rubber sole.</small></div>
        <i>→</i>
        <div class="override-result"><span>ATOMIC UPDATE</span><strong>stale preference retired</strong><small>correct target remains rank 1</small></div>
      </div>
      <div class="state-boundary"><b>one Turn Plan</b><i></i><b>one state revision</b><i></i><b>all transitions—or none</b></div>
    </div>`;
}

function shippedArchitecture() {
  return `
    <div class="shipped-architecture deep-architecture">
      <div class="architecture-heading"><p class="eyebrow">ONE TURN · TWO EXECUTION PLANES</p><h2>Meaning can be probabilistic.<br />Correctness cannot.</h2></div>
      <div class="plane control-plane">
        <div class="plane-name"><span>CONTROL PLANE</span><strong>conversation → trusted state</strong></div>
        <div class="plane-node"><span>01</span><strong>Message + dialog</strong><small>latest turn · prior evidence · budget</small></div><i>→</i>
        <div class="plane-node"><span>02</span><strong>Complete Turn Plan</strong><small>local or model interpretation</small></div><i>→</i>
        <div class="plane-node validator"><span>03</span><strong>Atomic validator</strong><small>all transitions—or none</small></div><i>→</i>
        <div class="plane-node state"><span>04</span><strong>Constraint State</strong><small>revisioned · deterministic</small></div>
      </div>
      <div class="plane data-plane">
        <div class="plane-name"><span>DATA PLANE</span><strong>state → ranked action</strong></div>
        <div class="route-cluster"><b>STRUCTURED</b><b>BM25 / FTS5</b><b>DENSE / QDRANT</b><small>three independent retrieval routes</small></div><i>→</i>
        <div class="plane-node"><span>05</span><strong>Frozen fusion</strong><small>normalize · popularity-aware pool</small></div><i>→</i>
        <div class="plane-node"><span>06</span><strong>Local cross-encoder</strong><small>MiniLM · 1.5 s deadline</small></div><i>→</i>
        <div class="plane-node output"><span>07</span><strong>Recommend or ask</strong><small>Top 10 · highest information value</small></div>
      </div>
      <div class="architecture-contract"><span>ONE CORRECTNESS BOUNDARY</span><i></i><strong>model timeout, malformed output, or budget exhaustion returns the valid local path</strong></div>
    </div>`;
}

function tournamentVisual() {
  return `
    <div class="tournament-visual">
      <div class="tournament-copy"><p class="eyebrow">PARALLEL NANO TOURNAMENT</p><h2>Coverage.<br />Not latency.</h2><p>A single listwise call reads 12 candidates. Four concurrent nano calls read 48 at roughly one-call wall clock.</p></div>
      <div class="tournament-diagram">
        <div class="single-call"><span>SINGLE CALL</span><strong>12</strong><small>candidates seen</small></div>
        <div class="parallel-bracket">
          <div class="chunk-row"><span>12</span><span>12</span><span>12</span><span>12</span></div>
          <strong>4 CHUNKS · IN PARALLEL</strong>
          <i>↓ top 3 from each</i>
          <div class="final-call"><b>12 FINALISTS</b><span>one nano final</span></div>
        </div>
        <div class="latency-lock"><span>SHIPPED P95</span><strong>3.44 s</strong><small>1.2 s chunks · 1.6 s final · fail-open</small></div>
      </div>
    </div>`;
}

function developmentVisual() {
  return `
    <div class="development-visual">
      <div class="development-heading"><p class="eyebrow">BENCHMARK-DRIVEN DEVELOPMENT</p><h2>Measure. Gate.<br />Reject. Freeze.</h2><p>Every component had to improve shopping quality inside a declared runtime envelope.</p></div>
      <div class="decision-ledger">
        <article class="accepted"><span>LOCAL RERANK DEPTH 50</span><strong>SELECTED</strong><dl><div><dt>score</dt><dd>0.5072</dd></div><div><dt>p95</dt><dd>0.55 s</dd></div><div><dt>full-run wall</dt><dd>800.6 s</dd></div></dl></article>
        <article class="rejected"><span>LOCAL RERANK DEPTH 100</span><strong>REJECTED</strong><dl><div><dt>score</dt><dd>0.5190</dd></div><div><dt>p95</dt><dd>1.09 s</dd></div><div><dt>full-run wall</dt><dd>1,366.9 s</dd></div></dl><small>failed the predeclared 900 s gate</small></article>
        <article class="tightened"><span>NANO TOURNAMENT</span><strong>TIGHTENED</strong><dl><div><dt>initial p95</dt><dd>4.2 s</dd></div><div><dt>shipped p95</dt><dd>3.44 s</dd></div><div><dt>score</dt><dd>0.8058</dd></div></dl><small>timeouts reduced to meet the &lt;4 s product constraint</small></article>
      </div>
      <div class="release-gate"><strong>RELEASE GATE</strong><span>exact</span><i>+</i><span>paraphrased</span><i>+</i><span>novel targets</span><b>→</b><small>ship only when the gain survives all three</small></div>
    </div>`;
}

function proofVisual() {
  return `
    <div class="proof-visual-new">
      <div class="evaluator-terminal"><span>OFFICIAL EVALUATOR · UNMODIFIED</span><code>$ python -m evaluator.local_evaluator</code><small>same bare command · connected with key · offline without one</small></div>
      <div class="proof-score-grid">
        <article><span>WEAK BM25 BASELINE</span><strong>0.107</strong><small>TechnicalScore</small></article>
        <i>→</i>
        <article class="offline"><span>BEELINE OFFLINE</span><strong>0.756</strong><small>0.8 s p95 · $0</small></article>
        <i>→</i>
        <article class="connected"><span>NANO TOURNAMENT</span><strong>0.806</strong><small>3.44 s p95 · ~$0.01/session</small></article>
      </div>
      <div class="proof-kpis"><div><span>HITRATE@10</span><strong>0.960</strong></div><div><span>MRR</span><strong>0.628</strong></div><div><span>MTTC</span><strong>4.13</strong></div><div><span>INVALID OUTPUTS</span><strong>0</strong></div></div>
    </div>`;
}

function honestyVisual() {
  return `
    <div class="honesty-visual">
      <div><p class="eyebrow">THE HONESTY GATE</p><h2>Exact score is not enough.</h2><p>Every customer message is reworded. Then 100 targets that appear in no public label are tested through the same scorer.</p></div>
      <div class="honesty-scores">
        <article><span>EXACT · 200</span><strong>0.806</strong></article>
        <article><span>PARAPHRASED · 200</span><strong>0.763</strong></article>
        <article><span>NOVEL TARGETS · 100</span><strong>0.756</strong></article>
      </div>
      <div class="honesty-rule"><strong>BENCHMARK-SPECIFIC GAINS DO NOT SHIP</strong><span>no evaluator imports · no template matching · model failure returns the valid local order</span></div>
    </div>`;
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
    hook: "On TikTok Shop, you have seconds, not minutes. Beeline is built around how few turns it takes to reach the exact product.",
    tiktok_context: "Most people are browsing. Only some reveal buying intent, and that conversion window can disappear with the next swipe. Beeline has to close before attention moves on.",
    demo_one: "Public 0044 converts at rank one from the first message.",
    demo_questions: "Public 0019 shows the information-value policy: each question splits the live pool, and dismissed attributes are never repeated.",
    demo_override: "Public 0013 changes its requirements on turn four. One atomic update retires stale evidence while the correct result stays at rank one.",
    architecture: "The control plane converts conversation into one atomically validated state revision. The data plane uses that state across three retrieval routes, frozen fusion, bounded local reranking, and either a Top Ten or one useful question.",
    tournament: "Four parallel nano chunks read forty-eight candidates at roughly one-call wall clock, then one final ranks the twelve leaders.",
    dev_process: "We predeclared runtime gates, rejected deeper configurations that broke them, tightened the connected path below four seconds, and required every gain to survive exact, paraphrased, and novel-target evaluation.",
    proof: "The unmodified official evaluator moves from a 0.107 starter to 0.806, with 0.960 HitRate at Ten and 3.44 seconds p95.",
    honesty: "The score survives full rewording and a hundred targets absent from public labels. Benchmark-specific gains do not ship.",
    close: "Beeline is the shortest honest path from attention to purchase.",
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
