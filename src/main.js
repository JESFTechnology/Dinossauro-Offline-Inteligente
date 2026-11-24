import { ACTIONS, ACTION_LABELS, STATES, STATE_MAP, QLearningAgent } from './qlearning.js';
import { DinoGame } from './game.js';
import { LearningChart } from './chart.js';

const agent = new QLearningAgent({
  alpha: 0.25,
  gamma: 0.90,
  epsilon: 0.85,
  epsilonDecay: 0.92,
  epsilonMin: 0.05,
  autoDecay: true,
});

const game = new DinoGame(agent);

// DOM references
const dinoCanvas = document.getElementById('dino-canvas');
const dinoCtx = dinoCanvas.getContext('2d');
const chartCanvas = document.getElementById('learning-chart');
const chartTooltip = document.getElementById('chart-tooltip');
const chart = new LearningChart(chartCanvas, chartTooltip);

const elEpisode = document.getElementById('stat-episode');
const elScore = document.getElementById('stat-score');
const elBest = document.getElementById('stat-best');
const elEpsilon = document.getElementById('stat-epsilon');

const elHudState = document.getElementById('hud-state-label');
const elHudDist = document.getElementById('hud-distance');
const elHudObs = document.getElementById('hud-obstacle');
const elHudAction = document.getElementById('hud-action');
const elHudMode = document.getElementById('hud-policy-mode');

const elBellmanDelta = document.getElementById('bellman-delta');
const elBellmanCalc = document.getElementById('bellman-calc');

const btnToggleRun = document.getElementById('btn-toggle-run');
const btnStep = document.getElementById('btn-step');
const btnResetEp = document.getElementById('btn-reset-ep');
const btnPretrain = document.getElementById('btn-pretrain');
const btnResetQ = document.getElementById('btn-reset-q');
const chkSensors = document.getElementById('chk-sensors');

const modeButtons = document.querySelectorAll('[data-mode]');
const speedButtons = document.querySelectorAll('.speed-btn');

const sliderAlpha = document.getElementById('slider-alpha');
const valAlpha = document.getElementById('val-alpha');
const sliderGamma = document.getElementById('slider-gamma');
const valGamma = document.getElementById('val-gamma');
const sliderEpsilon = document.getElementById('slider-epsilon');
const valEpsilon = document.getElementById('val-epsilon');
const chkDecay = document.getElementById('chk-decay');

const qTableBody = document.getElementById('qtable-body');
const stateCardsGrid = document.getElementById('state-cards-grid');

// Runtime state
let isRunning = true;
let simSpeed = 1;
let hoveredStateId = null;

// Cached DOM nodes for smooth real-time updates
const qRowRefs = {};
const stateCardRefs = {};

function formatQ(value) {
  const formatted = value.toFixed(2);
  return value > 0 ? `+${formatted}` : formatted;
}

function buildQTableDOM() {
  qTableBody.innerHTML = '';

  for (const state of STATES) {
    const tr = document.createElement('tr');
    tr.dataset.stateId = state.id;

    // State cell
    const tdState = document.createElement('td');
    const stateWrap = document.createElement('div');
    stateWrap.className = 'state-name-cell';
    const stateCode = document.createElement('span');
    stateCode.className = 'state-code mono';
    stateCode.textContent = state.label;
    const stateMeta = document.createElement('span');
    stateMeta.className = 'state-meta';
    stateMeta.textContent = state.meta;
    stateWrap.append(stateCode, stateMeta);
    tdState.appendChild(stateWrap);

    // Run Q cell
    const tdRun = document.createElement('td');
    tdRun.className = 'num-col';
    const runWrap = document.createElement('div');
    runWrap.className = 'qcell-wrap';
    const runVal = document.createElement('span');
    runVal.className = 'qval-number';
    runVal.textContent = '0.00';
    const runTrack = document.createElement('div');
    runTrack.className = 'qval-bar-track';
    const runBar = document.createElement('div');
    runBar.className = 'qval-bar-fill zero';
    runTrack.appendChild(runBar);
    runWrap.append(runVal, runTrack);
    tdRun.appendChild(runWrap);

    // Jump Q cell
    const tdJump = document.createElement('td');
    tdJump.className = 'num-col';
    const jumpWrap = document.createElement('div');
    jumpWrap.className = 'qcell-wrap';
    const jumpVal = document.createElement('span');
    jumpVal.className = 'qval-number';
    jumpVal.textContent = '0.00';
    const jumpTrack = document.createElement('div');
    jumpTrack.className = 'qval-bar-track';
    const jumpBar = document.createElement('div');
    jumpBar.className = 'qval-bar-fill zero';
    jumpTrack.appendChild(jumpBar);
    jumpWrap.append(jumpVal, jumpTrack);
    tdJump.appendChild(jumpWrap);

    // Best Policy cell
    const tdPolicy = document.createElement('td');
    const policySpan = document.createElement('span');
    policySpan.className = 'policy-label policy-tie';
    policySpan.textContent = 'Indefinido';
    tdPolicy.appendChild(policySpan);

    tr.append(tdState, tdRun, tdJump, tdPolicy);

    tr.addEventListener('mouseenter', () => setHoveredState(state.id));
    tr.addEventListener('mouseleave', () => setHoveredState(null));

    qTableBody.appendChild(tr);

    qRowRefs[state.id] = {
      tr,
      tdRun,
      runVal,
      runBar,
      tdJump,
      jumpVal,
      jumpBar,
      policySpan,
    };
  }
}

function buildStateCardsDOM() {
  stateCardsGrid.innerHTML = '';

  for (const state of STATES) {
    const card = document.createElement('button');
    card.type = 'button';
    card.className = 'state-card';
    card.dataset.stateId = state.id;

    const head = document.createElement('div');
    head.className = 'state-card-head';

    const title = document.createElement('span');
    title.className = 'state-card-title';
    title.textContent = state.label;

    const policy = document.createElement('span');
    policy.className = 'state-card-policy policy-tie';
    policy.textContent = 'Indefinido';

    head.append(title, policy);

    const desc = document.createElement('p');
    desc.className = 'state-card-desc';
    desc.textContent = state.description;

    const qSummary = document.createElement('div');
    qSummary.className = 'state-card-q mono';
    qSummary.textContent = 'Q(Correr): 0.00 · Q(Saltar): 0.00';

    card.append(head, desc, qSummary);

    card.addEventListener('mouseenter', () => setHoveredState(state.id));
    card.addEventListener('mouseleave', () => setHoveredState(null));
    card.addEventListener('click', () => {
      setHoveredState(hoveredStateId === state.id ? null : state.id);
    });

    stateCardsGrid.appendChild(card);
    stateCardRefs[state.id] = { card, policy, qSummary };
  }
}

function setHoveredState(stateId) {
  hoveredStateId = stateId;
  for (const s of STATES) {
    const row = qRowRefs[s.id];
    const cardRef = stateCardRefs[s.id];
    const isHovered = s.id === stateId;
    if (row) row.tr.classList.toggle('hover-highlight', isHovered);
    if (cardRef) cardRef.card.classList.toggle('is-active', isHovered);
  }
}

function triggerCellFlash(td, isPositive) {
  const cls = isPositive ? 'cell-flash-pos' : 'cell-flash-neg';
  td.classList.remove('cell-flash-pos', 'cell-flash-neg');
  // Force reflow to restart animation cleanly
  void td.offsetWidth;
  td.classList.add(cls);
}

function updateQCellVisual(valEl, barEl, value, maxAbs) {
  valEl.textContent = formatQ(value);
  valEl.classList.remove('pos', 'neg');
  barEl.classList.remove('pos', 'neg', 'zero');

  if (value > 0.005) {
    valEl.classList.add('pos');
    barEl.classList.add('pos');
  } else if (value < -0.005) {
    valEl.classList.add('neg');
    barEl.classList.add('neg');
  } else {
    barEl.classList.add('zero');
  }

  const pct = Math.min(100, Math.round((Math.abs(value) / maxAbs) * 100));
  barEl.style.width = `${Math.max(value === 0 ? 0 : 6, pct)}%`;
}

function renderQTable(activeStateId, updatedTrace = null) {
  // Compute scale across entire Q-table so bars are proportional
  let maxAbs = 20;
  for (const s of STATES) {
    const entry = agent.getEntry(s.id);
    maxAbs = Math.max(maxAbs, Math.abs(entry.run), Math.abs(entry.jump));
  }

  for (const s of STATES) {
    const entry = agent.getEntry(s.id);
    const ref = qRowRefs[s.id];
    const cardRef = stateCardRefs[s.id];

    ref.tr.classList.toggle('active-state', s.id === activeStateId);

    updateQCellVisual(ref.runVal, ref.runBar, entry.run, maxAbs);
    updateQCellVisual(ref.jumpVal, ref.jumpBar, entry.jump, maxAbs);

    const best = agent.getBestAction(s.id);
    ref.policySpan.classList.remove('policy-run', 'policy-jump', 'policy-tie');
    cardRef.policy.classList.remove('policy-run', 'policy-jump', 'policy-tie');

    if (best === ACTIONS.RUN) {
      ref.policySpan.textContent = '→ Correr';
      ref.policySpan.classList.add('policy-run');
      cardRef.policy.textContent = '→ Correr';
      cardRef.policy.classList.add('policy-run');
    } else if (best === ACTIONS.JUMP) {
      ref.policySpan.textContent = '↑ Saltar';
      ref.policySpan.classList.add('policy-jump');
      cardRef.policy.textContent = '↑ Saltar';
      cardRef.policy.classList.add('policy-jump');
    } else {
      ref.policySpan.textContent = 'Empate';
      ref.policySpan.classList.add('policy-tie');
      cardRef.policy.textContent = 'Empate';
      cardRef.policy.classList.add('policy-tie');
    }

    cardRef.qSummary.textContent = `Q(Correr): ${formatQ(entry.run)} · Q(Saltar): ${formatQ(entry.jump)}`;
  }

  if (updatedTrace) {
    const ref = qRowRefs[updatedTrace.stateId];
    if (ref) {
      const targetTd =
        updatedTrace.action === ACTIONS.RUN ? ref.tdRun : ref.tdJump;
      triggerCellFlash(targetTd, updatedTrace.delta >= 0);
    }
  }
}

function renderBellmanTrace(trace) {
  if (!trace) {
    elBellmanDelta.textContent = 'Aguardando atualização';
    elBellmanDelta.classList.remove('positive', 'negative');
    elBellmanCalc.textContent =
      'Execute a simulação para acompanhar cada cálculo de atualização da Tabela Q em tempo real.';
    return;
  }

  const deltaStr = trace.delta >= 0 ? `+${trace.delta.toFixed(2)}` : trace.delta.toFixed(2);
  elBellmanDelta.textContent = `ΔQ = ${deltaStr}`;
  elBellmanDelta.classList.remove('positive', 'negative');
  elBellmanDelta.classList.add(trace.delta >= 0 ? 'positive' : 'negative');

  const rewardStr = trace.reward >= 0 ? `+${trace.reward.toFixed(1)}` : trace.reward.toFixed(1);
  elBellmanCalc.textContent =
    `Q(${trace.stateLabel}, ${trace.actionLabel}) ← ${trace.oldQ.toFixed(2)} + ${trace.alpha.toFixed(2)} · [ ${rewardStr} + ${trace.gamma.toFixed(2)} · ${trace.maxNextQ.toFixed(2)} − (${trace.oldQ.toFixed(2)}) ] = ${trace.newQ.toFixed(2)}`;
}

function syncTelemetryUI(updatedTrace = null) {
  const perception = game.getPerception();
  const stateObj = STATE_MAP[perception.stateId];

  elEpisode.textContent = String(game.episode);
  elScore.textContent = String(Math.floor(game.score));
  elBest.textContent = String(Math.max(agent.bestScore, Math.floor(game.score)));

  const epsPct = `${(agent.epsilon * 100).toFixed(1)}%`;
  elEpsilon.textContent = epsPct;
  if (document.activeElement !== sliderEpsilon) {
    sliderEpsilon.value = agent.epsilon.toFixed(2);
  }
  valEpsilon.textContent = agent.epsilon.toFixed(2);

  elHudState.textContent = stateObj ? stateObj.label : perception.stateId;
  elHudDist.textContent =
    perception.distancePx === null ? 'Sem obstáculo próximo' : `${perception.distancePx} px`;
  elHudObs.textContent =
    perception.obstacleType === 'ALTO'
      ? 'Cacto Alto / Duplo'
      : perception.obstacleType === 'BAIXO'
      ? 'Cacto Baixo'
      : 'Nenhum próximo';

  const act = game.lastDecision.action;
  elHudAction.textContent = act === ACTIONS.JUMP ? '↑ Saltar' : '→ Correr';

  const modeLabels = {
    explore: 'Exploração aleatória (ε)',
    exploit: 'Explotação (melhor valor Q)',
    tie: 'Empate inicial (Q = 0)',
    manual: 'Controle Manual',
  };
  elHudMode.textContent = modeLabels[game.lastDecision.mode] ?? 'Agente';

  renderQTable(perception.stateId, updatedTrace);
  if (updatedTrace) {
    renderBellmanTrace(updatedTrace);
  }
}

// Animation Loop
function loop() {
  if (isRunning) {
    let latestTrace = null;
    let episodeEnded = false;

    for (let i = 0; i < simSpeed; i++) {
      const res = game.stepFrame();
      if (res.updatedTrace) latestTrace = res.updatedTrace;
      if (res.episodeEnded) episodeEnded = true;
    }

    if (episodeEnded) {
      chart.setHistory(agent.episodeHistory);
    }

    syncTelemetryUI(latestTrace);
  }

  game.render(dinoCtx);
  requestAnimationFrame(loop);
}

// Event Listeners
btnToggleRun.addEventListener('click', () => {
  isRunning = !isRunning;
  btnToggleRun.textContent = isRunning ? 'Pausar' : 'Iniciar';
});

btnStep.addEventListener('click', () => {
  if (isRunning) {
    isRunning = false;
    btnToggleRun.textContent = 'Iniciar';
  }
  const trace = game.stepDecision();
  chart.setHistory(agent.episodeHistory);
  syncTelemetryUI(trace);
  game.render(dinoCtx);
});

btnResetEp.addEventListener('click', () => {
  game.resetEpisode();
  syncTelemetryUI();
  game.render(dinoCtx);
});

btnPretrain.addEventListener('click', () => {
  const trace = game.runFastEpisodes(50);
  chart.setHistory(agent.episodeHistory);
  syncTelemetryUI(trace);
  game.render(dinoCtx);
});

btnResetQ.addEventListener('click', () => {
  agent.resetMemory();
  agent.epsilon = parseFloat(sliderEpsilon.value);
  if (agent.epsilon < 0.5) {
    agent.epsilon = 0.85;
  }
  game.episode = 1;
  game.resetEpisode();
  chart.setHistory([]);
  renderBellmanTrace(null);
  syncTelemetryUI();
  game.render(dinoCtx);
});

chkSensors.addEventListener('change', () => {
  game.showSensors = chkSensors.checked;
  game.render(dinoCtx);
});

modeButtons.forEach((btn) => {
  btn.addEventListener('click', () => {
    const newMode = btn.dataset.mode;
    game.mode = newMode;
    modeButtons.forEach((b) => b.classList.toggle('active', b === btn));
    game.resetEpisode();
    syncTelemetryUI();
    game.render(dinoCtx);
  });
});

speedButtons.forEach((btn) => {
  btn.addEventListener('click', () => {
    simSpeed = parseInt(btn.dataset.speed, 10);
    speedButtons.forEach((b) => b.classList.toggle('active', b === btn));
  });
});

sliderAlpha.addEventListener('input', () => {
  const v = parseFloat(sliderAlpha.value);
  agent.alpha = v;
  valAlpha.textContent = v.toFixed(2);
});

sliderGamma.addEventListener('input', () => {
  const v = parseFloat(sliderGamma.value);
  agent.gamma = v;
  valGamma.textContent = v.toFixed(2);
});

sliderEpsilon.addEventListener('input', () => {
  const v = parseFloat(sliderEpsilon.value);
  agent.epsilon = v;
  valEpsilon.textContent = v.toFixed(2);
  elEpsilon.textContent = `${(v * 100).toFixed(1)}%`;
});

chkDecay.addEventListener('change', () => {
  agent.autoDecay = chkDecay.checked;
});

// Keyboard & canvas tap for Manual Mode
window.addEventListener('keydown', (e) => {
  if (e.code === 'Space' || e.code === 'ArrowUp') {
    if (game.mode === 'manual') {
      e.preventDefault();
      game.requestManualJump();
    }
  }
});

dinoCanvas.addEventListener('click', () => {
  if (game.mode === 'manual') {
    game.requestManualJump();
  }
});

// Initialize UI
buildQTableDOM();
buildStateCardsDOM();
chart.setHistory(agent.episodeHistory);
renderBellmanTrace(null);
syncTelemetryUI();
game.render(dinoCtx);
requestAnimationFrame(loop);
