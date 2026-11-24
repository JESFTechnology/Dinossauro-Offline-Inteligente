export const ACTIONS = {
  RUN: 0,
  JUMP: 1,
};

export const ACTION_LABELS = {
  [ACTIONS.RUN]: 'Correr',
  [ACTIONS.JUMP]: 'Saltar',
};

/**
 * 9 discrete interpretable states combining distance to the next cactus
 * and obstacle height/type.
 */
export const STATES = [
  {
    id: 'LIVRE',
    label: 'LIVRE',
    meta: 'Distância > 250 px',
    distanceRange: '> 250 px',
    obstacleType: 'Qualquer',
    description:
      'Caminho livre à frente. Saltar agora desperdiça energia (−2) e faz o agente aterrissar antes do obstáculo.',
  },
  {
    id: 'LONGE_BAIXO',
    label: 'LONGE · Baixo',
    meta: '165–250 px · Cacto baixo',
    distanceRange: '165–250 px',
    obstacleType: 'Cacto Baixo',
    description:
      'Cacto baixo ainda distante. Correr aproxima o dinossauro da janela ideal de salto.',
  },
  {
    id: 'LONGE_ALTO',
    label: 'LONGE · Alto',
    meta: '165–250 px · Cacto alto',
    distanceRange: '165–250 px',
    obstacleType: 'Cacto Alto / Duplo',
    description:
      'Cacto alto ainda distante. Saltar cedo demais pode fazer o agente cair exatamente em cima dele.',
  },
  {
    id: 'MEDIA_BAIXO',
    label: 'MÉDIA · Baixo',
    meta: '110–165 px · Cacto baixo',
    distanceRange: '110–165 px',
    obstacleType: 'Cacto Baixo',
    description:
      'Distância intermediária de um cacto baixo. Correr mais um pouco ou saltar já começa a ser viável.',
  },
  {
    id: 'MEDIA_ALTO',
    label: 'MÉDIA · Alto',
    meta: '110–165 px · Cacto alto',
    distanceRange: '110–165 px',
    obstacleType: 'Cacto Alto / Duplo',
    description:
      'Janela antecipada ideal para obstáculos largos/altos, garantindo altura máxima sobre o cacto.',
  },
  {
    id: 'PERTO_BAIXO',
    label: 'PERTO · Baixo',
    meta: '50–110 px · Cacto baixo',
    distanceRange: '50–110 px',
    obstacleType: 'Cacto Baixo',
    description:
      'Zona crítica de salto para cacto baixo. Correr aqui leva à colisão iminente (−100).',
  },
  {
    id: 'PERTO_ALTO',
    label: 'PERTO · Alto',
    meta: '50–110 px · Cacto alto',
    distanceRange: '50–110 px',
    obstacleType: 'Cacto Alto / Duplo',
    description:
      'Zona crítica para cacto alto. Saltar supera o obstáculo (+20); correr resulta em colisão.',
  },
  {
    id: 'IMEDIATA_BAIXO',
    label: 'IMEDIATA · Baixo',
    meta: '≤ 50 px · Cacto baixo',
    distanceRange: '≤ 50 px',
    obstacleType: 'Cacto Baixo',
    description:
      'Obstáculo praticamente colado ao dinossauro. Último instante possível para tentar escapar com um salto.',
  },
  {
    id: 'IMEDIATA_ALTO',
    label: 'IMEDIATA · Alto',
    meta: '≤ 50 px · Cacto alto',
    distanceRange: '≤ 50 px',
    obstacleType: 'Cacto Alto / Duplo',
    description:
      'Perigo extremo diante de um cacto alto. O agente aprende rapidamente a evitar chegar até este estado correndo.',
  },
];

export const STATE_MAP = Object.fromEntries(STATES.map((s) => [s.id, s]));

/**
 * Maps continuous game measurements into one of the 9 discrete state IDs.
 */
export function discretizeState(distancePx, obstacleHeightType) {
  if (distancePx === null || distancePx > 250) {
    return 'LIVRE';
  }
  const suffix = obstacleHeightType === 'ALTO' ? 'ALTO' : 'BAIXO';
  if (distancePx > 165) {
    return `LONGE_${suffix}`;
  }
  if (distancePx > 110) {
    return `MEDIA_${suffix}`;
  }
  if (distancePx > 50) {
    return `PERTO_${suffix}`;
  }
  return `IMEDIATA_${suffix}`;
}

export class QLearningAgent {
  constructor(options = {}) {
    this.alpha = options.alpha ?? 0.25;
    this.gamma = options.gamma ?? 0.90;
    this.epsilon = options.epsilon ?? 0.85;
    this.epsilonDecay = options.epsilonDecay ?? 0.92;
    this.epsilonMin = options.epsilonMin ?? 0.05;
    this.autoDecay = options.autoDecay ?? true;

    this.qTable = {};
    this.lastUpdate = null;
    this.episodeHistory = [];
    this.bestScore = 0;

    this.resetMemory();
  }

  resetMemory() {
    this.qTable = {};
    for (const state of STATES) {
      this.qTable[state.id] = {
        run: 0,
        jump: 0,
        visitsRun: 0,
        visitsJump: 0,
        lastDeltaRun: 0,
        lastDeltaJump: 0,
        lastUpdateTick: 0,
      };
    }
    this.lastUpdate = null;
    this.episodeHistory = [];
    this.bestScore = 0;
  }

  getEntry(stateId) {
    return this.qTable[stateId];
  }

  getMaxQ(stateId) {
    const entry = this.qTable[stateId];
    if (!entry) return 0;
    return Math.max(entry.run, entry.jump);
  }

  getBestAction(stateId) {
    const entry = this.qTable[stateId];
    if (!entry) return ACTIONS.RUN;
    if (Math.abs(entry.run - entry.jump) < 1e-6) {
      return null; // Tie
    }
    return entry.jump > entry.run ? ACTIONS.JUMP : ACTIONS.RUN;
  }

  /**
   * Epsilon-greedy policy selection.
   */
  chooseAction(stateId) {
    const entry = this.qTable[stateId];
    const roll = Math.random();

    if (roll < this.epsilon) {
      const action = Math.random() < 0.5 ? ACTIONS.RUN : ACTIONS.JUMP;
      return {
        action,
        mode: 'explore',
        qRun: entry.run,
        qJump: entry.jump,
      };
    }

    if (Math.abs(entry.run - entry.jump) < 1e-6) {
      // Unbiased tie-break slightly favoring RUN when totally untrained
      const action = Math.random() < 0.7 ? ACTIONS.RUN : ACTIONS.JUMP;
      return {
        action,
        mode: 'tie',
        qRun: entry.run,
        qJump: entry.jump,
      };
    }

    const action = entry.jump > entry.run ? ACTIONS.JUMP : ACTIONS.RUN;
    return {
      action,
      mode: 'exploit',
      qRun: entry.run,
      qJump: entry.jump,
    };
  }

  /**
   * Performs the Bellman update:
   * Q(s, a) <- Q(s, a) + alpha * [ R + gamma * max_a' Q(s', a') - Q(s, a) ]
   */
  update(stateId, action, reward, nextStateId, isTerminal, tick = 0) {
    const entry = this.qTable[stateId];
    if (!entry) return null;

    const oldQ = action === ACTIONS.RUN ? entry.run : entry.jump;
    const maxNextQ = isTerminal ? 0 : this.getMaxQ(nextStateId);
    const target = reward + this.gamma * maxNextQ;
    const tdError = target - oldQ;
    const delta = this.alpha * tdError;
    const newQ = oldQ + delta;

    if (action === ACTIONS.RUN) {
      entry.run = newQ;
      entry.visitsRun += 1;
      entry.lastDeltaRun = delta;
    } else {
      entry.jump = newQ;
      entry.visitsJump += 1;
      entry.lastDeltaJump = delta;
    }
    entry.lastUpdateTick = tick;

    const trace = {
      stateId,
      stateLabel: STATE_MAP[stateId]?.label ?? stateId,
      action,
      actionLabel: ACTION_LABELS[action],
      oldQ,
      newQ,
      delta,
      reward,
      nextStateId: isTerminal ? 'TERMINAL' : nextStateId,
      nextStateLabel: isTerminal ? 'COLISÃO' : (STATE_MAP[nextStateId]?.label ?? nextStateId),
      maxNextQ,
      alpha: this.alpha,
      gamma: this.gamma,
      isTerminal,
    };

    this.lastUpdate = trace;
    return trace;
  }

  finishEpisode(score, totalReward) {
    if (score > this.bestScore) {
      this.bestScore = score;
    }

    const episodeNumber = this.episodeHistory.length + 1;
    this.episodeHistory.push({
      episode: episodeNumber,
      score,
      totalReward,
      epsilon: this.epsilon,
    });

    if (this.autoDecay) {
      this.epsilon = Math.max(this.epsilonMin, this.epsilon * this.epsilonDecay);
    }

    return episodeNumber;
  }
}
