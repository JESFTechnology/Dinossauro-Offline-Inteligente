import { ACTIONS, ACTION_LABELS, STATE_MAP, discretizeState } from './qlearning.js';

export const CANVAS_WIDTH = 760;
export const CANVAS_HEIGHT = 270;
export const GROUND_Y = 222;

const DINO_X = 82;
const DINO_W = 36;
const DINO_H = 44;
const SPEED = 6.4;
const JUMP_VY = -12.8;
const GRAVITY = 0.66;

const REWARDS = {
  SAFE_STEP: 0.2,
  JUMP_COST: -2.0,
  PASS_OBSTACLE: 20.0,
  CRASH: -100.0,
};

export class DinoGame {
  constructor(agent) {
    this.agent = agent;
    this.mode = 'agent'; // 'agent' | 'manual'
    this.showSensors = true;

    this.tick = 0;
    this.episode = 1;
    this.score = 0;
    this.episodeReward = 0;

    this.dino = {
      x: DINO_X,
      y: GROUND_Y - DINO_H,
      w: DINO_W,
      h: DINO_H,
      vy: 0,
      grounded: true,
    };

    this.obstacles = [];
    this.groundDots = this._createGroundDecor();
    this.clouds = [
      { x: 180, y: 48, w: 46 },
      { x: 460, y: 34, w: 58 },
      { x: 670, y: 62, w: 40 },
    ];

    this.crashed = false;
    this.crashFlashFrames = 0;

    // Decision tracking for Semi-MDP Q-Learning
    this.pendingJump = null; // { startState, rewardAcc }
    this.pendingRun = null; // { startState, rewardAcc, frames }
    this.lastDecision = {
      stateId: 'LIVRE',
      action: ACTIONS.RUN,
      mode: 'exploit',
      distancePx: null,
      obstacleType: null,
    };

    this.manualJumpQueued = false;
    this.resetEpisode();
  }

  _createGroundDecor() {
    const dots = [];
    for (let i = 0; i < 36; i++) {
      dots.push({
        x: i * 24 + (i % 3) * 7,
        y: GROUND_Y + 6 + (i % 4) * 5,
        w: (i % 3) + 2,
      });
    }
    return dots;
  }

  resetEpisode() {
    this.tick = 0;
    this.score = 0;
    this.episodeReward = 0;
    this.crashed = false;
    this.crashFlashFrames = 0;

    this.dino.y = GROUND_Y - DINO_H;
    this.dino.vy = 0;
    this.dino.grounded = true;

    this.obstacles = [];
    this._spawnObstacle(640);

    this.pendingJump = null;
    this.pendingRun = null;
    this.manualJumpQueued = false;

    const perception = this.getPerception();
    this.lastDecision = {
      stateId: perception.stateId,
      action: ACTIONS.RUN,
      mode: this.mode === 'manual' ? 'manual' : 'exploit',
      distancePx: perception.distancePx,
      obstacleType: perception.obstacleType,
    };
  }

  _spawnObstacle(startX) {
    const isTall = Math.random() < 0.48;
    const width = isTall ? 44 : 24;
    const height = isTall ? 58 : 40;
    this.obstacles.push({
      x: startX,
      y: GROUND_Y - height,
      w: width,
      h: height,
      type: isTall ? 'ALTO' : 'BAIXO',
      passed: false,
    });
  }

  requestManualJump() {
    this.manualJumpQueued = true;
  }

  getNextObstacle() {
    const dinoFront = this.dino.x + this.dino.w;
    for (const obs of this.obstacles) {
      if (obs.x + obs.w >= this.dino.x + 6) {
        return obs;
      }
    }
    return null;
  }

  getPerception() {
    const nextObs = this.getNextObstacle();
    if (!nextObs) {
      return {
        stateId: 'LIVRE',
        distancePx: null,
        obstacleType: null,
        obstacle: null,
      };
    }

    const dinoFront = this.dino.x + this.dino.w;
    const distancePx = Math.max(0, Math.round(nextObs.x - dinoFront));
    const stateId = discretizeState(distancePx, nextObs.type);

    return {
      stateId,
      distancePx,
      obstacleType: nextObs.type,
      obstacle: nextObs,
    };
  }

  _checkCollision() {
    // Slightly inset hitbox for crisp fairness
    const dx = this.dino.x + 6;
    const dy = this.dino.y + 4;
    const dw = this.dino.w - 12;
    const dh = this.dino.h - 8;

    for (const obs of this.obstacles) {
      const ox = obs.x + 3;
      const oy = obs.y + 3;
      const ow = obs.w - 6;
      const oh = obs.h - 4;

      if (dx < ox + ow && dx + dw > ox && dy < oy + oh && dy + dh > oy) {
        return true;
      }
    }
    return false;
  }

  /**
   * Advances simulation by 1 physics frame (60Hz baseline).
   * Returns event flags for UI/Bellman updates.
   */
  stepFrame() {
    if (this.crashed) {
      this.crashFlashFrames += 1;
      if (this.crashFlashFrames >= 16) {
        this.episode = this.agent.finishEpisode(Math.floor(this.score), this.episodeReward) + 1;
        this.resetEpisode();
        return { episodeEnded: true, updatedTrace: null };
      }
      return { episodeEnded: false, updatedTrace: null };
    }

    this.tick += 1;
    this.score += 0.25;
    let updatedTrace = null;

    // 1. If grounded and no active run/jump decision, or run window elapsed, make a decision
    if (this.dino.grounded) {
      const perception = this.getPerception();

      if (this.mode === 'manual') {
        if (this.manualJumpQueued) {
          this.dino.vy = JUMP_VY;
          this.dino.grounded = false;
          this.manualJumpQueued = false;
          this.lastDecision = {
            stateId: perception.stateId,
            action: ACTIONS.JUMP,
            mode: 'manual',
            distancePx: perception.distancePx,
            obstacleType: perception.obstacleType,
          };
        } else {
          this.lastDecision = {
            stateId: perception.stateId,
            action: ACTIONS.RUN,
            mode: 'manual',
            distancePx: perception.distancePx,
            obstacleType: perception.obstacleType,
          };
        }
      } else {
        // Agent mode
        const stateChanged =
          this.pendingRun && this.pendingRun.startState !== perception.stateId;
        const runIntervalComplete =
          this.pendingRun && this.pendingRun.frames >= 6;

        if (this.pendingRun && (stateChanged || runIntervalComplete)) {
          updatedTrace = this.agent.update(
            this.pendingRun.startState,
            ACTIONS.RUN,
            this.pendingRun.rewardAcc,
            perception.stateId,
            false,
            this.tick
          );
          this.pendingRun = null;
        }

        if (!this.pendingRun && !this.pendingJump) {
          const decision = this.agent.chooseAction(perception.stateId);
          this.lastDecision = {
            stateId: perception.stateId,
            action: decision.action,
            mode: decision.mode,
            distancePx: perception.distancePx,
            obstacleType: perception.obstacleType,
          };

          if (decision.action === ACTIONS.JUMP) {
            this.dino.vy = JUMP_VY;
            this.dino.grounded = false;
            this.pendingJump = {
              startState: perception.stateId,
              rewardAcc: REWARDS.JUMP_COST,
            };
            this.episodeReward += REWARDS.JUMP_COST;
          } else {
            this.pendingRun = {
              startState: perception.stateId,
              rewardAcc: 0,
              frames: 0,
            };
          }
        }
      }
    }

    // 2. Physics update
    if (!this.dino.grounded) {
      this.dino.vy += GRAVITY;
      this.dino.y += this.dino.vy;

      if (this.dino.y >= GROUND_Y - DINO_H) {
        this.dino.y = GROUND_Y - DINO_H;
        this.dino.vy = 0;
        this.dino.grounded = true;
      }
    }

    // 3. Move obstacles & ground decor
    let passedCount = 0;
    for (const obs of this.obstacles) {
      obs.x -= SPEED;
      if (!obs.passed && obs.x + obs.w < this.dino.x) {
        obs.passed = true;
        passedCount += 1;
        this.score += 10;
      }
    }

    // Remove off-screen obstacles
    if (this.obstacles.length > 0 && this.obstacles[0].x + this.obstacles[0].w < -40) {
      this.obstacles.shift();
    }

    // Spawn next obstacle with fair gap (340–480 px)
    const lastObs = this.obstacles[this.obstacles.length - 1];
    if (!lastObs || CANVAS_WIDTH - (lastObs.x + lastObs.w) >= 340 + Math.random() * 140) {
      if (!lastObs || lastObs.x < CANVAS_WIDTH - 340) {
        this._spawnObstacle(CANVAS_WIDTH + 30 + Math.random() * 60);
      }
    }

    for (const dot of this.groundDots) {
      dot.x -= SPEED;
      if (dot.x < -10) dot.x += 36 * 24;
    }

    for (const cloud of this.clouds) {
      cloud.x -= 0.7;
      if (cloud.x + cloud.w < -20) cloud.x = CANVAS_WIDTH + 30;
    }

    // 4. Accumulate step & pass rewards
    if (passedCount > 0) {
      const bonus = passedCount * REWARDS.PASS_OBSTACLE;
      this.episodeReward += bonus;
      if (this.pendingJump) {
        this.pendingJump.rewardAcc += bonus;
      } else if (this.pendingRun) {
        this.pendingRun.rewardAcc += bonus;
      }
    }

    if (this.pendingRun) {
      this.pendingRun.frames += 1;
      const stepR = REWARDS.SAFE_STEP / 6;
      this.pendingRun.rewardAcc += stepR;
      this.episodeReward += stepR;
    }

    // 5. Check collision
    if (this._checkCollision()) {
      this.crashed = true;
      this.crashFlashFrames = 0;
      this.episodeReward += REWARDS.CRASH;

      if (this.mode === 'agent') {
        if (this.pendingJump) {
          updatedTrace = this.agent.update(
            this.pendingJump.startState,
            ACTIONS.JUMP,
            this.pendingJump.rewardAcc + REWARDS.CRASH,
            null,
            true,
            this.tick
          );
          this.pendingJump = null;
        } else if (this.pendingRun) {
          updatedTrace = this.agent.update(
            this.pendingRun.startState,
            ACTIONS.RUN,
            this.pendingRun.rewardAcc + REWARDS.CRASH,
            null,
            true,
            this.tick
          );
          this.pendingRun = null;
        }
      }
      return { episodeEnded: false, crashedNow: true, updatedTrace };
    }

    // 6. If jump just landed safely, close the jump macro-action and update Q
    if (this.dino.grounded && this.pendingJump && this.mode === 'agent') {
      const landingPerception = this.getPerception();
      updatedTrace = this.agent.update(
        this.pendingJump.startState,
        ACTIONS.JUMP,
        this.pendingJump.rewardAcc,
        landingPerception.stateId,
        false,
        this.tick
      );
      this.pendingJump = null;
    }

    return { episodeEnded: false, updatedTrace };
  }

  /**
   * Advances exactly one Q-learning decision (for the "Passo" button).
   */
  stepDecision() {
    if (this.crashed) {
      this.episode = this.agent.finishEpisode(Math.floor(this.score), this.episodeReward) + 1;
      this.resetEpisode();
      return null;
    }

    let safetyCounter = 0;
    let trace = null;
    while (safetyCounter < 90) {
      safetyCounter += 1;
      const res = this.stepFrame();
      if (res.updatedTrace) {
        trace = res.updatedTrace;
        break;
      }
      if (res.episodeEnded || res.crashedNow) {
        break;
      }
    }
    return trace;
  }

  /**
   * Runs N complete episodes instantaneously (used for "+50 Episódios").
   */
  runFastEpisodes(count) {
    const prevMode = this.mode;
    this.mode = 'agent';
    let lastTrace = null;

    for (let i = 0; i < count; i++) {
      this.resetEpisode();
      let ticks = 0;
      // Cap single episode at 4500 ticks (~75s game time) so a master agent doesn't loop forever
      while (!this.crashed && ticks < 4500) {
        ticks += 1;
        const res = this.stepFrame();
        if (res.updatedTrace) {
          lastTrace = res.updatedTrace;
        }
      }
      this.episode = this.agent.finishEpisode(Math.floor(this.score), this.episodeReward) + 1;
    }

    this.mode = prevMode;
    this.resetEpisode();
    return lastTrace;
  }

  render(ctx) {
    ctx.clearRect(0, 0, CANVAS_WIDTH, CANVAS_HEIGHT);

    // Subtle background
    ctx.fillStyle = this.crashed ? '#fef2f2' : '#fcfdfe';
    ctx.fillRect(0, 0, CANVAS_WIDTH, CANVAS_HEIGHT);

    // Clouds
    ctx.fillStyle = '#e2e8f0';
    for (const c of this.clouds) {
      ctx.beginPath();
      ctx.roundRect(c.x, c.y, c.w, 12, 6);
      ctx.roundRect(c.x + 8, c.y - 6, c.w - 16, 10, 5);
      ctx.fill();
    }

    // Distance perception zones (drawn behind obstacles & dino)
    const perception = this.getPerception();
    if (this.showSensors) {
      this._renderPerceptionZones(ctx, perception);
    }

    // Ground horizon line
    ctx.strokeStyle = '#cbd5e1';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(0, GROUND_Y);
    ctx.lineTo(CANVAS_WIDTH, GROUND_Y);
    ctx.stroke();

    // Ground texture ticks
    ctx.fillStyle = '#cbd5e1';
    for (const dot of this.groundDots) {
      ctx.fillRect(dot.x, dot.y, dot.w, 2);
    }

    // Obstacles (Cacti)
    for (const obs of this.obstacles) {
      this._renderCactus(ctx, obs);
    }

    // Dinosaur
    this._renderDino(ctx);

    // Sensor beam to next obstacle
    if (this.showSensors && perception.obstacle) {
      this._renderSensorBeam(ctx, perception);
    }

    // Top-right in-canvas status readout (unboxed clean mono)
    ctx.fillStyle = '#475569';
    ctx.font = '600 12px "IBM Plex Mono", monospace';
    ctx.textAlign = 'right';
    const actionName = ACTION_LABELS[this.lastDecision.action];
    const modeName =
      this.mode === 'manual'
        ? 'Manual'
        : this.lastDecision.mode === 'explore'
        ? 'Exploração (ε)'
        : this.lastDecision.mode === 'tie'
        ? 'Empate Q=0'
        : 'Explotação (max Q)';
    ctx.fillText(
      `Ação: ${actionName} · ${modeName}`,
      CANVAS_WIDTH - 16,
      24
    );
    ctx.textAlign = 'left';
  }

  _renderPerceptionZones(ctx, perception) {
    const dinoFront = this.dino.x + this.dino.w;
    const zones = [
      { label: 'IMEDIATA', max: 50, color: 'rgba(220, 38, 38, 0.06)', line: '#fca5a5' },
      { label: 'PERTO', min: 50, max: 110, color: 'rgba(2, 132, 199, 0.06)', line: '#7dd3fc' },
      { label: 'MÉDIA', min: 110, max: 165, color: 'rgba(217, 119, 6, 0.04)', line: '#fcd34d' },
      { label: 'LONGE', min: 165, max: 250, color: 'rgba(100, 116, 139, 0.03)', line: '#cbd5e1' },
    ];

    for (const z of zones) {
      const startX = dinoFront + (z.min ?? 0);
      const width = z.max - (z.min ?? 0);
      ctx.fillStyle = z.color;
      ctx.fillRect(startX, 34, width, GROUND_Y - 34);

      ctx.strokeStyle = z.line;
      ctx.lineWidth = 1;
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      ctx.moveTo(dinoFront + z.max, 36);
      ctx.lineTo(dinoFront + z.max, GROUND_Y);
      ctx.stroke();
      ctx.setLineDash([]);

      ctx.fillStyle = '#64748b';
      ctx.font = '500 10px "IBM Plex Mono", monospace';
      ctx.fillText(z.label, startX + 5, 50);
    }

    // Highlight active state label on canvas top-left
    const stateObj = STATE_MAP[perception.stateId];
    ctx.fillStyle = '#0f172a';
    ctx.font = '600 12px "IBM Plex Mono", monospace';
    ctx.fillText(
      `Estado: ${stateObj ? stateObj.label : perception.stateId}`,
      16,
      24
    );
  }

  _renderSensorBeam(ctx, perception) {
    const obs = perception.obstacle;
    const startX = this.dino.x + this.dino.w;
    const endX = Math.max(startX, obs.x);
    const beamY = GROUND_Y - 18;

    ctx.strokeStyle = '#0284c7';
    ctx.lineWidth = 1.5;
    ctx.setLineDash([4, 3]);
    ctx.beginPath();
    ctx.moveTo(startX, beamY);
    ctx.lineTo(endX, beamY);
    ctx.stroke();
    ctx.setLineDash([]);

    if (endX - startX > 36) {
      ctx.fillStyle = '#0284c7';
      ctx.font = '600 11px "IBM Plex Mono", monospace';
      ctx.textAlign = 'center';
      ctx.fillText(`${perception.distancePx}px`, (startX + endX) / 2, beamY - 6);
      ctx.textAlign = 'left';
    }
  }

  _renderCactus(ctx, obs) {
    ctx.fillStyle = '#0f766e';
    const { x, y, w, h, type } = obs;

    if (type === 'BAIXO') {
      // Single crisp cactus
      ctx.beginPath();
      ctx.roundRect(x + 7, y, 10, h, 4);
      ctx.roundRect(x, y + 12, 7, 14, 3);
      ctx.roundRect(x + 17, y + 8, 7, 14, 3);
      ctx.fill();
    } else {
      // Tall / double cactus cluster
      ctx.beginPath();
      ctx.roundRect(x + 6, y, 11, h, 4);
      ctx.roundRect(x, y + 16, 7, 16, 3);
      ctx.roundRect(x + 16, y + 10, 7, 18, 3);
      // Second shorter stem next to it
      ctx.roundRect(x + 28, y + 14, 10, h - 14, 4);
      ctx.roundRect(x + 37, y + 22, 7, 13, 3);
      ctx.fill();
    }
  }

  _renderDino(ctx) {
    const { x, y, grounded } = this.dino;
    const isHit = this.crashed;

    ctx.fillStyle = isHit ? '#dc2626' : '#0f172a';

    // Tail
    ctx.fillRect(x, y + 18, 8, 10);
    ctx.fillRect(x + 4, y + 22, 8, 10);

    // Torso
    ctx.beginPath();
    ctx.roundRect(x + 8, y + 14, 18, 20, 4);
    ctx.fill();

    // Neck & Head
    ctx.beginPath();
    ctx.roundRect(x + 14, y, 22, 16, 4);
    ctx.fill();

    // Snout notch
    ctx.fillStyle = isHit ? '#fef2f2' : '#fcfdfe';
    ctx.fillRect(x + 28, y + 11, 8, 3);

    // Eye
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(x + 21, y + 4, 4, 4);
    if (isHit) {
      ctx.fillStyle = '#dc2626';
      ctx.fillRect(x + 22, y + 5, 2, 2);
    }

    // Little arm
    ctx.fillStyle = isHit ? '#dc2626' : '#0f172a';
    ctx.fillRect(x + 26, y + 20, 6, 4);

    // Legs (2-frame run animation when grounded)
    const stepPhase = Math.floor(this.tick / 5) % 2;
    if (!grounded) {
      ctx.fillRect(x + 11, y + 34, 5, 8);
      ctx.fillRect(x + 20, y + 34, 5, 8);
    } else if (stepPhase === 0) {
      ctx.fillRect(x + 11, y + 34, 5, 10);
      ctx.fillRect(x + 20, y + 34, 5, 6);
    } else {
      ctx.fillRect(x + 11, y + 34, 5, 6);
      ctx.fillRect(x + 20, y + 34, 5, 10);
    }
  }
}
