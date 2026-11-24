export class LearningChart {
  constructor(canvas, tooltipEl) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.tooltipEl = tooltipEl;
    this.history = [];
    this.points = [];

    this._bindHover();
  }

  _bindHover() {
    this.canvas.addEventListener('mousemove', (e) => {
      if (this.points.length === 0) {
        this.tooltipEl.classList.add('hidden');
        return;
      }

      const rect = this.canvas.getBoundingClientRect();
      const mouseX = ((e.clientX - rect.left) / rect.width) * this.canvas.width;

      let closest = null;
      let minDist = Infinity;
      for (const pt of this.points) {
        const dist = Math.abs(pt.x - mouseX);
        if (dist < minDist) {
          minDist = dist;
          closest = pt;
        }
      }

      if (closest && minDist < 28) {
        this.tooltipEl.textContent = `Ep ${closest.episode} · Score ${closest.score} · Média ${closest.avg.toFixed(1)}`;
        this.tooltipEl.style.left = `${(closest.x / this.canvas.width) * 100}%`;
        this.tooltipEl.style.top = `${(closest.y / this.canvas.height) * 100}%`;
        this.tooltipEl.classList.remove('hidden');
      } else {
        this.tooltipEl.classList.add('hidden');
      }
    });

    this.canvas.addEventListener('mouseleave', () => {
      this.tooltipEl.classList.add('hidden');
    });
  }

  setHistory(history) {
    this.history = history;
    this.render();
  }

  render() {
    const ctx = this.ctx;
    const W = this.canvas.width;
    const H = this.canvas.height;
    const pad = { top: 16, right: 16, bottom: 28, left: 46 };
    const plotW = W - pad.left - pad.right;
    const plotH = H - pad.top - pad.bottom;

    ctx.clearRect(0, 0, W, H);
    this.points = [];

    // Empty state
    if (this.history.length === 0) {
      ctx.fillStyle = '#f8fafc';
      ctx.fillRect(pad.left, pad.top, plotW, plotH);
      ctx.strokeStyle = '#e2e8f0';
      ctx.strokeRect(pad.left, pad.top, plotW, plotH);

      ctx.fillStyle = '#64748b';
      ctx.font = '500 13px "Plus Jakarta Sans", sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText(
        'Conclua o primeiro episódio para visualizar a curva de aprendizado.',
        W / 2,
        H / 2 + 4
      );
      ctx.textAlign = 'left';
      return;
    }

    const maxScore = Math.max(100, ...this.history.map((h) => h.score));
    const yMax = Math.ceil(maxScore / 50) * 50;

    // Gridlines & Y labels
    const ticks = 4;
    ctx.font = '500 11px "IBM Plex Mono", monospace';
    ctx.textAlign = 'right';
    for (let i = 0; i <= ticks; i++) {
      const val = Math.round((yMax / ticks) * i);
      const y = pad.top + plotH - (val / yMax) * plotH;

      ctx.strokeStyle = i === 0 ? '#cbd5e1' : '#f1f5f9';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(pad.left, y);
      ctx.lineTo(pad.left + plotW, y);
      ctx.stroke();

      ctx.fillStyle = '#64748b';
      ctx.fillText(String(val), pad.left - 8, y + 4);
    }

    const n = this.history.length;
    const xForIndex = (idx) => {
      if (n === 1) return pad.left + plotW / 2;
      return pad.left + (idx / (n - 1)) * plotW;
    };
    const yForScore = (s) => pad.top + plotH - (s / yMax) * plotH;

    // Compute moving average (window 10)
    const movingAvg = this.history.map((_, idx) => {
      const start = Math.max(0, idx - 9);
      let sum = 0;
      for (let j = start; j <= idx; j++) sum += this.history[j].score;
      return sum / (idx - start + 1);
    });

    // Draw raw episode score line
    ctx.strokeStyle = '#cbd5e1';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    this.history.forEach((h, idx) => {
      const x = xForIndex(idx);
      const y = yForScore(h.score);
      if (idx === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);

      this.points.push({
        x,
        y,
        episode: h.episode,
        score: h.score,
        avg: movingAvg[idx],
      });
    });
    ctx.stroke();

    // Draw episode dots if <= 60 episodes
    if (n <= 60) {
      ctx.fillStyle = '#94a3b8';
      for (const pt of this.points) {
        ctx.beginPath();
        ctx.arc(pt.x, pt.y, 2.5, 0, Math.PI * 2);
        ctx.fill();
      }
    }

    // Draw moving average line
    ctx.strokeStyle = '#0284c7';
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    movingAvg.forEach((avg, idx) => {
      const x = xForIndex(idx);
      const y = yForScore(avg);
      if (idx === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();

    // X-axis labels
    ctx.fillStyle = '#64748b';
    ctx.textAlign = 'center';
    ctx.fillText('Ep 1', pad.left + 12, H - 8);
    if (n > 2) {
      ctx.fillText(`Ep ${Math.ceil(n / 2)}`, pad.left + plotW / 2, H - 8);
    }
    if (n > 1) {
      ctx.fillText(`Ep ${n}`, pad.left + plotW - 12, H - 8);
    }
    ctx.textAlign = 'left';
  }
}
