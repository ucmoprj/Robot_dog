#!/usr/bin/env python3
"""Reads training logs and shows graphs with collapsible English explanations.

Unlike TensorBoard, this attaches a "what does this mean" / "which direction
is good" collapsible block directly below each chart. Uses only the stdlib
(http.server) plus tensorboard's event reader, no new dependencies.

Usage:
    python train/dashboard_server.py --logdir train/logs/live_demo --port 6007
Then open http://<host>:6007/ in a browser. Auto-refreshes every few seconds.
"""
from __future__ import annotations

import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# tag -> {group, title, desc(what it means), good(which direction is good)}
GRAPH_META: dict[str, dict[str, str]] = {
    "Train/mean_reward": {
        "title": "Mean Episode Reward",
        "desc": "Average total reward over the most recently completed episodes (up to 100). The single most important signal for whether the policy is improving.",
        "good": "Should go UP (upward trend)",
    },
    "Train/mean_episode_length": {
        "title": "Mean Episode Length",
        "desc": "Average number of control steps survived before falling or timing out.",
        "good": "Should go UP (toward the task's max episode length)",
    },
    "State/fall_rate": {
        "title": "Fall Rate",
        "desc": "Fraction of all parallel environments (e.g. 2048) that were in a fallen state at that step (0.0-1.0).",
        "good": "Should go DOWN, toward 0",
    },
    "Reward/upright": {
        "title": "Upright Reward",
        "desc": "1.0 = perfectly upright, 0 = lying on its side. Raw value before the task weight is applied.",
        "good": "Should climb toward 1.0",
    },
    "Reward/height": {
        "title": "Height Reward",
        "desc": "How close the body height is to the target (0.1196 m). 1.0 = exactly at target height.",
        "good": "Should climb toward 1.0",
    },
    "Reward/ang_vel_penalty": {
        "title": "Angular Velocity Penalty",
        "desc": "Penalty that grows (more negative) the more the body wobbles (gyro magnitude). Always <= 0.",
        "good": "Should move toward 0 (less negative)",
    },
    "Reward/action_rate_penalty": {
        "title": "Action Rate Penalty",
        "desc": "Penalty for commanded joint angles changing abruptly between steps. Protects the servos.",
        "good": "Should move toward 0 (less negative)",
    },
    "Reward/torque_penalty": {
        "title": "Torque Penalty",
        "desc": "How much actuator effort is used relative to the 0.06 N*m servo torque budget.",
        "good": "Should move toward 0 (less negative, more efficient movement)",
    },
    "Reward/alive": {
        "title": "Alive Bonus",
        "desc": "Fixed +1.0 bonus given every step the robot has not fallen.",
        "good": "Always fixed at 1.0 - not informative as a trend (reference only)",
    },
    "State/height": {
        "title": "Body Height (m)",
        "desc": "The robot body's actual z-axis height in the simulation.",
        "good": "Should converge near the target of 0.1196 m",
    },
    "State/upright": {
        "title": "Upright (raw value)",
        "desc": "Same value as Reward/upright (1.0 = fully upright), shown as a physical quantity rather than a reward.",
        "good": "Should climb toward 1.0",
    },
    "State/lin_vel_x": {
        "title": "Forward Velocity (m/s)",
        "desc": "The body's forward/backward velocity. The 'stand' task has no target velocity.",
        "good": "Should hover near 0 for the stand task (staying in place is the goal)",
    },
    "Loss/value": {
        "title": "Value Loss",
        "desc": "Error of the critic (value function) in predicting future rewards.",
        "good": "Should decrease or stabilize flat",
    },
    "Loss/surrogate": {
        "title": "Surrogate (Policy) Loss",
        "desc": "PPO's policy-gradient loss. Naturally noisy by design.",
        "good": "The trend matters more than the absolute value - fine as long as it doesn't diverge (blow up)",
    },
    "Loss/entropy": {
        "title": "Entropy (exploration level)",
        "desc": "How random the policy's action distribution still is. Higher = more exploratory/random, lower = more confident/deterministic.",
        "good": "Starts high, should slowly decrease as training progresses",
    },
    "Loss/learning_rate": {
        "title": "Learning Rate",
        "desc": "PPO's adaptive learning rate, auto-tuned based on KL divergence.",
        "good": "No fixed direction - fine as long as it isn't wildly oscillating",
    },
    "Policy/mean_std": {
        "title": "Action Std (exploration noise)",
        "desc": "Average standard deviation of the Gaussian action distribution - how much randomness is mixed into commanded joint angles.",
        "good": "Should decrease over training (policy becomes more confident/deterministic)",
    },
    "Perf/total_fps": {
        "title": "Simulated Steps per Second",
        "desc": "How many environment-steps per second the GPU is processing.",
        "good": "Higher = faster, but unrelated to whether the training result is good or bad",
    },
}

_lock = threading.Lock()
_cache: dict = {"loaded_at": 0.0, "series": {}}
_CACHE_TTL_S = 2.0


def _load_series(logdir: str) -> dict:
    now = time.time()
    with _lock:
        if now - _cache["loaded_at"] < _CACHE_TTL_S and _cache["series"]:
            return _cache["series"]
        ea = EventAccumulator(logdir, size_guidance={"scalars": 0})
        ea.Reload()
        series = {}
        for tag in ea.Tags().get("scalars", []):
            events = ea.Scalars(tag)
            series[tag] = [[e.step, e.value] for e in events]
        _cache["loaded_at"] = now
        _cache["series"] = series
        return series


PAGE_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>MG90S Dog - Training Dashboard</title>
<style>
  body { font-family: -apple-system, Arial, sans-serif; background:#111318; color:#e6e6e6; margin:0; padding:20px; }
  h1 { font-size:20px; }
  h2 { font-size:15px; color:#8fb8ff; margin-top:32px; border-bottom:1px solid #333; padding-bottom:4px; }
  .status { color:#9aa0a6; font-size:13px; margin-bottom:16px; }
  .grid { display:grid; grid-template-columns: repeat(auto-fill, minmax(360px,1fr)); gap:16px; }
  .card { background:#1b1e26; border:1px solid #2a2e38; border-radius:8px; padding:12px; }
  .card h3 { margin:0 0 6px 0; font-size:14px; }
  .tag { color:#6c7280; font-size:11px; font-family:monospace; }
  canvas { width:100%; height:120px; display:block; background:#0d0f14; border-radius:4px; }
  .latest { float:right; font-family:monospace; color:#7CFC9E; }
  details { margin-top:8px; font-size:13px; color:#c7cad1; }
  summary { cursor:pointer; color:#8fb8ff; padding:4px 8px; display:inline-block; border:1px solid #3a5a8a; border-radius:4px; background:#1a2436; }
  summary:hover { background:#243349; }
  .good { color:#ffd479; }
</style>
</head>
<body>
  <h1>MG90S Dog - Training Dashboard</h1>
  <div class="status" id="status">Loading...</div>
  <div id="root"></div>

<script>
const META = __GRAPH_META__;

function drawChart(canvas, points) {
  const ctx = canvas.getContext('2d');
  const w = canvas.width = canvas.clientWidth * 2;
  const h = canvas.height = canvas.clientHeight * 2;
  ctx.clearRect(0,0,w,h);
  if (!points || points.length === 0) return;
  const ys = points.map(p => p[1]);
  let minY = Math.min(...ys), maxY = Math.max(...ys);
  if (minY === maxY) { minY -= 1; maxY += 1; }
  const pad = (maxY - minY) * 0.12;
  minY -= pad; maxY += pad;
  const n = points.length;
  ctx.strokeStyle = '#4f8ef7';
  ctx.lineWidth = 3;
  ctx.beginPath();
  points.forEach((p, i) => {
    const x = (i/(Math.max(n-1,1))) * (w-16) + 8;
    const y = h - ((p[1]-minY)/(maxY-minY)) * (h-24) - 12;
    if (i===0) ctx.moveTo(x,y); else ctx.lineTo(x,y);
  });
  ctx.stroke();
}

function render(series) {
  // Group and sort exactly like TensorBoard's default SCALARS tab does:
  // the group is the tag name before the first "/", group names are sorted
  // alphabetically, and tags within a group are sorted alphabetically too.
  const root = document.getElementById('root');
  const groups = {};
  for (const tag in META) {
    const g = tag.split('/')[0];
    (groups[g] = groups[g] || []).push(tag);
  }
  const groupNames = Object.keys(groups).sort();
  let html = '';
  for (const g of groupNames) {
    groups[g].sort();
    html += `<h2>${g}</h2><div class="grid">`;
    for (const tag of groups[g]) {
      const meta = META[tag];
      const pts = series[tag] || [];
      const latest = pts.length ? pts[pts.length-1][1].toFixed(4) : '-';
      html += `
        <div class="card">
          <h3>${meta.title} <span class="latest">${latest}</span></h3>
          <div class="tag">${tag}</div>
          <canvas id="c_${tag.replace(/[^a-zA-Z0-9]/g,'_')}"></canvas>
          <details>
            <summary>+ Show / hide explanation</summary>
            <p>${meta.desc}</p>
            <p class="good">Direction: ${meta.good}</p>
          </details>
        </div>`;
    }
    html += `</div>`;
  }
  root.innerHTML = html;
  for (const tag in META) {
    const c = document.getElementById('c_' + tag.replace(/[^a-zA-Z0-9]/g,'_'));
    if (c) drawChart(c, series[tag]);
  }
}

async function tick() {
  try {
    const res = await fetch('/data.json');
    const data = await res.json();
    document.getElementById('status').innerText =
      `Last updated: ${new Date().toLocaleTimeString()}  |  Iterations recorded: ${data.max_len}`;
    render(data.series);
  } catch (e) {
    document.getElementById('status').innerText = 'Could not load data yet (if training just started, please wait): ' + e;
  }
}
tick();
setInterval(tick, 3000);
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    logdir = "train/logs/live_demo"

    def log_message(self, fmt, *args):
        pass  # avoid console spam

    def do_GET(self):
        if self.path.startswith("/data.json"):
            series = _load_series(self.logdir)
            max_len = max((len(v) for v in series.values()), default=0)
            body = json.dumps({"series": series, "max_len": max_len}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            body = PAGE_TEMPLATE.replace("__GRAPH_META__", json.dumps(GRAPH_META, ensure_ascii=False)).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", default="train/logs/live_demo")
    parser.add_argument("--port", type=int, default=6007)
    args = parser.parse_args()

    Handler.logdir = args.logdir
    server = ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
    print(f"Dashboard: http://0.0.0.0:{args.port}/  (logdir={args.logdir})")
    server.serve_forever()


if __name__ == "__main__":
    main()
