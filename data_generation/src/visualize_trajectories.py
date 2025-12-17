#!/usr/bin/env python
"""
Serve a local web UI for visualizing trajectory files with inline stats.

Usage:
  python refactored_scripts/serve_trajectories.py --root data/<your_step6_dir> --port 8899

Security:
- Only serves files under the provided --root directory.
- Binds to 127.0.0.1 by default.
- IDs in API responses are indexes, so there is no path traversal surface.
"""

import argparse
import html
import json
import os
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import List
from urllib.parse import parse_qs, urlparse


MAX_CONTENT_CHARS = 200_000  # Prevent runaway memory usage in UI/API.


@dataclass
class Trajectory:
    idx: int
    path: Path
    name: str
    size_bytes: int
    lines: int
    words: int
    chars: int


def scan(root: Path) -> List[Trajectory]:
    files = sorted(p for p in root.glob("*.txt") if p.is_file())
    trajectories: List[Trajectory] = []
    for i, p in enumerate(files):
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except Exception:
            # If unreadable, skip but keep the server alive.
            continue
        size = p.stat().st_size
        lines = text.count("\n") + 1 if text else 0
        words = len(text.split())
        chars = len(text)
        trajectories.append(
            Trajectory(
                idx=i,
                path=p,
                name=p.name,
                size_bytes=size,
                lines=lines,
                words=words,
                chars=chars,
            )
        )
    return trajectories


def format_size(bytes_count: int) -> str:
    for unit in ["B", "KB", "MB", "GB"]:
        if bytes_count < 1024 or unit == "GB":
            return f"{bytes_count:.1f} {unit}"
        bytes_count /= 1024
    return f"{bytes_count:.1f} GB"


class TrajectoryHandler(BaseHTTPRequestHandler):
    def _json(self, payload, status=200):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _html(self, body: str, status=200):
        encoded = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(encoded)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/":
            return self._html(self.render_index())
        if parsed.path == "/api/trajectories":
            return self._json(self.list_trajectories())
        if parsed.path == "/api/trajectory":
            return self._json(self.get_trajectory(parsed), status=200)
        self.send_error(404, "Not Found")

    def list_trajectories(self):
        return {
            "root": str(self.server.root),
            "count": len(self.server.trajectories),
            "stats": [
                {
                    "id": t.idx,
                    "name": t.name,
                    "size_bytes": t.size_bytes,
                    "size_pretty": format_size(t.size_bytes),
                    "lines": t.lines,
                    "words": t.words,
                    "chars": t.chars,
                }
                for t in self.server.trajectories
            ],
        }

    def get_trajectory(self, parsed):
        qs = parse_qs(parsed.query or "")
        try:
            idx = int(qs.get("id", [""])[0])
        except Exception:
            return {"error": "invalid id"}
        if idx < 0 or idx >= len(self.server.trajectories):
            return {"error": "id out of range"}
        traj = self.server.trajectories[idx]
        try:
            text = traj.path.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            return {"error": f"could not read file: {e}"}
        truncated = len(text) > MAX_CONTENT_CHARS
        if truncated:
            text = text[:MAX_CONTENT_CHARS]
        return {
            "id": traj.idx,
            "name": traj.name,
            "size_bytes": traj.size_bytes,
            "size_pretty": format_size(traj.size_bytes),
            "lines": traj.lines,
            "words": traj.words,
            "chars": traj.chars,
            "content": text,
            "truncated": truncated,
        }

    def log_message(self, fmt, *args):
        # Keep output minimal.
        return

    def render_index(self) -> str:
        # All assets are inline for offline safety.
        return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>ThreadWeaver Trajectory Explorer</title>
  <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    :root {{
      --bg-primary: #0a0e1a;
      --bg-secondary: #111827;
      --bg-tertiary: #1a2332;
      --card-bg: rgba(30, 41, 59, 0.6);
      --card-hover: rgba(30, 41, 59, 0.8);
      --accent-primary: #10b981;
      --accent-secondary: #3b82f6;
      --accent-tertiary: #8b5cf6;
      --accent-warning: #f59e0b;
      --text-primary: #f1f5f9;
      --text-secondary: #94a3b8;
      --text-muted: #64748b;
      --border: rgba(148, 163, 184, 0.1);
      --border-hover: rgba(148, 163, 184, 0.2);
      --shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
      --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
      --shadow-lg: 0 20px 25px -5px rgba(0, 0, 0, 0.4);
      --shadow-glow: 0 0 20px rgba(16, 185, 129, 0.15);
      --radius-sm: 6px;
      --radius-md: 10px;
      --radius-lg: 16px;
      --transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
    }}

    * {{
      box-sizing: border-box;
      margin: 0;
      padding: 0;
    }}

    body {{
      font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
      background: linear-gradient(135deg, #0a0e1a 0%, #1a2332 100%);
      color: var(--text-primary);
      min-height: 100vh;
      padding: 24px;
      line-height: 1.6;
    }}

    .header {{
      max-width: 1600px;
      margin: 0 auto 32px;
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      flex-wrap: wrap;
      gap: 20px;
    }}

    .header-left h1 {{
      font-size: 2.5rem;
      font-weight: 700;
      background: linear-gradient(135deg, var(--accent-primary), var(--accent-secondary));
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      background-clip: text;
      margin-bottom: 8px;
      letter-spacing: -0.02em;
    }}

    .subtitle {{
      color: var(--text-secondary);
      font-size: 1rem;
    }}

    .header-right {{
      display: flex;
      gap: 12px;
      align-items: center;
    }}

    .stat-pill {{
      background: var(--card-bg);
      backdrop-filter: blur(10px);
      padding: 12px 20px;
      border-radius: var(--radius-lg);
      border: 1px solid var(--border);
      box-shadow: var(--shadow-md);
      display: flex;
      flex-direction: column;
      gap: 4px;
      min-width: 140px;
    }}

    .stat-pill .label {{
      font-size: 0.75rem;
      color: var(--text-muted);
      text-transform: uppercase;
      letter-spacing: 0.05em;
      font-weight: 600;
    }}

    .stat-pill .value {{
      font-size: 1.5rem;
      font-weight: 700;
      background: linear-gradient(135deg, var(--accent-primary), var(--accent-secondary));
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      background-clip: text;
    }}

    .container {{
      max-width: 1600px;
      margin: 0 auto;
    }}

    .stats-overview {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
      gap: 16px;
      margin-bottom: 24px;
    }}

    .stat-card {{
      background: var(--card-bg);
      backdrop-filter: blur(10px);
      padding: 20px;
      border-radius: var(--radius-md);
      border: 1px solid var(--border);
      box-shadow: var(--shadow-md);
      transition: var(--transition);
    }}

    .stat-card:hover {{
      border-color: var(--border-hover);
      transform: translateY(-2px);
      box-shadow: var(--shadow-lg);
    }}

    .stat-card .stat-label {{
      font-size: 0.875rem;
      color: var(--text-muted);
      margin-bottom: 8px;
      display: flex;
      align-items: center;
      gap: 6px;
    }}

    .stat-card .stat-value {{
      font-size: 1.75rem;
      font-weight: 700;
      color: var(--text-primary);
    }}

    .stat-card .stat-subtext {{
      font-size: 0.75rem;
      color: var(--text-secondary);
      margin-top: 4px;
    }}

    .main-grid {{
      display: grid;
      grid-template-columns: 1fr;
      gap: 24px;
    }}

    @media (min-width: 1200px) {{
      .main-grid {{
        grid-template-columns: 400px 1fr;
      }}
    }}

    .panel {{
      background: var(--card-bg);
      backdrop-filter: blur(10px);
      border: 1px solid var(--border);
      border-radius: var(--radius-lg);
      padding: 24px;
      box-shadow: var(--shadow-lg);
      transition: var(--transition);
    }}

    .panel-header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 20px;
      padding-bottom: 16px;
      border-bottom: 1px solid var(--border);
    }}

    .panel-title {{
      font-size: 1.25rem;
      font-weight: 600;
      color: var(--text-primary);
    }}

    .search-box {{
      position: relative;
      margin-bottom: 16px;
    }}

    .search-box input {{
      width: 100%;
      padding: 12px 16px 12px 40px;
      background: var(--bg-secondary);
      border: 1px solid var(--border);
      border-radius: var(--radius-md);
      color: var(--text-primary);
      font-size: 0.875rem;
      transition: var(--transition);
    }}

    .search-box input:focus {{
      outline: none;
      border-color: var(--accent-primary);
      box-shadow: 0 0 0 3px rgba(16, 185, 129, 0.1);
    }}

    .search-box::before {{
      content: "🔍";
      position: absolute;
      left: 14px;
      top: 50%;
      transform: translateY(-50%);
      opacity: 0.5;
    }}

    .traj-list {{
      max-height: 600px;
      overflow-y: auto;
      margin: -8px;
      padding: 8px;
    }}

    .traj-item {{
      padding: 12px 16px;
      margin-bottom: 8px;
      background: var(--bg-secondary);
      border: 1px solid var(--border);
      border-radius: var(--radius-sm);
      cursor: pointer;
      transition: var(--transition);
    }}

    .traj-item:hover {{
      background: var(--bg-tertiary);
      border-color: var(--accent-primary);
      transform: translateX(4px);
    }}

    .traj-item.active {{
      background: var(--bg-tertiary);
      border-color: var(--accent-primary);
      box-shadow: var(--shadow-glow);
    }}

    .traj-item .traj-name {{
      font-weight: 600;
      color: var(--text-primary);
      margin-bottom: 6px;
      font-size: 0.875rem;
    }}

    .traj-item .traj-meta {{
      display: flex;
      gap: 12px;
      font-size: 0.75rem;
      color: var(--text-muted);
    }}

    .content-panel {{
      display: flex;
      flex-direction: column;
      min-height: 600px;
    }}

    .content-header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 20px;
      flex-wrap: wrap;
      gap: 16px;
    }}

    .content-title {{
      font-size: 1.125rem;
      font-weight: 600;
      color: var(--text-primary);
    }}

    .content-actions {{
      display: flex;
      gap: 8px;
    }}

    .btn {{
      padding: 8px 16px;
      background: var(--bg-secondary);
      border: 1px solid var(--border);
      border-radius: var(--radius-sm);
      color: var(--text-primary);
      font-size: 0.875rem;
      cursor: pointer;
      transition: var(--transition);
      font-weight: 500;
    }}

    .btn:hover {{
      background: var(--bg-tertiary);
      border-color: var(--accent-primary);
    }}

    .tags {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-bottom: 16px;
    }}

    .tag {{
      padding: 6px 12px;
      border-radius: 999px;
      font-size: 0.75rem;
      font-weight: 600;
      display: inline-flex;
      align-items: center;
      gap: 6px;
    }}

    .tag.parallel {{
      background: linear-gradient(135deg, rgba(139, 92, 246, 0.2), rgba(139, 92, 246, 0.1));
      color: var(--accent-tertiary);
      border: 1px solid rgba(139, 92, 246, 0.3);
    }}

    .tag.thinking {{
      background: linear-gradient(135deg, rgba(59, 130, 246, 0.2), rgba(59, 130, 246, 0.1));
      color: var(--accent-secondary);
      border: 1px solid rgba(59, 130, 246, 0.3);
    }}

    .tag.boxed {{
      background: linear-gradient(135deg, rgba(16, 185, 129, 0.2), rgba(16, 185, 129, 0.1));
      color: var(--accent-primary);
      border: 1px solid rgba(16, 185, 129, 0.3);
    }}

    .tag.outline {{
      background: linear-gradient(135deg, rgba(245, 158, 11, 0.2), rgba(245, 158, 11, 0.1));
      color: var(--accent-warning);
      border: 1px solid rgba(245, 158, 11, 0.3);
    }}

    .metrics-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(100px, 1fr));
      gap: 12px;
      margin-bottom: 16px;
    }}

    .metric {{
      background: var(--bg-secondary);
      padding: 12px;
      border-radius: var(--radius-sm);
      border: 1px solid var(--border);
      text-align: center;
    }}

    .metric .metric-value {{
      font-size: 1.25rem;
      font-weight: 700;
      color: var(--accent-primary);
    }}

    .metric .metric-label {{
      font-size: 0.75rem;
      color: var(--text-muted);
      margin-top: 4px;
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }}

    .content-viewer {{
      flex: 1;
      background: var(--bg-primary);
      border: 1px solid var(--border);
      border-radius: var(--radius-md);
      padding: 20px;
      overflow: auto;
      font-family: 'JetBrains Mono', monospace;
      font-size: 0.875rem;
      line-height: 1.7;
      white-space: pre-wrap;
      word-wrap: break-word;
      max-height: 70vh;
    }}

    .content-viewer.syntax-highlight {{
      color: var(--text-secondary);
    }}

    .content-viewer .highlight-think {{
      color: #60a5fa;
      font-weight: 500;
    }}

    .content-viewer .highlight-parallel {{
      color: #a78bfa;
      font-weight: 500;
    }}

    .content-viewer .highlight-boxed {{
      color: #34d399;
      font-weight: 600;
    }}

    .content-viewer .highlight-outline {{
      color: #fbbf24;
      font-weight: 500;
    }}

    .alert {{
      padding: 12px 16px;
      border-radius: var(--radius-sm);
      margin-top: 12px;
      font-size: 0.875rem;
      display: flex;
      align-items: center;
      gap: 8px;
    }}

    .alert.warning {{
      background: rgba(245, 158, 11, 0.1);
      border: 1px solid rgba(245, 158, 11, 0.3);
      color: var(--accent-warning);
    }}

    .empty-state {{
      text-align: center;
      padding: 60px 20px;
      color: var(--text-muted);
    }}

    .empty-state svg {{
      width: 64px;
      height: 64px;
      margin-bottom: 16px;
      opacity: 0.3;
    }}

    ::-webkit-scrollbar {{
      width: 8px;
      height: 8px;
    }}

    ::-webkit-scrollbar-track {{
      background: var(--bg-secondary);
      border-radius: 4px;
    }}

    ::-webkit-scrollbar-thumb {{
      background: var(--border);
      border-radius: 4px;
    }}

    ::-webkit-scrollbar-thumb:hover {{
      background: var(--border-hover);
    }}

    .loading {{
      display: inline-block;
      width: 16px;
      height: 16px;
      border: 2px solid var(--border);
      border-top-color: var(--accent-primary);
      border-radius: 50%;
      animation: spin 0.8s linear infinite;
    }}

    @keyframes spin {{
      to {{ transform: rotate(360deg); }}
    }}

    .fade-in {{
      animation: fadeIn 0.3s ease-in;
    }}

    @keyframes fadeIn {{
      from {{ opacity: 0; transform: translateY(10px); }}
      to {{ opacity: 1; transform: translateY(0); }}
    }}
  </style>
</head>
<body>
  <div class="header">
    <div class="header-left">
      <h1>ThreadWeaver Explorer</h1>
      <div class="subtitle">Visualize and analyze trajectory generation outputs</div>
    </div>
    <div class="header-right">
      <div class="stat-pill">
        <div class="label">Total Files</div>
        <div class="value" id="total-count">-</div>
      </div>
    </div>
  </div>

  <div class="container">
    <div class="stats-overview" id="stats-overview"></div>

    <div class="main-grid">
      <div class="panel">
        <div class="panel-header">
          <div class="panel-title">Trajectories</div>
        </div>

        <div class="search-box">
          <input type="text" id="search" placeholder="Search trajectories..." />
        </div>

        <div class="traj-list" id="traj-list"></div>
      </div>

      <div class="panel content-panel">
        <div class="content-header">
          <div class="content-title" id="content-title">Select a trajectory</div>
          <div class="content-actions">
            <button class="btn" id="copy-btn" style="display:none;">📋 Copy</button>
            <button class="btn" id="highlight-btn" style="display:none;">✨ Highlight</button>
          </div>
        </div>

        <div id="content-tags" class="tags"></div>
        <div id="content-metrics" class="metrics-grid"></div>
        <div class="content-viewer" id="content-viewer">
          <div class="empty-state">
            <svg fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
            </svg>
            <div>Select a trajectory from the list to view its content</div>
          </div>
        </div>
        <div id="truncated-alert" style="display:none;" class="alert warning">
          ⚠️ Content truncated to improve performance
        </div>
      </div>
    </div>
  </div>

<script>
const fmt = new Intl.NumberFormat('en-US');
let trajectories = [];
let currentId = null;
let highlightEnabled = false;

async function fetchJSON(url) {{
  const res = await fetch(url);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}}

function calculateStats(data) {{
  const totalLines = data.reduce((sum, t) => sum + t.lines, 0);
  const totalWords = data.reduce((sum, t) => sum + t.words, 0);
  const totalChars = data.reduce((sum, t) => sum + t.chars, 0);
  const totalSize = data.reduce((sum, t) => sum + t.size_bytes, 0);

  const avgLines = Math.round(totalLines / data.length);
  const avgWords = Math.round(totalWords / data.length);
  const avgChars = Math.round(totalChars / data.length);

  return {{
    total: data.length,
    totalLines, totalWords, totalChars, totalSize,
    avgLines, avgWords, avgChars,
    maxLines: Math.max(...data.map(t => t.lines)),
    minLines: Math.min(...data.map(t => t.lines))
  }};
}}

function formatBytes(bytes) {{
  if (bytes < 1024) return bytes + ' B';
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
  if (bytes < 1024 * 1024 * 1024) return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
  return (bytes / (1024 * 1024 * 1024)).toFixed(1) + ' GB';
}}

function renderStatsOverview(stats) {{
  const overview = document.getElementById('stats-overview');
  overview.innerHTML = `
    <div class="stat-card fade-in">
      <div class="stat-label">📊 Average Lines</div>
      <div class="stat-value">${{fmt.format(stats.avgLines)}}</div>
      <div class="stat-subtext">Min: ${{fmt.format(stats.minLines)}} • Max: ${{fmt.format(stats.maxLines)}}</div>
    </div>
    <div class="stat-card fade-in">
      <div class="stat-label">📝 Average Words</div>
      <div class="stat-value">${{fmt.format(stats.avgWords)}}</div>
      <div class="stat-subtext">Total: ${{fmt.format(stats.totalWords)}}</div>
    </div>
    <div class="stat-card fade-in">
      <div class="stat-label">💾 Total Size</div>
      <div class="stat-value">${{formatBytes(stats.totalSize)}}</div>
      <div class="stat-subtext">Across ${{stats.total}} files</div>
    </div>
    <div class="stat-card fade-in">
      <div class="stat-label">🔤 Average Chars</div>
      <div class="stat-value">${{fmt.format(stats.avgChars)}}</div>
      <div class="stat-subtext">Total: ${{fmt.format(stats.totalChars)}}</div>
    </div>
  `;
}}

function renderTrajectoryList(data, filter = '') {{
  const list = document.getElementById('traj-list');
  const filtered = filter
    ? data.filter(t => t.name.toLowerCase().includes(filter.toLowerCase()))
    : data;

  list.innerHTML = filtered.map(t => `
    <div class="traj-item ${{currentId === t.id ? 'active' : ''}}" onclick="selectTrajectory(${{t.id}})" data-id="${{t.id}}">
      <div class="traj-name">#${{t.id}} — ${{t.name}}</div>
      <div class="traj-meta">
        <span>${{fmt.format(t.lines)}} lines</span>
        <span>${{fmt.format(t.words)}} words</span>
        <span>${{t.size_pretty}}</span>
      </div>
    </div>
  `).join('');
}}

function analyzeContent(content) {{
  return {{
    hasParallel: content.includes('<Parallel>'),
    hasThink: content.includes('<think>') || content.includes('<Think>'),
    hasBoxed: content.includes('\\\\boxed'),
    hasOutlines: content.includes('<Outlines>'),
    hasThreads: (content.match(/<Thread>/g) || []).length,
    hasConclusion: content.includes('<Conclusion>')
  }};
}}

function highlightContent(content) {{
  return content
    .replace(/(<think>|<\/think>|<Think>|<\/Think>)/g, '<span class="highlight-think">$1</span>')
    .replace(/(<Parallel>|<\/Parallel>|<Thread>|<\/Thread>)/g, '<span class="highlight-parallel">$1</span>')
    .replace(/(\\\\boxed\{{[^}}]+\}})/g, '<span class="highlight-boxed">$1</span>')
    .replace(/(<Outlines>|<\/Outlines>|<Outline>)/g, '<span class="highlight-outline">$1</span>');
}}

async function selectTrajectory(id) {{
  currentId = id;
  const data = await fetchJSON(`/api/trajectory?id=${{id}}`);

  if (data.error) {{
    document.getElementById('content-viewer').innerHTML = `<div class="empty-state">${{data.error}}</div>`;
    return;
  }}

  // Update active state
  document.querySelectorAll('.traj-item').forEach(item => {{
    item.classList.toggle('active', parseInt(item.dataset.id) === id);
  }});

  // Analysis
  const analysis = analyzeContent(data.content);

  // Render tags
  const tags = [];
  if (analysis.hasParallel) tags.push('<div class="tag parallel">🔀 Parallel</div>');
  if (analysis.hasThink) tags.push('<div class="tag thinking">💭 Thinking</div>');
  if (analysis.hasBoxed) tags.push('<div class="tag boxed">📦 Boxed Answer</div>');
  if (analysis.hasOutlines) tags.push('<div class="tag outline">📋 Outlines</div>');
  if (analysis.hasThreads) tags.push(`<div class="tag outline">🧵 ${{analysis.hasThreads}} Thread${{analysis.hasThreads > 1 ? 's' : ''}}</div>`);

  document.getElementById('content-tags').innerHTML = tags.join('');

  // Render metrics
  document.getElementById('content-metrics').innerHTML = `
    <div class="metric">
      <div class="metric-value">${{fmt.format(data.lines)}}</div>
      <div class="metric-label">Lines</div>
    </div>
    <div class="metric">
      <div class="metric-value">${{fmt.format(data.words)}}</div>
      <div class="metric-label">Words</div>
    </div>
    <div class="metric">
      <div class="metric-value">${{fmt.format(data.chars)}}</div>
      <div class="metric-label">Characters</div>
    </div>
    <div class="metric">
      <div class="metric-value">${{data.size_pretty}}</div>
      <div class="metric-label">Size</div>
    </div>
  `;

  // Update content
  document.getElementById('content-title').textContent = data.name;
  const viewer = document.getElementById('content-viewer');
  viewer.textContent = data.content;

  if (highlightEnabled) {{
    viewer.innerHTML = highlightContent(data.content);
    viewer.classList.add('syntax-highlight');
  }} else {{
    viewer.classList.remove('syntax-highlight');
  }}

  document.getElementById('truncated-alert').style.display = data.truncated ? 'block' : 'none';
  document.getElementById('copy-btn').style.display = 'block';
  document.getElementById('highlight-btn').style.display = 'block';
}}

// Event listeners
document.getElementById('search').addEventListener('input', (e) => {{
  renderTrajectoryList(trajectories, e.target.value);
}});

document.getElementById('copy-btn').addEventListener('click', () => {{
  const content = document.getElementById('content-viewer').textContent;
  navigator.clipboard.writeText(content).then(() => {{
    const btn = document.getElementById('copy-btn');
    btn.textContent = '✓ Copied!';
    setTimeout(() => btn.textContent = '📋 Copy', 2000);
  }});
}});

document.getElementById('highlight-btn').addEventListener('click', () => {{
  highlightEnabled = !highlightEnabled;
  const btn = document.getElementById('highlight-btn');
  btn.textContent = highlightEnabled ? '✨ Plain' : '✨ Highlight';
  if (currentId !== null) selectTrajectory(currentId);
}});

// Keyboard shortcuts
document.addEventListener('keydown', (e) => {{
  if (e.key === 'ArrowDown' && currentId !== null && currentId < trajectories.length - 1) {{
    e.preventDefault();
    selectTrajectory(currentId + 1);
    document.querySelector(`[data-id="${{currentId + 1}}"]`)?.scrollIntoView({{ block: 'nearest' }});
  }}
  if (e.key === 'ArrowUp' && currentId !== null && currentId > 0) {{
    e.preventDefault();
    selectTrajectory(currentId - 1);
    document.querySelector(`[data-id="${{currentId - 1}}"]`)?.scrollIntoView({{ block: 'nearest' }});
  }}
  if (e.key === '/' && document.activeElement.id !== 'search') {{
    e.preventDefault();
    document.getElementById('search').focus();
  }}
}});

async function init() {{
  const meta = await fetchJSON('/api/trajectories');
  trajectories = meta.stats;

  document.getElementById('total-count').textContent = meta.count;

  const stats = calculateStats(trajectories);
  renderStatsOverview(stats);
  renderTrajectoryList(trajectories);

  if (trajectories.length) selectTrajectory(trajectories[0].id);
}}

init().catch(err => {{
  document.getElementById('content-viewer').innerHTML = `<div class="empty-state">Failed to load: ${{err.message}}</div>`;
}});
</script>
</body>
</html>
"""


def main():
    parser = argparse.ArgumentParser(description="Serve a trajectory browser UI.")
    parser.add_argument("--root", type=str, required=True, help="Directory containing trajectory .txt files (e.g., step6.1 output).")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind. Defaults to 127.0.0.1 for safety.")
    parser.add_argument("--port", type=int, default=8899, help="Port to serve on.")
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        raise SystemExit(f"Root does not exist or is not a directory: {root}")

    trajectories = scan(root)
    if not trajectories:
        raise SystemExit(f"No .txt trajectories found under {root}")

    class _Server(ThreadingHTTPServer):
        daemon_threads = True

    server = _Server((args.host, args.port), TrajectoryHandler)
    server.root = root
    server.trajectories = trajectories

    print(f"Serving {len(trajectories)} trajectories from {root}")
    print(f"Open http://{args.host}:{args.port} in your browser.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")


if __name__ == "__main__":
    main()
