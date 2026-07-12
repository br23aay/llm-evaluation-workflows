"""
llm_eval_project/src/report.py
──────────────────────────────────────────────────────────────────────────────
Evaluation Report Generator
Bharadwaj Rachuri | github.com/br23aay
──────────────────────────────────────────────────────────────────────────────

Generates:
  - Console summary (coloured terminal output)
  - JSON results file
  - HTML report (self-contained, no server needed)
"""

from __future__ import annotations

import json
import os
from collections import Counter
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .evaluator import EvalResult


# ─── CONSOLE REPORT ──────────────────────────────────────────────────────────

_RESET  = "\033[0m"
_BOLD   = "\033[1m"
_GREEN  = "\033[92m"
_YELLOW = "\033[93m"
_RED    = "\033[91m"
_CYAN   = "\033[96m"
_GREY   = "\033[90m"


def _colour(text: str, status: str) -> str:
    c = {
        "PASS": _GREEN,
        "WARN": _YELLOW,
        "FAIL": _RED,
        "INFO": _CYAN,
    }.get(status, _RESET)
    return f"{c}{text}{_RESET}"


def print_summary(results: list["EvalResult"]) -> None:
    """Print a formatted summary to console."""
    n = len(results)
    counts = Counter(r.status for r in results)
    avg_composite = sum(r.composite_score for r in results) / n if n else 0
    avg_h = sum(r.hallucination_score for r in results) / n if n else 0
    avg_s = sum(r.safety_score for r in results) / n if n else 0
    avg_a = sum(r.alignment_score for r in results) / n if n else 0

    print(f"\n{_BOLD}{'─' * 70}{_RESET}")
    print(f"{_BOLD}  LLM EVALUATION SUMMARY{_RESET}  ·  {n} outputs evaluated")
    print(f"{'─' * 70}")
    p_str = f"PASS: {counts['PASS']}"
    w_str = f"WARN: {counts['WARN']}"
    f_str = f"FAIL: {counts['FAIL']}"
    print(
        f"  Status breakdown:  "
        f"{_colour(p_str, 'PASS')}  "
        f"{_colour(w_str, 'WARN')}  "
        f"{_colour(f_str, 'FAIL')}"
    )
    print(f"\n  Average scores:")
    print(f"    Composite  : {_BOLD}{avg_composite:.3f}{_RESET}  (threshold: PASS≥0.70, WARN≥0.50)")
    print(f"    Hallucination : {avg_h:.3f}  (weight 35%)")
    print(f"    Safety        : {avg_s:.3f}  (weight 40%)")
    print(f"    Alignment     : {avg_a:.3f}  (weight 25%)")

    # Worst performers
    failures = [r for r in results if r.status == "FAIL"]
    if failures:
        print(f"\n  {_colour('FAIL cases:', 'FAIL')}")
        for r in failures[:5]:
            print(f"    · {r.output_id:<25} composite={r.composite_score:.3f}")

    # Most common flags
    all_flags = (
        [f for r in results for f in r.hallucination_flags]
        + [f for r in results for f in r.safety_flags]
        + [f for r in results for f in r.alignment_flags]
    )
    if all_flags:
        common = Counter(all_flags).most_common(3)
        print(f"\n  Most common issues:")
        for flag, count in common:
            trimmed = flag[:70] + "..." if len(flag) > 70 else flag
            print(f"    · [{count}×] {_GREY}{trimmed}{_RESET}")

    print(f"{'─' * 70}\n")


# ─── JSON EXPORT ─────────────────────────────────────────────────────────────

def save_json(results: list["EvalResult"], path: str) -> None:
    """Save evaluation results as JSON."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "total_evaluated": len(results),
        "pass_count": sum(1 for r in results if r.status == "PASS"),
        "warn_count": sum(1 for r in results if r.status == "WARN"),
        "fail_count": sum(1 for r in results if r.status == "FAIL"),
        "avg_composite": round(sum(r.composite_score for r in results) / len(results), 4) if results else 0,
        "results": [r.to_dict() for r in results],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"  JSON saved → {path}")


# ─── HTML REPORT ─────────────────────────────────────────────────────────────

def _status_colour(status: str) -> str:
    return {"PASS": "#00e676", "WARN": "#ff9100", "FAIL": "#ff1744"}.get(status, "#888")


def save_html_report(results: list["EvalResult"], path: str) -> None:
    """Generate a self-contained HTML evaluation report."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    n = len(results)
    counts = Counter(r.status for r in results)
    avg = lambda attr: round(sum(getattr(r, attr) for r in results) / n, 3) if n else 0

    rows = ""
    for r in results:
        sc = _status_colour(r.status)
        flags_html = ""
        all_flags = r.hallucination_flags + r.safety_flags + r.alignment_flags
        if all_flags:
            flags_html = "<br>".join(f"<span class='flag'>⚠ {f[:80]}</span>" for f in all_flags[:4])

        rows += f"""
        <tr>
          <td class="mono">{r.output_id}</td>
          <td><span class="status-badge" style="color:{sc};border-color:{sc}">{r.status}</span></td>
          <td class="score">{r.composite_score:.3f}</td>
          <td class="score">{r.hallucination_score:.2f}</td>
          <td class="score">{r.safety_score:.2f}</td>
          <td class="score">{r.alignment_score:.2f}</td>
          <td class="small">{r.prompt[:80]}{'...' if len(r.prompt)>80 else ''}</td>
          <td class="small flags">{flags_html}</td>
          <td class="score small">{r.latency_ms:.0f}ms</td>
        </tr>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>LLM Evaluation Report — Bharadwaj Rachuri</title>
<style>
  :root {{
    --bg:#0d0f1a; --card:#131627; --border:#1e2540;
    --cyan:#00e5ff; --purple:#9c5fff; --green:#00e676;
    --orange:#ff9100; --red:#ff1744; --text:#e0e6ff; --muted:#6b7aab;
  }}
  * {{ box-sizing:border-box; margin:0; padding:0; }}
  body {{ background:var(--bg); color:var(--text); font-family:-apple-system,sans-serif; padding:2rem; }}
  h1 {{ font-size:20px; color:var(--cyan); margin-bottom:4px; letter-spacing:1px; }}
  .sub {{ font-size:12px; color:var(--muted); margin-bottom:1.5rem; }}
  .kpi-row {{ display:grid; grid-template-columns:repeat(4,1fr); gap:12px; margin-bottom:1.5rem; }}
  .kpi {{ background:var(--card); border:1px solid var(--border); border-radius:10px; padding:1rem; }}
  .kpi-label {{ font-size:10px; color:var(--muted); text-transform:uppercase; letter-spacing:1px; margin-bottom:6px; }}
  .kpi-val {{ font-size:24px; font-weight:700; }}
  table {{ width:100%; border-collapse:collapse; background:var(--card); border-radius:10px; overflow:hidden; font-size:12px; }}
  thead th {{ background:#0d1128; color:var(--muted); padding:10px 12px; text-align:left; font-size:10px; text-transform:uppercase; letter-spacing:1px; }}
  tbody td {{ padding:9px 12px; border-bottom:1px solid var(--border); vertical-align:top; }}
  tbody tr:hover td {{ background:rgba(255,255,255,0.02); }}
  .mono {{ font-family:monospace; color:var(--cyan); font-size:11px; }}
  .score {{ font-weight:600; }}
  .small {{ font-size:11px; color:var(--muted); max-width:200px; }}
  .flags {{ max-width:260px; }}
  .flag {{ display:block; color:var(--orange); font-size:10px; margin-top:2px; }}
  .status-badge {{ font-size:10px; font-weight:700; padding:2px 8px; border-radius:20px; border:1px solid; letter-spacing:0.5px; }}
  footer {{ text-align:center; margin-top:2rem; font-size:11px; color:var(--muted); }}
  footer a {{ color:var(--cyan); }}
  @media(max-width:768px) {{ .kpi-row {{ grid-template-columns:repeat(2,1fr); }} }}
</style>
</head>
<body>
<h1>LLM Evaluation Report</h1>
<div class="sub">
  Hallucination · Safety · Alignment · {n} outputs evaluated ·
  Generated {datetime.utcnow().strftime('%Y-%m-%d %H:%M')} UTC ·
  Bharadwaj Rachuri | <a href="https://br23aay.github.io" style="color:var(--cyan)">br23aay.github.io</a>
</div>

<div class="kpi-row">
  <div class="kpi">
    <div class="kpi-label">Total Evaluated</div>
    <div class="kpi-val" style="color:var(--cyan)">{n}</div>
  </div>
  <div class="kpi">
    <div class="kpi-label">PASS</div>
    <div class="kpi-val" style="color:var(--green)">{counts['PASS']}</div>
  </div>
  <div class="kpi">
    <div class="kpi-label">WARN</div>
    <div class="kpi-val" style="color:var(--orange)">{counts['WARN']}</div>
  </div>
  <div class="kpi">
    <div class="kpi-label">FAIL</div>
    <div class="kpi-val" style="color:var(--red)">{counts['FAIL']}</div>
  </div>
    <div class="kpi">
      <div class="kpi-label">Avg Composite</div>
      <div class="kpi-val" style="color:var(--cyan)">{avg('composite_score')}</div>
    </div>
    <div class="kpi">
      <div class="kpi-label">Avg Hallucination</div>
      <div class="kpi-val" style="color:var(--purple)">{avg('hallucination_score')}</div>
    </div>
    <div class="kpi">
      <div class="kpi-label">Avg Safety</div>
      <div class="kpi-val" style="color:var(--green)">{avg('safety_score')}</div>
    </div>
    <div class="kpi">
      <div class="kpi-label">Avg Alignment</div>
      <div class="kpi-val" style="color:var(--orange)">{avg('alignment_score')}</div>
    </div>
</div>

<table>
  <thead>
    <tr>
      <th>ID</th><th>Status</th><th>Composite</th>
      <th>Hallucination</th><th>Safety</th><th>Alignment</th>
      <th>Prompt</th><th>Flags</th><th>Latency</th>
    </tr>
  </thead>
  <tbody>{rows}</tbody>
</table>

<footer>
  LLM Evaluation Workflow · Azure AI Foundry Research ·
  <a href="https://github.com/br23aay">github.com/br23aay</a>
</footer>
</body></html>"""

    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  HTML report saved → {path}")
