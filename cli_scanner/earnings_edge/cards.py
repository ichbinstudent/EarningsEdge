"""Shared Telegram HTML rendering for signals: entry proposals, the FF
ladder, and exits all render through this module so they share one visual
language — bold headers, consistent emoji, escaped dynamic text — whether
they arrive as a single card or grouped under a per-strategy batch message.

HTML, not Markdown: strategy names and OCC option symbols are full of
underscores, which silently break Telegram's Markdown parser and drop the
message. HTML has no such landmine as long as dynamic text is escaped —
every function here does that via `esc()`, so callers never need to.

Pure string building; no Telegram imports (unit-testable).
"""
from __future__ import annotations

import html
from typing import Optional

ENTRY_EMOJI = "📋"
FF_EMOJI = "🪜"
EXIT_EMOJI = "🚪"
AUTO_EMOJI = "⚡"


def esc(value) -> str:
    """HTML-escape any dynamic value before it goes into a card."""
    return html.escape(str(value), quote=False)


def bold(value) -> str:
    return f"<b>{esc(value)}</b>"


def code(value) -> str:
    return f"<code>{esc(value)}</code>"


def header(emoji: str, title: str) -> str:
    return f"{emoji} {bold(title)}"


# ── single-card scaffold ────────────────────────────────────────────────

def card_frame(emoji: str, title: str, subtitle: str, body_lines: list[str],
               footer: str = "") -> str:
    """Common visual scaffold for entry/exit/FF cards: bold emoji header,
    a plain (already HTML-safe) subtitle line, pre-built HTML-safe body
    lines, and an optional footer. Callers escape their own dynamic values
    via esc()/code()/bold() before handing lines in here."""
    lines = [header(emoji, title)]
    try:
        from earnings_edge.alpaca_mode import alpaca_live_enabled
        if alpaca_live_enabled():
            lines.append("🔴 <b>LIVE</b>")
    except Exception:
        pass
    if subtitle:
        lines.append(subtitle)
    lines.extend(body_lines)
    if footer:
        lines.append(footer)
    return "\n".join(lines)


# ── grouping (batched per-strategy messages) ────────────────────────────

def group_by_strategy(rows: list[dict]) -> dict[str, list[dict]]:
    """Preserve first-seen order; group proposal/exit rows by strategy."""
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(row["strategy"], []).append(row)
    return groups


def entry_summary_line(row: dict) -> str:
    """One line for a pending_trades row inside a grouped batch message."""
    score = row.get("model_score")
    score_txt = f" · ML {score:.3f}" if score is not None else ""
    return f"• {bold(row['ticker'])} {esc(row['side'])}{score_txt}"


def exit_summary_line(row: dict) -> str:
    """One line for an exit_proposals row inside a grouped batch message."""
    reason = (row.get("reason") or "")[:60]
    tail = f" — {esc(reason)}" if reason else ""
    return f"• {bold(row['ticker'])} {esc(row['rule'])}{tail}"


_SUMMARY_LINE = {"entry": entry_summary_line, "exit": exit_summary_line}
_GROUP_EMOJI = {"entry": ENTRY_EMOJI, "exit": EXIT_EMOJI}
_GROUP_NOUN = {"entry": "proposal", "exit": "exit"}


def group_message(strategy: str, rows: list[dict], kind: str) -> str:
    """Full body for a per-strategy batch message: header + one line/row.

    ``kind`` is "entry" (pending_trades rows) or "exit" (exit_proposals
    rows). Used both when a batch is first pushed and to rebuild the
    message after one row in the group is decided.
    """
    emoji = _GROUP_EMOJI[kind]
    noun = _GROUP_NOUN[kind]
    n = len(rows)
    plural = "" if n == 1 else "s"
    lines = [f"{emoji} {bold(strategy)} — {n} {noun}{plural} pending"]
    line_fn = _SUMMARY_LINE[kind]
    lines.extend(line_fn(r) for r in rows)
    return "\n".join(lines)


def batch_overview(strategy_counts: dict, kind: str = "entry",
                    extra: Optional[str] = None) -> str:
    """Top-of-cycle summary across every strategy in one push cycle."""
    emoji = _GROUP_EMOJI[kind]
    noun = _GROUP_NOUN[kind]
    total = sum(strategy_counts.values())
    plural = "" if total == 1 else "s"
    lines = [f"{emoji} {bold(f'{total} {noun}{plural} pending')}"]
    for name, n in strategy_counts.items():
        lines.append(f"  {esc(name)}: {n}")
    if extra:
        lines.append(esc(extra))
    return "\n".join(lines)
