"""One Telegram inbox: entries, exits, orphans, assignments, failed jobs.

Pure assembly over already-fetched rows. Callers own I/O. Stale entry cards
(older than ``ttl_hours``) are marked expired rather than dropped so the
operator sees them die.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional

from earnings_edge import cards

KINDS = ("entry", "exit", "orphan", "assignment", "job")
DEFAULT_TTL_HOURS = 8.0


def _parse_ts(raw) -> Optional[datetime]:
    if raw is None:
        return None
    if isinstance(raw, datetime):
        ts = raw
    else:
        try:
            ts = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
        except ValueError:
            return None
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts


@dataclass
class InboxItem:
    kind: str
    item_id: str
    ticker: str
    created_at: Optional[str] = None
    expired: bool = False
    detail: str = ""
    strategy: str = ""
    actions: tuple = ()  # Execute/Skip or Close/Snooze or Adopt/Close/Ignore


@dataclass
class Inbox:
    items: list[InboxItem] = field(default_factory=list)

    @property
    def live(self) -> list[InboxItem]:
        return [i for i in self.items if not i.expired]

    @property
    def expired(self) -> list[InboxItem]:
        return [i for i in self.items if i.expired]

    def by_kind(self) -> dict[str, list[InboxItem]]:
        out: dict[str, list[InboxItem]] = {k: [] for k in KINDS}
        for i in self.items:
            out.setdefault(i.kind, []).append(i)
        return out

    def grouped(self) -> dict[str, list[InboxItem]]:
        """Group live items by kind, expired last under 'expired'."""
        groups: dict[str, list[InboxItem]] = {}
        for i in self.live:
            groups.setdefault(i.kind, []).append(i)
        if self.expired:
            groups["expired"] = list(self.expired)
        return groups


def assemble_inbox(
    *,
    entries: list[dict] | None = None,
    exits: list[dict] | None = None,
    orphans: list[dict] | None = None,
    assignments: list[dict] | None = None,
    jobs: list[dict] | None = None,
    now: Optional[datetime] = None,
    ttl_hours: float = DEFAULT_TTL_HOURS,
) -> Inbox:
    """Build one inbox. ``now`` is injectable for tests."""
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    cutoff = now - timedelta(hours=ttl_hours)
    inbox = Inbox()

    for row in entries or []:
        ts = _parse_ts(row.get("created_at"))
        expired = ts is not None and ts < cutoff
        inbox.items.append(InboxItem(
            kind="entry",
            item_id=str(row.get("id", "")),
            ticker=str(row.get("ticker") or ""),
            created_at=row.get("created_at"),
            expired=expired,
            detail=str(row.get("side") or ""),
            strategy=str(row.get("strategy") or ""),
            actions=("Execute", "Skip") if not expired else (),
        ))
    for row in exits or []:
        inbox.items.append(InboxItem(
            kind="exit",
            item_id=str(row.get("id", "")),
            ticker=str(row.get("ticker") or ""),
            created_at=row.get("created_at"),
            expired=False,
            detail=str(row.get("rule") or ""),
            strategy=str(row.get("strategy") or ""),
            actions=("Close", "Snooze"),
        ))
    for row in orphans or []:
        inbox.items.append(InboxItem(
            kind="orphan",
            item_id=str(row.get("symbol") or row.get("id") or ""),
            ticker=str(row.get("ticker") or row.get("symbol") or ""),
            created_at=row.get("ts") or row.get("created_at"),
            detail=str(row.get("detail") or "at broker, not local"),
            actions=("Adopt", "Close", "Ignore"),
        ))
    for row in assignments or []:
        inbox.items.append(InboxItem(
            kind="assignment",
            item_id=str(row.get("symbol") or ""),
            ticker=str(row.get("ticker") or row.get("symbol") or ""),
            created_at=row.get("ts"),
            detail=str(row.get("detail") or "short call became stock"),
            actions=("Cover", "Adopt"),
        ))
    for row in jobs or []:
        inbox.items.append(InboxItem(
            kind="job",
            item_id=str(row.get("id") or row.get("job_name") or ""),
            ticker=str(row.get("job_name") or "job"),
            created_at=row.get("finished_at") or row.get("started_at"),
            detail=str(row.get("error") or "failed"),
            actions=(),
        ))
    return inbox


def render_inbox(inbox: Inbox) -> str:
    """HTML (escaped) operator view of the inbox."""
    lines = [cards.header("📥", "Pending inbox")]
    groups = inbox.grouped()
    if not groups:
        lines.append("Nothing pending.")
        return "\n".join(lines)
    labels = {
        "entry": "Entries",
        "exit": "Exits",
        "orphan": "Orphans",
        "assignment": "Assignments",
        "job": "Failed jobs",
        "expired": "Expired (stale)",
    }
    for kind, items in groups.items():
        lines.append("")
        lines.append(cards.bold(labels.get(kind, kind)))
        for it in items:
            acts = f" [{'/'.join(it.actions)}]" if it.actions else ""
            expired = " EXPIRED" if it.expired else ""
            tail = f" — {cards.esc(it.detail)}" if it.detail else ""
            lines.append(f"• {cards.bold(it.ticker)} {cards.esc(it.kind)}{expired}{tail}{cards.esc(acts)}")
    return "\n".join(lines)


def inbox_keyboard(inbox: Inbox) -> list[list]:
    """Inline rows for the live inbox panel (callback_data ≤ 64 bytes)."""
    from telegram import InlineKeyboardButton
    rows = []
    for it in inbox.live:
        if it.kind == "entry" and it.item_id.isdigit():
            rows.append([
                InlineKeyboardButton(f"✅ {it.ticker}", callback_data=f"in_ex_{it.item_id}"),
                InlineKeyboardButton("Skip", callback_data=f"in_sk_{it.item_id}"),
            ])
        elif it.kind == "exit" and it.item_id.isdigit():
            rows.append([
                InlineKeyboardButton(f"🔒 {it.ticker}", callback_data=f"in_cl_{it.item_id}"),
                InlineKeyboardButton("Snooze", callback_data=f"in_sn_{it.item_id}"),
            ])
        elif it.kind == "orphan" and it.item_id:
            rows.append([
                InlineKeyboardButton(f"Adopt {it.ticker}", callback_data=f"in_ad_{it.item_id}"),
                InlineKeyboardButton("Ignore", callback_data=f"in_ig_{it.item_id}"),
            ])
        elif it.kind == "assignment" and it.item_id:
            rows.append([
                InlineKeyboardButton(f"Adopt {it.ticker}", callback_data=f"in_ad_{it.item_id}"),
                InlineKeyboardButton(f"Close {it.ticker}", callback_data=f"in_xs_{it.item_id}"),
            ])
    if len(rows) > 98:
        rows = rows[:98]
    rows.append([InlineKeyboardButton("🔄 Refresh", callback_data="desk_pd")])
    return rows
