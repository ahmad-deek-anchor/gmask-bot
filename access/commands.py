"""Slack-side administration of the access rules, without the model in the loop.

Any message to the bot that starts with ``access`` (or ``permissions``) is handled here:

    access help
    access whoami
    access list                                  (lead, admin)
    access add user @someone <role>              (admin)   roles: viewer, desk, lead, admin, none
    access remove user @someone                  (admin)
    access add channel [here | #channel] [confidential]   (admin)
    access remove channel [here | #channel]      (admin)
    access default <role>                        (admin)
    access pause | access resume                 (admin)

Slack delivers mentions as ``<@U123>`` and channels as ``<#C123|name>``; the parser accepts
those, bare ids, and "here" / "me". The reply is Slack mrkdwn.
"""

from __future__ import annotations

import re
from typing import Optional

from access.policy import CHANNEL_MODES, ROLE_ORDER, AccessControl, Decision

COMMAND_RE = re.compile(r"^\s*(?:access|permissions?|perms)\b", re.IGNORECASE)
USER_RE = re.compile(r"<@([A-Z0-9]+)(?:\|[^>]*)?>|\b(U[A-Z0-9]{6,})\b")
CHANNEL_RE = re.compile(r"<#([A-Z0-9]+)(?:\|[^>]*)?>|\b(C[A-Z0-9]{6,}|G[A-Z0-9]{6,})\b")
LEADING_MENTION_RE = re.compile(r"^\s*<@[A-Z0-9]+>\s*")

HELP = (
    "*GM bot access commands*\n"
    "• `access whoami` - your role and this channel's status\n"
    "• `access list` - users, channels and settings (lead, admin)\n"
    "• `access add user @someone <viewer|desk|lead|admin|none>` (admin)\n"
    "• `access remove user @someone` (admin)\n"
    "• `access add channel here` or `access add channel #name` - enable the bot there; add `confidential` "
    "to allow desk positions / PnL in that channel (admin)\n"
    "• `access remove channel here|#name` (admin)\n"
    "• `access default <role>` - role for users not listed (admin)\n"
    "• `access pause` / `access resume` - stop or restart answering for everyone but admins (admin)\n"
    "_Roles: viewer = public market data, news, ETF, CME, macro; desk = + positions, PnL, spot sheet, full report; "
    "lead = + raw tables, shared memory, channel rules; admin = everything + these commands._"
)


def is_access_command(text: str) -> bool:
    return bool(COMMAND_RE.match(text or ""))


def strip_leading_mention(raw_text: str) -> str:
    return LEADING_MENTION_RE.sub("", raw_text or "", count=1).strip()


def _user_id(token: str, sender: str) -> Optional[str]:
    if token.lower() == "me":
        return sender
    m = USER_RE.search(token)
    return (m.group(1) or m.group(2)) if m else None


def _channel_id(token: Optional[str], here: Optional[str]) -> Optional[str]:
    if not token or token.lower() == "here":
        return here
    m = CHANNEL_RE.search(token)
    return (m.group(1) or m.group(2)) if m else None


def _tokens(text: str) -> list[str]:
    body = COMMAND_RE.sub("", text, count=1).strip()
    return body.split()


def run_access_command(raw_text: str, decision: Decision, control: AccessControl) -> str:
    """Execute one command for the sender described by ``decision``; returns the reply text."""
    text = strip_leading_mention(raw_text)
    toks = _tokens(text)
    sender = decision.user_id
    here = decision.channel_id
    verb = toks[0].lower() if toks else "help"
    args = toks[1:]
    admin = decision.is_admin

    if verb in ("help", "?"):
        return HELP

    if verb == "whoami":
        place = "DM" if decision.is_dm else f"channel <#{here}> ({control.channel_mode(here) or 'not enabled'})"
        return (f"You are <@{sender}> with role *{decision.role}*; this is a {place}. "
                f"Default role for unlisted users: *{control.default_role}*." + (" The bot is *paused*." if decision.paused else ""))

    if verb == "list":
        if decision.role not in ("lead", "admin"):
            return "Only leads and admins can list access rules."
        rules = control.rules(force=True)
        lines = ["*Access rules*"]
        lines.append(f"• default role: *{control.default_role}*" + (" • *PAUSED*" if control.paused else ""))
        admins = ", ".join(f"<@{u}>" for u in control.policy.bootstrap_admins) or "-"
        lines.append(f"• bootstrap admins: {admins}")
        users = rules["users"]
        if users:
            by_role: dict[str, list[str]] = {}
            for uid, role in sorted(users.items()):
                by_role.setdefault(role, []).append(f"<@{uid}>")
            for role in reversed(ROLE_ORDER):
                if role in by_role:
                    lines.append(f"• {role}: {', '.join(by_role[role])}")
        else:
            lines.append("• users: none listed (everyone gets the default role)")
        chans = rules["channels"]
        if chans:
            for cid, mode in sorted(chans.items()):
                lines.append(f"• channel <#{cid}>: {mode}")
        else:
            lines.append("• channels: none enabled yet (`access add channel here`)")
        return "\n".join(lines)

    if not admin:
        return f"Only admins can change access (you are *{decision.role}*). Try `access whoami` or `access help`."

    try:
        if verb in ("add", "set", "grant") and args and args[0].lower() == "user":
            if len(args) < 3:
                return "Usage: `access add user @someone <viewer|desk|lead|admin|none>`"
            uid = _user_id(args[1], sender)
            role = args[2].lower()
            if not uid:
                return f"Could not read a Slack user from `{args[1]}`; mention them with @."
            control.set_user(uid, role, by=sender)
            return f"Done: <@{uid}> is now *{role}*."
        if verb in ("remove", "delete", "revoke") and args and args[0].lower() == "user":
            if len(args) < 2:
                return "Usage: `access remove user @someone`"
            uid = _user_id(args[1], sender)
            if not uid:
                return f"Could not read a Slack user from `{args[1]}`."
            ok = control.remove_user(uid, by=sender)
            return (f"Done: <@{uid}> removed; they fall back to the default role *{control.default_role}*." if ok
                    else f"<@{uid}> was not listed (they already have the default role *{control.default_role}*).")
        if verb in ("add", "set", "enable") and args and args[0].lower() == "channel":
            target = _channel_id(args[1] if len(args) > 1 and args[1].lower() != "confidential" else None, here)
            if not target:
                return "Could not read a channel; use `here` or #channel."
            mode = "confidential" if any(a.lower() == "confidential" for a in args[1:]) else "allowed"
            control.set_channel(target, mode, by=sender)
            extra = (" Confidential desk tools (positions, PnL, spot sheet) are enabled here." if mode == "confidential"
                     else " Public tools only; add `confidential` to allow desk data here.")
            return f"Done: the bot now answers in <#{target}> ({mode}).{extra}"
        if verb in ("remove", "delete", "disable") and args and args[0].lower() == "channel":
            target = _channel_id(args[1] if len(args) > 1 else None, here)
            if not target:
                return "Could not read a channel; use `here` or #channel."
            ok = control.remove_channel(target, by=sender)
            return f"Done: the bot no longer answers in <#{target}>." if ok else f"<#{target}> was not enabled."
        if verb == "default" and args:
            control.set_default_role(args[0].lower(), by=sender)
            return f"Done: users not listed now get the *{args[0].lower()}* role."
        if verb == "pause":
            control.set_paused(True, by=sender)
            return "Paused: the bot answers admins only until `access resume`."
        if verb == "resume":
            control.set_paused(False, by=sender)
            return "Resumed: the bot answers everyone again."
    except ValueError as e:
        return f"{e}"
    return "I did not understand that access command.\n" + HELP


__all__ = ["HELP", "is_access_command", "run_access_command", "strip_leading_mention"]
