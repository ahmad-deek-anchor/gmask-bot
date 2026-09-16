"""Access control for the Slack bot: roles -> tool groups, places, and the tool wrapper that
enforces them at execution time.

    policy   = Policy.load("access/policy.yaml")                # static: roles, tool groups, messages
    store    = open_access_store()                              # dynamic: users, channels, settings
    control  = AccessControl(policy, store)
    decision = control.decide(user_id, channel_id, is_dm)       # role + place for one message
    tools    = control.gate_tools(default_tools())              # every tool re-checks the policy when called

The wrapper reads the request identity from ``tools.context`` (set by ``slack_bot.answer`` around
the agent turn), so nothing about permissions is ever passed through - or decided by - the model.
A denied call returns a short message the model relays; it never raises into the agent.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

import yaml

from access.store import AccessStore, AccessStoreUnavailable, MemoryAccessStore, open_access_store
from tools import context as _ctx

logger = logging.getLogger(__name__)
audit = logging.getLogger("access.audit")

DEFAULT_POLICY_PATH = Path(__file__).resolve().parent / "policy.yaml"   # shipped with the package (deploy/ is not in the image)
ROLE_ORDER = ("none", "viewer", "desk", "lead", "admin")
CHANNEL_MODES = ("allowed", "confidential")
UNGATED_TOOLS = {"current_time"}      # utility tools outside every group


# ---------------------------------------------------------------------------
# static policy
# ---------------------------------------------------------------------------

@dataclass
class Policy:
    roles: dict                      # name -> {"inherits": str|None, "tool_groups": [..], "description": str}
    tool_groups: dict                # group -> [tool names]
    confidential_groups: set
    argument_rules: list             # [{"tools": [...], "argument", "values", "requires_group", "confidential"}]
    messages: dict
    default_role: str = "viewer"
    bootstrap_admins: list = field(default_factory=list)
    deny_unlisted_channels: bool = True
    dms_enabled: bool = True
    dms_min_role: str = "viewer"
    dms_confidential_allowed: bool = True
    refresh_seconds: int = 60
    path: Optional[str] = None

    @classmethod
    def load(cls, path: str | Path | None = None) -> "Policy":
        p = Path(path or DEFAULT_POLICY_PATH)
        raw = yaml.safe_load(p.read_text()) or {}
        return cls.from_dict(raw, path=str(p))

    @classmethod
    def from_dict(cls, raw: dict, path: Optional[str] = None) -> "Policy":
        roles = {}
        for name, spec in (raw.get("roles") or {}).items():
            spec = spec or {}
            roles[name] = {"inherits": spec.get("inherits"), "tool_groups": list(spec.get("tool_groups") or []),
                           "description": str(spec.get("description") or "")}
        rules = []
        for r in raw.get("argument_rules") or []:
            tools = r.get("tools") or ([r["tool"]] if r.get("tool") else [])
            rules.append({"tools": list(tools), "argument": r.get("argument"), "values": [str(v).lower() for v in (r.get("values") or [])],
                          "requires_group": r.get("requires_group"), "confidential": bool(r.get("confidential"))})
        places = raw.get("places") or {}
        dms = places.get("dms") or {}
        pol = cls(
            roles=roles, tool_groups={g: list(t or []) for g, t in (raw.get("tool_groups") or {}).items()},
            confidential_groups=set(raw.get("confidential_groups") or []), argument_rules=rules,
            messages=dict(raw.get("messages") or {}), default_role=str(raw.get("default_role") or "viewer"),
            bootstrap_admins=[str(u) for u in (raw.get("bootstrap_admins") or [])],
            deny_unlisted_channels=bool(places.get("deny_unlisted_channels", True)),
            dms_enabled=bool(dms.get("enabled", True)), dms_min_role=str(dms.get("min_role") or "viewer"),
            dms_confidential_allowed=bool(dms.get("confidential_allowed", True)),
            refresh_seconds=int(raw.get("refresh_seconds") or 60), path=path,
        )
        pol.validate()
        return pol

    def validate(self) -> None:
        for name, spec in self.roles.items():
            if name not in ROLE_ORDER:
                raise ValueError(f"unknown role {name!r} (allowed: {', '.join(ROLE_ORDER)})")
            if spec["inherits"] and spec["inherits"] not in self.roles:
                raise ValueError(f"role {name} inherits unknown role {spec['inherits']}")
            for g in spec["tool_groups"]:
                if g not in self.tool_groups:
                    raise ValueError(f"role {name} references unknown tool group {g}")
        for g in self.confidential_groups:
            if g not in self.tool_groups:
                raise ValueError(f"confidential group {g} is not a tool group")
        seen: dict[str, str] = {}
        for g, tools in self.tool_groups.items():
            for t in tools:
                if t in seen:
                    raise ValueError(f"tool {t} is in two groups: {seen[t]} and {g}")
                seen[t] = g
        if self.default_role not in ROLE_ORDER:
            raise ValueError(f"default_role {self.default_role!r} unknown")

    # -- lookups ------------------------------------------------------------

    def groups_for(self, role: str) -> set:
        out: set = set()
        seen = set()
        cur = role
        while cur and cur in self.roles and cur not in seen:
            seen.add(cur)
            out.update(self.roles[cur]["tool_groups"])
            cur = self.roles[cur]["inherits"]
        return out

    def tools_for(self, role: str) -> set:
        return {t for g in self.groups_for(role) for t in self.tool_groups.get(g, [])}

    def group_of(self, tool_name: str) -> Optional[str]:
        for g, tools in self.tool_groups.items():
            if tool_name in tools:
                return g
        return None

    def role_needed_for_group(self, group: str) -> str:
        """Lowest role (by ROLE_ORDER) whose groups include ``group``."""
        for role in ROLE_ORDER:
            if role in self.roles and group in self.groups_for(role):
                return role
        return "admin"

    def message(self, key: str, **kw) -> str:
        text = self.messages.get(key) or key
        try:
            return text.format(**kw)
        except (KeyError, IndexError):
            return text


def role_rank(role: Optional[str]) -> int:
    return ROLE_ORDER.index(role) if role in ROLE_ORDER else 0


# ---------------------------------------------------------------------------
# decisions
# ---------------------------------------------------------------------------

@dataclass
class Decision:
    user_id: str
    channel_id: Optional[str]
    is_dm: bool
    role: str
    place_ok: bool
    place_confidential: bool
    paused: bool
    reason: str = ""

    @property
    def is_admin(self) -> bool:
        return self.role == "admin"

    @property
    def allowed(self) -> bool:
        return self.place_ok and self.role != "none" and (not self.paused or self.is_admin)


class AccessControl:
    """Policy + store + cache. Thread-safe; the cache is refreshed every ``policy.refresh_seconds``."""

    def __init__(self, policy: Policy, store: AccessStore, now: Callable[[], float] = time.time):
        self.policy = policy
        self.store = store
        self._now = now
        self._lock = threading.Lock()
        self._cache: Optional[dict] = None
        self._cache_at = 0.0

    # -- rules ----------------------------------------------------------------

    def rules(self, force: bool = False) -> dict:
        """{"users": {id: role}, "channels": {id: mode}, "settings": {key: value}} from the store (cached)."""
        with self._lock:
            if self._cache is not None and not force and (self._now() - self._cache_at) < self.policy.refresh_seconds:
                return self._cache
            out = {"users": {}, "channels": {}, "settings": {}, "meta": {}}
            try:
                for r in self.store.list_rules():
                    bucket = {"user": "users", "channel": "channels", "setting": "settings"}.get(r["kind"])
                    if bucket:
                        out[bucket][r["id"]] = r["value"]
                        out["meta"][(r["kind"], r["id"])] = r
            except Exception as e:  # noqa: BLE001 - keep the last good cache
                logger.error("access store read failed (%s); using cached rules", e)
                if self._cache is not None:
                    return self._cache
            self._cache, self._cache_at = out, self._now()
            return out

    def invalidate(self) -> None:
        with self._lock:
            self._cache = None

    @property
    def default_role(self) -> str:
        return self.rules()["settings"].get("default_role") or self.policy.default_role

    @property
    def paused(self) -> bool:
        return self.rules()["settings"].get("paused") == "1"

    def role_for(self, user_id: Optional[str]) -> str:
        uid = user_id or ""
        if uid in self.policy.bootstrap_admins:
            return "admin"
        role = self.rules()["users"].get(uid)
        if role in ROLE_ORDER:
            return role
        return self.default_role

    def channel_mode(self, channel_id: Optional[str]) -> Optional[str]:
        return self.rules()["channels"].get(channel_id or "")

    def decide(self, user_id: Optional[str], channel_id: Optional[str], is_dm: bool) -> Decision:
        role = self.role_for(user_id)
        paused = self.paused
        if is_dm:
            place_ok = self.policy.dms_enabled and role_rank(role) >= role_rank(self.policy.dms_min_role)
            confidential = self.policy.dms_confidential_allowed
            reason = "dm"
        else:
            mode = self.channel_mode(channel_id)
            place_ok = mode in CHANNEL_MODES or not self.policy.deny_unlisted_channels
            confidential = mode == "confidential"
            reason = f"channel:{mode or 'unlisted'}"
        if role == "admin":
            place_ok = True          # admins can run `access add channel here` anywhere
        return Decision(user_id or "", channel_id, bool(is_dm), role, place_ok, confidential, paused, reason)

    def current_decision(self) -> Decision:
        """Decision for the request identity in tools.context (inside a tool call)."""
        return self.decide(_ctx.current_user_id.get(), _ctx.current_channel_id.get(), _ctx.current_is_dm.get())

    # -- tool checks ------------------------------------------------------------

    def check_tool(self, tool_name: str, args: Optional[dict] = None, decision: Optional[Decision] = None) -> Optional[str]:
        """None when the call may proceed, else the message to return instead of running the tool."""
        if tool_name in UNGATED_TOOLS:
            return None
        group = self.policy.group_of(tool_name)
        if group is None:
            return None
        d = decision or self.current_decision()
        groups = self.policy.groups_for(d.role)
        args = {k: v for k, v in (args or {}).items()}
        if group not in groups:
            return self._deny(d, tool_name, "role", self.policy.message(
                "denied_tool", tool=tool_name, role=d.role, role_needed=self.policy.role_needed_for_group(group)))
        confidential = group in self.policy.confidential_groups
        for rule in self.policy.argument_rules:
            if tool_name not in rule["tools"] or not rule["argument"]:
                continue
            val = args.get(rule["argument"])
            if val is None or str(val).lower() not in rule["values"]:
                continue
            need = rule.get("requires_group")
            if need and need not in groups:
                return self._deny(d, tool_name, "argument", self.policy.message(
                    "denied_tool", tool=f"{tool_name}({rule['argument']}={val})", role=d.role,
                    role_needed=self.policy.role_needed_for_group(need)))
            if rule.get("confidential"):
                confidential = True
        if confidential and not d.place_confidential:
            return self._deny(d, tool_name, "place", self.policy.message("denied_place_tool", tool=tool_name))
        if confidential:
            audit.info("allow tool=%s user=%s channel=%s role=%s confidential=1", tool_name, d.user_id, d.channel_id, d.role)
        return None

    def _deny(self, d: Decision, tool_name: str, why: str, message: str) -> str:
        audit.warning("deny tool=%s user=%s channel=%s dm=%s role=%s why=%s", tool_name, d.user_id, d.channel_id, d.is_dm, d.role, why)
        return message

    def gate_tools(self, tools: Iterable[Any]) -> list:
        """Wrap LangChain tools so each call runs ``check_tool`` first. Names, descriptions and
        argument schemas are unchanged, so the model sees the same tool set."""
        out = []
        for t in tools:
            out.append(self._gate(t))
        return out

    def _gate(self, t):
        control = self
        name = t.name
        func, coro = getattr(t, "func", None), getattr(t, "coroutine", None)
        if func is None and coro is None:
            return t

        def sync(*args, **kwargs):
            denial = control.check_tool(name, kwargs)
            if denial:
                return denial
            return func(*args, **kwargs)

        async def asyn(*args, **kwargs):
            denial = control.check_tool(name, kwargs)
            if denial:
                return denial
            return await coro(*args, **kwargs)

        update = {}
        if func is not None:
            update["func"] = sync
        if coro is not None:
            update["coroutine"] = asyn
        return t.model_copy(update=update)

    # -- admin operations (called by access.commands) -----------------------------

    def set_user(self, user_id: str, role: str, by: str) -> None:
        if role not in ROLE_ORDER:
            raise ValueError(f"unknown role {role!r}; choose from {', '.join(ROLE_ORDER)}")
        self.store.put_rule("user", user_id, role, by)
        self.invalidate()
        audit.warning("admin set_user user=%s role=%s by=%s", user_id, role, by)

    def remove_user(self, user_id: str, by: str) -> bool:
        ok = self.store.delete_rule("user", user_id)
        self.invalidate()
        audit.warning("admin remove_user user=%s by=%s removed=%s", user_id, by, ok)
        return ok

    def set_channel(self, channel_id: str, mode: str, by: str) -> None:
        if mode not in CHANNEL_MODES:
            raise ValueError(f"unknown channel mode {mode!r}; choose from {', '.join(CHANNEL_MODES)}")
        self.store.put_rule("channel", channel_id, mode, by)
        self.invalidate()
        audit.warning("admin set_channel channel=%s mode=%s by=%s", channel_id, mode, by)

    def remove_channel(self, channel_id: str, by: str) -> bool:
        ok = self.store.delete_rule("channel", channel_id)
        self.invalidate()
        audit.warning("admin remove_channel channel=%s by=%s removed=%s", channel_id, by, ok)
        return ok

    def set_default_role(self, role: str, by: str) -> None:
        if role not in ROLE_ORDER:
            raise ValueError(f"unknown role {role!r}; choose from {', '.join(ROLE_ORDER)}")
        self.store.put_rule("setting", "default_role", role, by)
        self.invalidate()
        audit.warning("admin set_default_role role=%s by=%s", role, by)

    def set_paused(self, paused: bool, by: str) -> None:
        self.store.put_rule("setting", "paused", "1" if paused else "0", by)
        self.invalidate()
        audit.warning("admin set_paused paused=%s by=%s", paused, by)


# ---------------------------------------------------------------------------
# process-wide instance
# ---------------------------------------------------------------------------

_control: Optional[AccessControl] = None
_control_built = False
_control_lock = threading.Lock()


def get_access_control() -> Optional[AccessControl]:
    """Shared AccessControl, or None when ``ACCESS_CONTROL=off``. Memoised.

    When the store cannot be opened the bot still starts, with an in-memory store: only the
    bootstrap admins and the default role apply, no channels are listed, and an ERROR is logged.
    """
    global _control, _control_built
    if _control_built:
        return _control
    with _control_lock:
        if _control_built:
            return _control
        import os
        if (os.getenv("ACCESS_CONTROL") or "on").strip().lower() in ("off", "0", "false", "no"):
            logger.warning("ACCESS_CONTROL=off: every Slack user gets every tool")
            _control, _control_built = None, True
            return None
        policy = Policy.load(os.getenv("ACCESS_POLICY_PATH") or None)
        try:
            store = open_access_store()
        except AccessStoreUnavailable as e:
            logger.error("access store unavailable (%s); running with bootstrap admins and the default role only", e)
            store = MemoryAccessStore()
        _control = AccessControl(policy, store)
        logger.info("Access control on: policy %s, store %s, default role %s, bootstrap admins %d",
                    policy.path, store.label, _control.default_role, len(policy.bootstrap_admins))
        _control_built = True
        return _control


def set_access_control(control: Optional[AccessControl]) -> None:
    """Install a specific instance (tests) - None disables gating."""
    global _control, _control_built
    with _control_lock:
        _control, _control_built = control, True


def reset_access_control() -> None:
    global _control, _control_built
    with _control_lock:
        _control, _control_built = None, False


__all__ = ["AccessControl", "CHANNEL_MODES", "Decision", "Policy", "ROLE_ORDER", "UNGATED_TOOLS", "get_access_control",
           "reset_access_control", "role_rank", "set_access_control"]
