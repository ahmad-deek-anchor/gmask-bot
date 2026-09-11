"""Outbound notifiers. Only Slack (incoming webhook) is kept; it is opt-in."""

from notifiers.slack import post_message

__all__ = ["post_message"]
