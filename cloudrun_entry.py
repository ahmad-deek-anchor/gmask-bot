#!/usr/bin/env python
"""Cloud Run entrypoint for the Slack bot.

Cloud Run services must accept TCP connections on $PORT or the revision fails its startup
probe. Socket Mode itself needs no inbound HTTP, so this wrapper starts a trivial aiohttp
health listener on $PORT and then runs the unchanged ``slack_bot.run_bot()`` in the same
event loop. Locally keep using ``python slack_bot.py``; this file is only for the container.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys

from aiohttp import web

import slack_bot

log = logging.getLogger("slack_bot")


async def _serve_health() -> web.AppRunner:
    port = int(os.environ.get("PORT", "8080"))
    app = web.Application()

    async def ok(_request):
        return web.Response(text="ok")

    app.router.add_get("/", ok)
    app.router.add_get("/healthz", ok)
    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, "0.0.0.0", port).start()
    log.info("health endpoint listening on :%d", port)
    return runner


async def _main() -> int:
    runner = await _serve_health()
    try:
        return await slack_bot.run_bot()
    finally:
        await runner.cleanup()


if __name__ == "__main__":
    args = slack_bot.parse_args()
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    for noisy in ("httpx", "urllib3", "google", "cm_client"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    log.setLevel(logging.INFO)
    try:
        sys.exit(asyncio.run(_main()))
    except KeyboardInterrupt:
        sys.exit(130)
