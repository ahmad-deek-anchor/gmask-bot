# Deploying the Slack bot to Cloud Run

**Deployed 2026-09-14** as Cloud Run service `trading-signals-slack` (project `anchorage-corp-eng-playground`, region `us-east1`, revision 00001, service account `gm-bot@anchorage-corp-eng-playground.iam.gserviceaccount.com`, entrypoint `python cloudrun_entry.py -v`). `cloudrun_entry.py` adds the `$PORT` health listener that Cloud Run's startup probe requires and then runs the unchanged `slack_bot.run_bot()`. `gm-bot` was granted `secretmanager.secretAccessor` on the three `trading_signals_slack_*` secrets the same day. Still open: `roles/bigquery.dataViewer` on `anc-global-markets:brokerage_a1` / `pricing` for `gm-bot` (desk tools answer "not available" until granted) and A1 Metrics Dashboard sheet access (domain policy blocks sharing with a service account). The text below is the plan that was followed.

Originally: nothing in this document had been executed yet. It is the plan for running
`slack_bot.py` as a single always-on Cloud Run instance, plus a local systemd
alternative and the daily-post timer.

## The one-listener rule (read first)

Slack Socket Mode delivers every event to **every** connected listener that uses the
same app token, and each one replies. Today the old prototype in `~/slackbot/bot.py`
holds that connection. **Stop it before the new bot connects anywhere** (Cloud Run,
systemd, or `python slack_bot.py` in a terminal):

```bash
pkill -f "slackbot/.venv/bin/python bot.py"   # or kill the pid shown by: pgrep -af 'bot.py'
```

While developing, use `python slack_bot.py --selftest "question"` — it runs the real
agent (Vertex + market data) against a fake Slack client and never opens a socket.

## Target

| | |
|---|---|
| Deploy project | `anchorage-corp-eng-playground` (the existing `hello-global-markets` test service lives there) |
| Region | `us-east1` |
| Service | `trading-signals-slack` |
| Secrets project | `anchorage-trading-solutions` |
| Vertex project | `anchorage-ai-development` (`us-east5`, `claude-sonnet-4-6`) |

Socket Mode only opens an **outbound** WebSocket, so the service needs no public
ingress and no unauthenticated access; Cloud Run still requires a container that
does not exit, which `slack_bot.py` satisfies (the Bolt handler blocks forever).
Cloud Run does expect the container to listen on `$PORT` for health checks unless
`--no-cpu-throttling` + min instances keep it alive; if the deploy complains about a
missing listener, add a trivial `aiohttp` health endpoint on `$PORT` (10 lines) in
`slack_bot.py` — leave this until the first deploy proves it is needed.

## 1. Service account and IAM

```bash
DEPLOY_PROJECT=anchorage-corp-eng-playground
SECRETS_PROJECT=anchorage-trading-solutions
VERTEX_PROJECT=anchorage-ai-development
SA_NAME=trading-signals-slack
SA="$SA_NAME@$DEPLOY_PROJECT.iam.gserviceaccount.com"

gcloud iam service-accounts create "$SA_NAME" --project "$DEPLOY_PROJECT" \
  --display-name "Trading signals Slack bot"

# Secret Manager: exactly the five secrets the bot reads, nothing project-wide.
for s in coinmetrics_trial_api amberdata_key \
         trading_signals_slack_bot_token trading_signals_slack_app_token \
         trading_signals_slack_channel_id; do
  gcloud secrets add-iam-policy-binding "$s" --project "$SECRETS_PROJECT" \
    --member "serviceAccount:$SA" --role roles/secretmanager.secretAccessor
done

# Vertex AI (Claude via Model Garden) in the AI project.
gcloud projects add-iam-policy-binding "$VERTEX_PROJECT" \
  --member "serviceAccount:$SA" --role roles/aiplatform.user
```

Notes:
- `trading_signals_slack_channel_id` currently has **no versions**; the bot tolerates
  that (`Config().SLACK_CHANNEL_ID` is `None`) and only the daily post needs it.
- `slack_webhhok_url` is deliberately **not** granted: the webhook path is env-only.
- Cloud Build (used by `--source`) needs the default Cloud Build SA to be able to push
  to Artifact Registry in `$DEPLOY_PROJECT`; that is the project default.

## 2. Deploy

Run from the repo root (`.dockerignore` keeps `.venv`, `data/`, `.env`, tests out of
the image; the `Dockerfile` bakes **no** secrets).

```bash
gcloud run deploy trading-signals-slack \
  --source . \
  --project "$DEPLOY_PROJECT" --region us-east1 \
  --service-account "$SA" \
  --min-instances 1 --max-instances 1 \
  --no-cpu-throttling \
  --ingress internal --no-allow-unauthenticated \
  --memory 1Gi --cpu 1 --timeout 3600 \
  --set-env-vars GCP_SECRETS_PROJECT=anchorage-trading-solutions,VERTEX_PROJECT=anchorage-ai-development,VERTEX_LOCATION=us-east5,SLACK_BOT_DB=/tmp/slack_bot.db
```

Why these flags:
- `--min-instances 1 --max-instances 1`: exactly one Socket Mode listener, always up.
  Two instances would double-reply (same problem as the old prototype).
- `--no-cpu-throttling`: the WebSocket must keep running between requests.
- `--ingress internal --no-allow-unauthenticated`: nothing needs to reach the service.
- `SLACK_BOT_DB=/tmp/slack_bot.db`: `/tmp` is the only writable path on Cloud Run.

Redeploy after a code change with the same command; `--source` rebuilds the image.

Logs: `gcloud run services logs tail trading-signals-slack --project "$DEPLOY_PROJECT" --region us-east1`

## 3. Caveats

- **Conversation memory is ephemeral.** The SQLite checkpoint file lives in the
  instance's `/tmp`; a redeploy or instance restart forgets every thread. Acceptable
  for now. Durable options later: swap `AsyncSqliteSaver` for
  `langgraph-checkpoint-postgres` on Cloud SQL, or a Firestore-backed checkpointer,
  behind the same `build_agent()` context manager.
- **Only one listener at a time.** Stop `~/slackbot/bot.py` (see top) before the
  first Cloud Run revision goes live, and never run `python slack_bot.py` locally
  while the service is up.
- Model/tool calls run inside the request loop with a 240 s cap (`AGENT_TIMEOUT`);
  the daily report is a separate job (below), not the bot.

## 4. Daily signals post

The daily post is `python run_signals.py --days 45 --post-slack`. It posts with the
bot token to `SLACK_CHANNEL_ID` (secret `trading_signals_slack_channel_id`, still to
be created — add a version with the channel id, e.g. `C0BV4T93BT3`) and skips with a
WARNING (exit 0) while that is unset. Options:

- **Cloud Scheduler + Cloud Run Job** (recommended once the bot is on Cloud Run):
  `gcloud run jobs create trading-signals-daily --source . --region us-east1
  --service-account "$SA" --set-env-vars ... --command python --args run_signals.py,--days,45,--post-slack`
  then `gcloud scheduler jobs create http ... --schedule "0 7 * * *" --time-zone America/New_York`.
- **Local systemd timer** (below) while running from this machine.

## 5. Local alternative: systemd --user units

Runs the bot and the daily post from this checkout's `.venv`. Edit the two paths
(`WorkingDirectory`, `ExecStart`) if the repo is not at
`/home/ahmaddeek/program/signals-main`. These are **not enabled** by this repo.

```bash
mkdir -p ~/.config/systemd/user
cp deploy/trading-signals-slack.service deploy/trading-signals-daily.service deploy/trading-signals-daily.timer ~/.config/systemd/user/
systemctl --user daemon-reload

# Bot (stop ~/slackbot/bot.py first!)
systemctl --user enable --now trading-signals-slack.service
journalctl --user -u trading-signals-slack -f

# Daily post at 07:00 America/New_York
systemctl --user enable --now trading-signals-daily.timer
systemctl --user list-timers trading-signals-daily.timer
systemctl --user start trading-signals-daily.service   # run once by hand

# keep user services alive after logout
loginctl enable-linger "$USER"
```
