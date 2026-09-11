# Trading Signals

Z-score anomaly detection on crypto market data, written up by Claude (via Google
Vertex AI), with a terminal chat agent and a Slack bot that call the same tools.

- **Spot data** (price, OHLCV, spot volume): Coin Metrics
- **Derivatives data** (funding rate, perp OI, perp volume, liquidations): Amberdata,
  aggregated over binance, bybit, okx, deribit, bitget, hyperliquid
- **Options data** (DVOL, ATM IV term structure, 25-delta skew, put/call ratios, options
  and block volume, dealer gamma, block trades): Amberdata options analytics, Deribit -
  only for tokens with listed options (btc, eth, sol, hype in the universe; Deribit also lists avax,
  xrp, trx). DVOL exists only for btc and eth. See [Options](#options-deribit-via-amberdata).
- **Signals**: per-metric z-scores against a 30-day rolling median; volume metrics
  separate weekends from weekdays. |z| >= 1.0 is *significant*, |z| >= 2.5 an *outlier*.
  Options metrics (`dvol_close`, `atm_iv_30d`, `pcr_oi`, `options_notional_volume`,
  `options_block_notional_volume`) are z-scored the same way (plain window); `skew_25d_30d`
  and `pcr_volume_24h` are reported as levels with a 7-day change.
- **LLM**: `claude-sonnet-4-6` on Vertex AI Model Garden (project `anchorage-ai-development`, `us-east5`)

## Setup

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt

# GCP credentials (Secret Manager for API keys, Vertex AI for Claude)
gcloud auth application-default login
gcloud auth application-default set-quota-project anchorage-ai-development

cp .env.example .env   # optional; every value has a default or a Secret Manager fallback
```

`.env.example` keys (all optional):

| Key | Default / fallback |
|---|---|
| `COINMETRICS_API_KEY` | Secret Manager `coinmetrics_trial_api` |
| `AMBERDATA_API_KEY` | Secret Manager `amberdata_key` |
| `GCP_SECRETS_PROJECT` | `anchorage-trading-solutions` |
| `VERTEX_PROJECT` / `VERTEX_LOCATION` / `VERTEX_MODEL` | `anchorage-ai-development` / `us-east5` / `claude-sonnet-4-6` |
| `SLACK_BOT_TOKEN` / `SLACK_APP_TOKEN` | Secret Manager `trading_signals_slack_bot_token` / `trading_signals_slack_app_token`; read only by `slack_bot.py` and `--post-slack` |
| `SLACK_CHANNEL_ID` | Secret Manager `trading_signals_slack_channel_id` (no version yet -> `None`); daily-post target |
| `SLACK_BOT_DB` | `data/slack_bot.db` - SQLite conversation memory for the bot |
| `SLACK_SESSION_IDLE_MIN` | `120` - minutes of silence before a top-level mention/DM starts a new conversation |
| `SLACK_SESSIONS_FILE` | `data/slack_sessions.json` - session bookkeeping (who is in which conversation) |
| `SLACK_WEBHOOK_URL` | env only (never Secret Manager); legacy fallback for `--post-slack` |

`Config()` never touches GCP until an API key attribute is read; Vertex uses ADC (no API key).

## Usage

Batch signals report (markdown to stdout). Verified live, Sept 2026:

```bash
python run_signals.py --tokens btc eth --days 45 --verbose --json out/btc_eth.json
                                               # explicit tokens: every token gets a Claude write-up
python run_signals.py                          # test universe (btc eth sol sui hype uni jto): write-ups for |z| >= 1.0 only
python run_signals.py --all                    # write up every token in the universe
python run_signals.py --full-universe --days 45 --json out/full.json   # 28 tokens, ~5 min
python run_signals.py --tokens btc --significant-only
python run_signals.py --post-slack             # post to Slack via the bot (SLACK_CHANNEL_ID) or webhook; warns + exits 0 if neither is set
python run_signals.py --post-slack --slack-channel C0123456789   # override the channel for one run
python run_signals.py --tokens btc --no-options # skip the options feed (6 fewer Amberdata calls per listed token; report says `options: not fetched`)
```

`--days 45` fetches 45 daily rows; the z-score window is 30 days and needs >10 observations.
The current (incomplete) UTC day is always dropped before z-scoring.

Interactive chat agent (LangGraph ReAct over the signal tools):

```bash
python chat.py
# /tokens  list the universe    /reset  clear history    /quit
python chat.py -q "Give me the BTC options snapshot: term structure, skew, put/call and biggest block trades"
python chat.py -q "Is ETH implied vol unusually high vs its 30 day history?"   # uses dvol_close / atm_iv_30d z-scores
```

Tests (offline, no GCP or network):

```bash
python -m pytest -q
```

## Slack bot

`slack_bot.py` is the Slack app **summarize** (Bolt, Socket Mode) with the chat agent as
its brain: the same LangGraph ReAct agent as `chat.py` (`build_chat_agent`, all
`tools/chat_tools.py` tools plus `current_time`), the chat system prompt with a Slack
addendum (`prompts/slack_prompt.md`: mrkdwn only, code-block tables, short replies,
untrusted input), and conversation memory persisted in SQLite via `AsyncSqliteSaver`,
keyed per *session*: a top-level mention or DM continues the sender's current
conversation in that channel until `SLACK_SESSION_IDLE_MIN` (default 120) minutes of
silence; replies inside a thread share that thread's session; saying just `reset` or
`new topic` starts over. Session bookkeeping lives in `data/slack_sessions.json`.
conversation). It answers `@summarize` mentions in channels in the channel itself (set
`SLACK_REPLY_IN_THREAD=1` to thread replies under the question instead); overflow beyond
3900 chars and any follow-up you post under the bot's reply stay in that thread. DMs are
answered top-level.

Flow per message: react :eyes: -> post ":hourglass_flowing_sand: Working on it…" ->
run the agent (240 s cap; full reports are slow) -> `to_mrkdwn()` safety net
(`**bold**`, `# headers`, `- bullets`, `[links](url)`, pipe tables -> ``` block) ->
update the placeholder, overflow beyond 3900 chars threaded. Timeouts and errors
become a one-line :warning: reply and a log entry.

```bash
python slack_bot.py -v                                  # connect to Slack (Socket Mode)
python slack_bot.py --selftest "Is ETH implied vol unusually high vs its 30 day history? Keep it to 3 bullets."
                                                        # real agent + Vertex + market data, fake Slack client, prints the reply; no Slack connection
```

**One listener rule.** Socket Mode delivers every event to every process holding the app
token, and each one replies. Only one instance of the bot may run at a time - stop any
other listener (including the old `~/slackbot/bot.py` prototype) before starting this one.
Use `--selftest` for development; it never opens a socket.

**Daily post.** `python run_signals.py --days 45 --post-slack` posts the report with the
bot token to `SLACK_CHANNEL_ID` (`notifiers.slack.post_via_bot`: mrkdwn, chunked at 3900
chars, first chunk in the channel, the rest threaded). Precedence: bot token + channel id
-> env `SLACK_WEBHOOK_URL` -> `WARNING Slack post skipped: set SLACK_CHANNEL_ID (secret
trading_signals_slack_channel_id) or SLACK_WEBHOOK_URL` and exit 0. The channel-id secret
has no version yet, so today the daily run skips; add a version (the channel id) to turn it
on. `--slack-channel C...` overrides for one run.

**Secrets** (project `anchorage-trading-solutions`, env override in brackets):
`trading_signals_slack_bot_token` [`SLACK_BOT_TOKEN`], `trading_signals_slack_app_token`
[`SLACK_APP_TOKEN`], `trading_signals_slack_channel_id` [`SLACK_CHANNEL_ID`]. Bot scopes:
chat:write, app_mentions:read, im:history/read/write, channels:history, groups:history,
users:read, reactions:read/write, assistant:write (no channels:read / conversations.info,
so the bot never looks channels up by name).

**Deploy.** `Dockerfile` + `.dockerignore` (no secrets baked in) and
[`deploy/cloud-run.md`](deploy/cloud-run.md): Cloud Run service `trading-signals-slack` in
`anchorage-corp-eng-playground` / `us-east1` (1 instance, no ingress, dedicated service
account with `secretmanager.secretAccessor` on the five secrets and `aiplatform.user` on
`anchorage-ai-development`), plus systemd `--user` units for running from this checkout
(`deploy/trading-signals-slack.service`, `trading-signals-daily.service` + `.timer` at
07:00 America/New_York). Nothing in `deploy/` is enabled or deployed yet. SQLite memory is
ephemeral on Cloud Run (`/tmp`); Cloud SQL / Firestore checkpointers are the durable
option later.

Tests: `tests/test_slack_bot.py` (fake Slack client, fake agent, fake LLM + real SQLite
checkpointer; no network).

## Data coverage (full universe, 45 days, 2026-09-10)

Every token in the 28-token universe resolves on both providers except where noted.
Values are the latest complete UTC day.

| Metric | Provider | Coverage | Notes |
|---|---|---|---|
| price | Coin Metrics `PriceUSD` | 28/28 | `PriceUSD` not offered for hype, jto (and several small caps) -> falls back to the daily candle close of the token's primary spot market |
| spot_volume | Coin Metrics candles `candle_usd_volume` | 28/28 | USD, summed over coinbase/binance/kraken/bybit (per-token list in `providers/coinmetrics.py`); BTC ~ $2.3B/day |
| perp_volume | Amberdata open-interest-total `volumeMilUSD` (swaps) | 28/28 | USD, summed over the 6 exchanges |
| perp_oi | Amberdata open-interest-total `usd` (swaps) | 28/28 | USD; BTC ~ $25B |
| funding_rate | Amberdata funding-rates `fundingRateNormalized8h` | 28/28 | annualized %, mean across exchanges of each exchange's USD-margined perp |
| liquidations | Amberdata liquidations-total | 27/28 | **ray**: no liquidation rows on Amberdata (RAY perps only on bitget/okx there) -> `not available` |
| options (dvol, atm_iv, skew, pcr, options volume) | Amberdata options analytics (Deribit) | 4/28 (btc, eth, sol, hype; DVOL only for btc, eth) | see [Options](#options-deribit-via-amberdata) |

Token-level notes: `pol` is Coin Metrics asset `pol` (the old `matic` markets are dead);
`sky` is `sky_sky` (`sky` is Skycoin). Both are mapped in `providers/coinmetrics.py`.

## Options (Deribit via Amberdata)

`providers/amberdata_options.py` (`AmberdataOptionsProvider`, built by
`providers.factory.get_options_provider()`; `None` when there is no Amberdata key) wraps
`https://api.amberdata.com/markets/derivatives/analytics`. `tools.metrics.fetch_token_metrics`
merges its daily series onto the spot/perp frame for every token in
`provider.supported_tokens()`; other tokens get no options columns and the LLM report says
`options: not listed on Deribit`. Pass `include_options=False` / `--no-options` to skip the
feed entirely (the provider is then never constructed).

| Column(s) | Endpoint | Unit | Notes |
|---|---|---|---|
| `dvol_open/high/low/close` | `volatility/index` | vol points (annualised % IV) | with `timeInterval=day` the whole range comes in **one call** (without it the endpoint caps at 1 day). Only `BTC` and `ETH` have a DVOL index; every `*_USDC` currency returns 0 rows -> `not available` |
| `atm_iv_7d/30d/60d/90d/180d`, `ts_richness` | `volatility/term-structures/richness` | vol points | daily history in one call |
| `skew_25d_30d`, `skew_10d_30d` | `volatility/delta-surfaces/constant` | vol points, put IV - call IV (positive = puts richer) | daily history works: `startDate`/`endDate` + `timeInterval=day` returns one 00:00Z surface per day (verified 2026-09-10, 45 rows in one call); the 30d tenor row is used |
| `pcr_oi`, `pcr_volume_24h` | `trades-flow/put-call-ratio` | ratio (>1 = more puts) | daily |
| `options_contract/notional/premium_volume`, `options_block_notional_volume` | `trades-flow/volume-aggregates` | contracts / USD | with `timeInterval=day` one row per day for the whole range in **one call**; on-screen + block summed (block notional kept separately) |
| snapshot: term structure | `volatility/term-structures/forward-volatility/constant` | vol points | chat `get_vol_term_structure` / `get_options_snapshot` |
| snapshot: delta surface / skew by tenor | `volatility/delta-surfaces/constant` | vol points | chat `get_options_snapshot` |
| snapshot: dealer gamma by strike | `trades-flow/gamma-exposures-snapshots` | gamma units per strike | up to 10k rows/day, aggregated by strike; flip point = sign change nearest the index; chat `get_gamma_exposure` |
| snapshot: top block trades | `trades-flow/block-volumes` | contracts, premium USD | chat `get_options_snapshot` |

### Options coverage (Deribit via `instruments/information`, observed 2026-09-10, 45-day run)

Latest complete day 2026-09-09. `supported_tokens()` = avax, btc, eth, hype, sol, trx, xrp.

| token | currency | in universe | dvol_close | atm_iv_30d | skew_25d_30d | pcr_oi / pcr_volume_24h | options_notional_volume | options_block_notional_volume | snapshot (term structure, surface, gamma) | block trades (7d) |
|---|---|---|---|---|---|---|---|---|---|---|
| btc | `BTC` | yes | 40.2 | 38.0 | -0.83 | 0.53 / 0.36 | $1.59B | $722.7M | yes (92 strikes, flip ~$77.5K) | yes |
| eth | `ETH` | yes | 53.7 | 51.2 | -1.53 | 0.55 / 0.65 | $161.8M | $11.1M | yes (95 strikes) | yes |
| sol | `SOL_USDC` | yes | not available | 52.1 | -2.11 | 0.53 / 1.38 | $4.27M | $0 (no block trades) | yes (88 strikes) | yes (tiny) |
| hype | `HYPE_USDC` | yes | not available | 63.3 | -4.59 | 1.36 / 1.19 | $1.30M | $0 | yes (61 strikes) | yes (tiny) |
| avax | `AVAX_USDC` | no (available, not in universe) | not available | 56.4 | -0.91 | 0.45 / 0.86 | $212K | $0 | yes (37 strikes) | none in window |
| xrp | `XRP_USDC` | no (available, not in universe) | not available | 52.1 | -3.22 | 0.57 / 0.25 | $3.96M | $0 | yes (54 strikes) | none in window |
| trx | `TRX_USDC` | no (available, not in universe) | not available | 22.0 | +0.75 | 1.09 / 0.67 | $3.4K | $0 | yes (36 strikes) | none in window |

Notes: every `*_USDC` currency lacks a DVOL index (0 rows) and has essentially no block
trades, so `options_block_notional_volume` is 0 and its z-score is meaningless for those
tokens (constant series -> no z, or a tiny z around 0). `pcr_oi` / `pcr_volume_24h` arrive
rounded to 2 dp, so their z-scores are coarse (often exactly 0 when the value equals the
rolling median). The richness endpoint omits the first day of the requested range (44 rows
for a 45-day window); `trx` had one day without any trade. All other universe tokens have
no listed options on Deribit and report `options: not listed on Deribit`.

API notes:

- **Every options endpoint requires `Accept-Encoding: gzip`** (otherwise HTTP 400 "Compression
  required"); `requests` decodes transparently once the header is sent.
- Currency naming: `BTC`, `ETH`, **`SOL_USDC`** (not `SOL`); `provider.currency_for(token)`
  holds the mapping, other listed currencies come from `instruments/information`.
- Exchange: `deribit` (best coverage). `okex` works for BTC only, `bybit` returns empty.
- **Cost** (measured 2026-09-10 via `provider.call_count`): a 45-day `run_signals.py` run
  makes **6 Amberdata options calls for the first listed token** (1 `instruments/information`
  discovery + 5 daily series: DVOL, richness, delta-surface history, put/call ratio,
  volume-aggregates - each range in a single request thanks to `timeInterval=day`) and **5
  per additional listed token**; unlisted tokens cost 0. A chat `get_options_snapshot` adds 6
  (term structure, delta surface, 14-day DVOL, 14-day PCR, gamma snapshot, 7-day block volumes),
  i.e. 12 for btc daily series + snapshot. Results are memoised per provider instance, so
  repeated calls inside one run are free. No day-by-day looping is needed at 45 days; the
  provider only falls back to a 1-day loop if the API answers 400 "range over the maximum
  allowed". Use `--no-options` when you only need spot/perp signals.
- **403 on this tier**: `volatility/implied-vs-realized` and `trades-flow/top-trades` -
  never called.
- Retries / backoff on 429 and 5xx, WARNING on 401/403, INFO + `None` on empty data, the
  partial current UTC day is dropped - same conventions as the perp provider.

Chat tools: `get_options_snapshot(token)`, `get_vol_term_structure(token)`,
`get_options_flow(token, days)`, `get_gamma_exposure(token)`; `get_zscore_signals` and
`get_token_metrics` include the options columns automatically for listed tokens.

### Restarting the bot safely

Kill by an anchored pattern so the shell running the command does not match itself:

```bash
for pid in $(pgrep -f '^\.venv/bin/python slack_bot\.py'); do kill $pid; done
nohup .venv/bin/python slack_bot.py -v > data/slack_bot.log 2>&1 &
```

### IPv4-first name resolution

`utils/net.py` forces IPv4 lookups (`FORCE_IPV4=0` disables it). The dev container
advertises a global IPv6 address with a broken upstream; Python tries AAAA first and every
Google token refresh, Vertex call and Secret Manager read hangs for the connect timeout,
while curl silently falls back. On this machine `/etc/gai.conf` also carries
`precedence ::ffff:0:0/96 100` so gcloud and bq prefer IPv4 too.

## Desk data (BigQuery)

The chat agent and the Slack bot can also read the Global Markets desk's own book from
BigQuery, read-only: Haruko PnL / greeks / positions (options, futures, perps, spot
balances for **A1 Ltd** - entity 20 - and **ADSD**, Anchorage Digital Swap Dealer - entity
86), the OTC derivatives blotter, live Talos orders and the internal price feed.
Implementation: `providers/bigquery.py` (`DeskBigQuery`), `tools/desk_tools.py`
(`get_desk_tools()`, registered next to `get_chat_tools()` by `chat.default_tools()`),
prompt section "Desk data (BigQuery)" in `prompts/chat_assistant_prompt.md`.

**Access model** (verified 2026-09-10). Data project `anc-global-markets` (us-west1).
The user has `bigquery.tables.getData` on datasets **`brokerage_a1`** (125 objects,
mostly dbt views) and **`pricing`** (28) only, and **no** `bigquery.jobs.create` in that
project - so every job is submitted to and billed by another project
(`BQ_BILLING_PROJECT`, default `anchorage-corp-eng-playground`; `anchorage-ai-development`
also works). Credentials are Application Default Credentials, the same as Vertex.
Table metadata comes from the table API (works with these grants) and is cached for 24 h in
`data/bq_catalog.json`.

| Env var | Default | Meaning |
|---|---|---|
| `BQ_DATA_PROJECT` | `anc-global-markets` | where the tables live |
| `BQ_BILLING_PROJECT` | `anchorage-corp-eng-playground` | jobs run + are billed here |
| `BQ_ALLOWED_DATASETS` | `brokerage_a1,pricing` | the only datasets SQL may reference |
| `BQ_MAX_BYTES_BILLED` | `2000000000` (2 GB) | `maximum_bytes_billed` on every job |
| `BQ_MAX_ROWS` | `200` | `LIMIT` appended / lowered on every query |
| `BQ_TIMEOUT_S` | `60` | query timeout |
| `BQ_CATALOG_PATH` | `data/bq_catalog.json` | schema cache (TTL 24 h) |

**Guards** (`providers.bigquery.validate_sql`, applied before anything reaches BigQuery):
single `SELECT` / `WITH` statement only (DML, DDL, scripting and `;`-separated statements
are refused); every table reference - backticked or not, `project.dataset.table`,
`dataset.table`, `INFORMATION_SCHEMA` views - must be in an allowed dataset of the data
project, otherwise the error lists the allowed datasets; trailing `LIMIT` capped at
`BQ_MAX_ROWS`; per-job cost cap. Denied datasets are never queried, so a bad request costs
nothing. Some Haruko convenience views (`fct_otc_haruko_pnl_by_venue*`, `*_by_strategy*`,
`fct_otc_haruko_pnl_summary`, `fct_otc_haruko_futures_positions`, `fct_otc_haruko_position_pnl`,
`fct_otc_haruko_trades_derivs`) scan 1.5-260 GB and are refused by the cap; the tools use the
partitioned `fct_otc_haruko_pnl_position_history_eod` (a few MB per day) instead.
`fct_perps_positions` is stale (last snapshot 2026-07-10) and is not used.

**Tools** (all return compact markdown with the as-of timestamp and Haruko's
`data_quality_flag` / `valid_pricer_pct` where present):

| Tool | Source | Answers |
|---|---|---|
| `get_desk_risk_snapshot()` | `fct_otc_haruko_pnl_portfolio` (snapshot every ~5 min) | positions/venues/assets, gross & net notional, equity, day/WTD/MTD/QTD/YTD/LTD PnL, funding, fees, delta/gamma/vega/theta USD, risk levels, large-change flags, data quality; per entity + combined |
| `get_desk_pnl_history(days, by)` | `fct_otc_haruko_pnl_portfolio_history_eod` / `..._position_history_eod` | daily EOD PnL series at portfolio level (<= 90 d) or pivoted by strategy / venue (<= 28 d) |
| `get_desk_greeks_history(days)` | `fct_otc_haruko_greeks_history_eod` | daily EOD delta / gamma / vega / theta per entity |
| `get_perp_positions(top_n)` | FUTURES rows of `..._position_history_eod` + live `fct_otc_haruko_position_summary` | perps and dated futures: side, size, notional, avg vs mark, open/day PnL, funding (day, LTD), delta, live qty; maps symbols to tokens for Amberdata cross-checks |
| `get_desk_positions_by_symbol(symbol, top_n)` | `..._position_history_eod` (+ live spot from position summary) | exposure by underlying (or by instrument x venue for one symbol) |
| `get_otc_derivatives_trades(days, top_n)` | `fct_otcderivatives_trades` + `dim_otcderivatives_*` | OTC options blotter with counterparty, strike, premium, expiry, status summary |
| `get_open_orders()` | `fct_a1_talos_open_orders_live` | working Talos orders |
| `get_internal_price(asset, hours)` | `pricing.fct_current_asset_prices_live` + `pricing.intraday_price` | live internal tick and intraday path (USD) |
| `list_desk_tables(keyword)` / `describe_desk_table(table)` | catalog cache | discovery |
| `query_desk_data(sql)` | any allowed table | guarded escape hatch; 40 rows shown, bytes scanned reported |

Symbol mapping heuristic (`tools.desk_tools.desk_symbol_to_token`): drop option / dated-future
tails, keep the part before the first `-` / `/` / `_`, strip a trailing `USDT`/`USDC`/`USD`/`PERP`
when at least three characters remain (`BTC-PERPETUAL`, `BTCUSDT`, `BTC-USDT-SWAP`,
`BTC-25SEP26-80000-C` -> `btc`; `PYUSD` stays `pyusd`).

Greeks are Haruko USD-normalised totals (`total_delta_usd`; `total_gamma_percent_usd` = delta
USD change per 1% spot move; `total_vega` per vol point; `total_theta` per day). As of Sept
2026 both entities carry `data_quality_flag = High Invalid Pricer Rate` (14% / 28% valid
pricers); the tools and the prompt surface that caveat.

**Denied datasets.** `brokerage`, `hold`, `hold_recon`, `crms`, `marketdata`,
`otc_derivatives`, `lending`, `stables_yield`, `external_accounts`, `cryptio`, `reporting`,
`mart`, `intermediate`, `staging` return 403 and the guard refuses them up front. To extend
coverage request `roles/bigquery.dataViewer` on the dataset through the Atlassian service
desk, portal 5, "Data or Role Access Request", then add it to `BQ_ALLOWED_DATASETS`.

**Cloud Run.** The service account in `deploy/cloud-run.md` additionally needs
`roles/bigquery.jobUser` on the billing project (`anchorage-corp-eng-playground`) and
`roles/bigquery.dataViewer` on `anc-global-markets:brokerage_a1` and
`anc-global-markets:pricing` (dataset-level grants, not project-wide). Without them the
desk tools answer "not available" / "access denied" and the market-data tools keep working.

Tests: `tests/test_bigquery_provider.py` (guard, catalog cache with a fake client,
describe formatting) and `tests/test_desk_tools.py` (every tool against canned frames
modelled on real rows) - no GCP, no network.

## Known limitations

- **Coin Metrics trial key**: `ReferenceRateUSD` returns 403 for every asset; `PriceUSD` is
  unsupported for some small caps (verified: hype, jto, pol). Those use the candle close of the
  first market in the token's exchange list, so price is exchange-specific rather than a
  reference rate. Candle requests are one call per market per token (4 per token by default).
- **Amberdata**: `bitget` is excluded from the liquidation aggregate only (its liquidation feed
  is mis-scaled by 20x-30000x vs its OI as of Sept 2026); it still contributes OI, volume and
  funding. `/volumes` and `/liquidations-total` accept at most 31 days per request, so long
  ranges are fetched in 30-day chunks; short windows return sub-daily buckets that are re-summed
  to UTC days. Unknown assets return HTTP 200 with empty data, which becomes `None`.
- **Partial days**: Amberdata returns the current UTC day in progress; `fetch_token_metrics`
  drops it. Coin Metrics daily data ends at the last complete day.
- **Calm markets**: with the default (universe) mode, tokens with all |z| < 1.0 get no LLM call.
  Use `--tokens ...` or `--all` to force a write-up; missing metrics are stated as
  `not available` in the prompt so the model never sees NaN.
- **Slack**: `--post-slack` resolves the channel id / bot token secrets only when passed; the
  webhook is env-only. With neither configured it logs a warning, prints "skipped" and exits 0.
  The bot's thread memory is a local SQLite file; only one Socket Mode listener may run.

## Architecture

```
run_signals.py / chat.py / slack_bot.py      entry points (CLI report, terminal REPL, Slack bot in Socket Mode)
  -> workflows/signals_workflow.py            fetch -> z-scores -> per-token Claude analysis (streams TokenEvent/FinalEvent)
  -> agents/trading_agent.py                  ReAct agent over the @tool functions
       tools/signals.py                       calculate_statistical_signals + signal tools
       tools/metrics.py                       fetch_token_metrics (tz-safe daily join, float64, partial-day drop, options merge), fetch_options_snapshot, z-score math, universes
       tools/chat_tools.py                    @tool wrappers for chat.py incl. the four options tools
         providers/factory.get_provider()     CompositeProvider(spot=CoinMetricsProvider, derivatives=AmberdataProvider)
         providers/factory.get_options_provider()  AmberdataOptionsProvider (Deribit) or None without a key
         providers/base.py                    standard schemas: time, price | spot_volume | funding_rate (annualized %) | perp_oi | ...
  utils/llm.get_llm()                         ChatAnthropicVertex (cached per model/temperature/max_tokens)
  utils/config.Config + utils/secrets         env override -> GCP Secret Manager
  notifiers/slack                             to_mrkdwn / chunk_text, post_via_bot (bot token), post_message (env webhook)
  prompts/*.md                                system, per-token, whole-universe, chat and Slack-addendum prompts
  deploy/                                     Cloud Run notes, systemd units (not enabled)
```
