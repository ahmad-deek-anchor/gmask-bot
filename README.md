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
| `MEMORY_DB_URL` / `MEMORY_EPISODE_TTL_DAYS` / `MEMORY_EMBEDDINGS` | `sqlite:///data/memory.db` / `90` / `vertex` - long-term memory, see [Long-term memory](#long-term-memory) |

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
# /tokens  list the universe    /reset  clear history (summarised into memory)    /memory  what is remembered    /quit
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

## Token universe: curated list plus any Coin Metrics asset

Added 2026-09-16. Two tiers:

- **Curated universe** (`tools.metrics.FULL_TOKEN_UNIVERSE`, ~28 tokens): what the daily
  snapshot job, `run_signals.py` and the full written report cover. Every token here costs
  API calls and model time per day, so this list stays deliberate.
- **On demand**: every chat tool (prices, candles, metrics, z-scores, tape) accepts any asset
  Coin Metrics tracks. `tools.metrics.resolve_token` accepts curated tokens directly and
  otherwise asks `providers.universe.TokenUniverse` (reached through
  `providers.factory.get_universe()`, the Coin Metrics provider's `.universe`); with no
  universe available (tests, fake providers) only the curated list counts.

`TokenUniverse` (`providers/universe.py`): `resolve(symbol)` maps a ticker to the Coin Metrics
asset id (`tao` -> `tao_bittensor`, `sky` -> `sky_sky`, largest market cap wins when several
ids share a prefix); `spot_markets(asset_id)` discovers live spot markets from
`catalog_market_candles_v2` in a fixed exchange preference (Coinbase USD, Kraken USD,
Binance USDT, ...), dropping markets with no candle in 7 days; `top_assets(n, by)` ranks by
`CapMrktEstUSD` or `volume_trusted_spot_usd_1d`, excluding stablecoins, gold tokens and
wrapped / staked duplicates (`STABLES_AND_WRAPPED`) unless asked. The ranking is 6 API
calls (~20 s on our key, ~840 assets), cached for 24 h in memory and in
`data/universe_cache.json`; the Cloud Run entrypoint warms it in a background thread at
start-up. The provider uses curated market lists for curated tokens (no network) and the
universe for everything else.

Tool: `list_top_assets(n=100, by="market_cap", include_stables_and_wrapped=False)`; the
`list_token_universe` output now explains the two tiers. Coverage caveat: spot is near
universal; Amberdata derivatives and Deribit options cover far fewer assets, so an
off-list token can return spot data with empty perp / options columns.

Tests: `tests/test_universe.py` (fake client, no network).

## News and sector classification (Messari)

Added 2026-09-16. `providers/messari.py` wraps Messari's Enterprise API (secret
`messari_api_key2`; the older `messari_api_key` is dead) behind `MessariProvider`, reached
through `providers.factory.get_messari_provider()` (None when no key). It reuses the Amberdata
HTTP plumbing (`providers/_http.py`, now parameterised on header name and retry statuses),
memoises every call with a TTL and caches Messari's ranked asset table (about 4,500 assets
with rank > 0, 7-8 requests, ~4 s) for 24 h in `data/messari_assets_cache.json`
(`MESSARI_CACHE_PATH`); the Cloud Run entrypoint warms it at start-up.

What the key gives us (checked 2026-09-16): the curated news feed (about 420 news-outlet items
a day plus blogs and forums; publish time, source, link, tagged assets, model sentiment), the
asset taxonomy (`sectorV2` - DeFi, Networks, AI, Meme, DePIN, Stablecoins, Gaming, NFTs, CeFi,
Tools, Blockchain Infrastructure, Others; `subSectorV2` ~100 names such as Layer-1, Layer-2,
Decentralized Exchange, Real World Assets, Privacy; `tags`), ETF issuer / asset tables and
time series, Intel events and mindshare signals. Research reports return 403.

Known vendor quirk: filtering the news feed by asset (`assetIds=`) times out (HTTP 524 or a
client timeout) almost every time. `MessariProvider.news()` tries the filtered call once with
a 12 s timeout and then falls back to scanning the unfiltered feed (up to 8 pages of 100) and
matching items by Messari's asset tags or by ticker / project name in the title; the tool says
so in its footer because body-only mentions are missed.

Ticker resolution (`asset_record` / `slug_for`): a small override table (btc, eth, sol, hype,
tao -> `bittensor-0`, sky -> `sky-protocol`, pol -> `polygon-ecosystem-token`), then an exact
symbol (or slug / name) match in the ranked table, then Messari's `search=`. Tickers shared by
several assets (SKY) resolve to the best-ranked one unless the Coin Metrics asset id carries a
qualifier (`sky_sky`, `tao_bittensor`), which the tools pass along when the universe is up.

Tools (`tools/messari_tools.py`, registered after the intraday tools):

| Tool | What it answers |
|---|---|
| `get_crypto_news(tokens=[], hours=24, limit=15, include_blogs=False)` | headlines for up to 5 tokens or the market: UTC time, source, link, tagged assets, Messari sentiment |
| `classify_tokens(tokens)` | Messari sector / sub-sector / tags and rank per ticker |
| `get_sector_members(sector, n=25)` | constituents of a sector, sub-sector or tag ("DePIN", "Layer-2", "Proof-of-Work") by rank |
| `list_crypto_sectors()` | the taxonomy: sectors, sub-sectors, counts, largest tickers |
| `get_etf_overview(asset="bitcoin")` | crypto ETF AUM / latest flow / product counts per underlying (bitcoin, ethereum, solana, xrp, multi-asset), regional split, top issuers (Blockworks data) |
| `get_etf_flows(asset="bitcoin", days=30)` | daily issuer-reported spot / futures ETF flows, AUM and regional flows; bitcoin rows also show the Coin Metrics on-chain net flow |
| `get_intel_events(tokens, days=30, importance="")` | Messari Intel events (upgrades, governance, listings, legal, hacks) with importance / status / link |
| `get_social_signals(tokens)` | mindshare (share of tracked influencer attention), sentiment, post counts, Messari's generated insight |

ETF endpoints (confirmed 2026-09-16): `/metrics/v2/protocols/etfs` (issuers), `/metrics/v2/protocols/etfs/assets`
(per underlying, latest), `/metrics/v2/protocols/etfs/assets/{asset}/metrics/overview/time-series/1d` and
`/metrics/v2/protocols/etfs/{provider}/metrics/overview/time-series/1d` (both need `start` and `end`).
Metric keys are normalised to snake_case (`typeSpotUnitedstatesFlowUsd` -> `us_spot_flow_usd`). Flows for
the latest day arrive as null until Blockworks publishes; the tools print "n/a", never 0. Intel events
use `GET /intel/v1/events?primaryOrSecondaryAssets=a,b&startTime=...`; signals `GET /signal/v1/assets?assetIds=a,b`.

`list_top_assets` now carries Messari's sector / sub-sector per ticker when the Messari provider is up
(`TokenUniverse.top_assets(classifier=...)`, Coin Metrics asset ids passed for collision-safe resolution).

Traditional-finance classifications (GICS) and equity / bond / commodity prices are not
covered by Messari; see the FRED section for the macro series we do have.

Tests: `tests/test_messari.py` (fake session, no network).

## CME crypto futures and BTC ETF on-chain flows (Coin Metrics)

Added 2026-09-16. The Coin Metrics key already covers CME: `reference_data_markets(exchange="cme",
type="future")` lists every contract (115 active on 2026-09-16 across BTC, ETH, SOL, XRP and
CME's newer listings), daily / hourly candles carry USD volume, ticks are live, and per-contract
open interest is published once a day at 21:00 UTC. There is no CME mark or index price on the
key, and a contract that is listed but has never traded raises "market not supported" on the
candle endpoint (the provider batches, then retries per market and drops it).

Symbols: `<product><month code><year>` - BTCV6 = Oct 2026 5-BTC contract, MBTV6 = micro
(0.1 BTC), ETHV6 (50 ETH) / METV6 (0.1 ETH), SOLV6 (500 SOL) / MSLV6 (25 SOL, booked under
base `msl`), XRPV6 (50,000 XRP) / MXPV6 (2,500 XRP, base `mxp`); BFF<mdd> are the weekly
Bitcoin Friday futures (0.02 BTC) and hyphenated symbols are calendar spreads.

Provider (`providers/coinmetrics.py`): `cme_contracts(base, include_micro, include_weekly)`,
`cme_curve(base, include_micro)` (last daily close, USD volume, OI in contracts / USD / coins,
basis = close / spot last trade - 1, annualised x 365 / days, all against Coinbase spot and
labelled as such), `cme_history(base, contract=None, days)` (OI / volume per day, summed over
every active outright or for one contract, with Coin Metrics' all-venue futures OI for the CME
share), `etf_onchain_flows("btc", days, frequency)` (`FlowInEtfUSD` / `FlowOutEtfUSD` at 1d or
1h and `SplyEtfNtv` / `SplyEtfUSD` at 1d; BTC only - the metrics are inferred from ETF-labelled
addresses and lag issuer reports by about a day).

Tools (`tools/cme_tools.py`, registered after the Messari tools):

| Tool | What it answers |
|---|---|
| `get_cme_curve(token, include_micro=False)` | the curve: per contract close, basis and annualised basis vs spot, volume, OI; total OI and curve shape |
| `get_cme_open_interest(token, days=30, contract="")` | OI / volume history for all active outrights (with CME share of all-venue OI) or one contract |
| `get_btc_etf_onchain_flows(days=30, hourly=False)` | BTC ETF in / out / net flows and ETF-held supply inferred on-chain, daily or hourly |

Tests: `tests/test_cme.py` (fake client, no network).

## Macro data (FRED)

Added 2026-09-16. Traditional-finance data is FRED only for now (decision 2026-09-16): the Coin
Metrics key is not entitled to its FMP / Databento equity and futures feeds and Stooq blocks API
use, so equity / ETF tickers, FX pairs, commodity futures curves and GICS sector data wait for a
vendor. `providers/fred.py` (`FredProvider`, key from secret `fred_api_key` or env `FRED_API_KEY`;
create a free key at https://fred.stlouisfed.org/docs/api/api_key.html) wraps the FRED JSON API
(120 requests / minute, `api_key` as a query parameter that is never logged) with retries and a
15-minute memo. `MACRO_SERIES` is the curated dashboard: DGS2 / DGS10 / DGS30, T10Y2Y, DFII10,
SOFR, FEDFUNDS, DTWEXBGS (broad dollar), DCOILWTICO / DCOILBRENTEU, SP500 / NASDAQCOM / DJIA,
VIXCLS, BAMLH0A0HYM2 / BAMLC0A0CM (HY / IG OAS), T5YIE / T10YIE, CPIAUCSL, UNRATE. FRED's LBMA gold
series was discontinued, so gold is not on the dashboard (search_fred finds alternatives).

Two free official feeds that need no key sit beside FRED (`providers/official_macro.py`, added
2026-09-17): the **US Treasury daily par yield curve** (Treasury's Atom XML, one month per call,
published the same evening, so a day fresher than FRED's DGS series) and **CBOE's VIX history
file** (daily OHLC since 1990). Both are memoised 30 minutes. Google Finance and Yahoo have no
licensed API and Stooq blocks programmatic use; index futures and a live dollar index still
need a paid vendor (Databento was considered and deferred on 2026-09-17).

Tools (`tools/macro_tools.py`, registered after the CME tools):

| Tool | What it answers |
|---|---|
| `get_macro_snapshot(groups="")` | the dashboard by group (rates, fx, commodities, equities, volatility, credit, inflation, labour) with observation date and change vs prior print / 1w / 1m (bp for rates and spreads) |
| `get_fred_series(series_id, days=90)` | one series' history with title, units, frequency, latest value, window change, high / low |
| `search_fred(text, limit=10)` | find a series id by keywords, sorted by FRED popularity |
| `get_treasury_curve(days=30)` | the Treasury's own curve as of the latest close: every tenor with 1d / 1w / 1m changes in bp, 2s10s / 3m10y / 5s30s slopes |
| `get_vix_history(days=30)` | CBOE's own VIX closes: latest, changes, window range, one-year percentile |

Every value is an observation with a date (one-day publication lag, business days only); the
prompt tells the model to quote it as such and never as a live quote.

Tests: `tests/test_fred.py` (fake session, no network).

## Live and intraday prices (Coin Metrics market data)

Added 2026-09-16. The Coin Metrics key has full, undelayed access to the market-level
endpoints: 1-minute candles land about a minute after the bar closes, market trades and
top-of-book quotes are live. Asset-level reference rates (`ReferenceRateUSD`) are forbidden
on the key at every frequency, so "live price" means the token's primary spot market
(Coinbase USD first, then the fallbacks in `providers/coinmetrics.py`), not an index. The
tools say which market and timestamp every number came from.

Provider (`providers/coinmetrics.py`): `get_intraday_candles(token, frequency, lookback_minutes)`
(1m, 5m, 10m, 15m, 30m, 1h, 4h; max one week), `get_latest_trade(token)`, `get_latest_quote(token)`,
`get_recent_trades(token, minutes)` (max 60 minutes). Tools (`tools/intraday_tools.py`,
registered by `chat.default_tools()` right after the market-data tools):

| Tool | What it answers |
|---|---|
| `get_live_price(token)` | "what is BTC trading at": last trade, bid/ask + spread, last 1m bar, 1h and 24h change, 24h high/low/volume on the market |
| `get_intraday_candles(token, frequency="5m", lookback_minutes=180)` | intraday bars with open/high/low/last/change/volume/VWAP; tables longer than 48 bars print the summary plus the last 48 |
| `get_recent_trades(token, minutes=5, min_trade_usd=50000)` | the tape: count, taker buy vs sell notional, VWAP, largest prints |

Tests: `tests/test_intraday_tools.py` (fake client, no network). Live check 2026-09-16 13:25 UTC:
BTC last trade 3 s old, quote 8 s old, 1m bar 1 min old; a full `get_live_price` call takes
~0.3 s warm (about 5 s on the first call while the client initialises).

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

### Operations: restarts, timeouts, watchdogs

**Restart = `kill <pid>` (SIGTERM).** The bot handles SIGTERM/SIGINT gracefully: it stops
accepting events (anything arriving meanwhile gets ":warning: I am restarting right now —
please ask again in a minute."), closes the Socket Mode connection, waits up to
`SLACK_SHUTDOWN_GRACE_S` (default 20 s) for in-flight requests to finish and post their
answers, cancels the rest, replaces every remaining ":hourglass_flowing_sand: Working on it…"
placeholder with ":warning: I was restarted before finishing this request — please ask
again." and exits 0. Kill by an anchored pattern so the shell running the command does not
match itself; never `kill -9` unless the process is actually hung (then the next start-up
cleans up, see below):

```bash
for pid in $(pgrep -f '^\.venv/bin/python slack_bot\.py'); do kill -TERM $pid; done
sleep 5
nohup .venv/bin/python slack_bot.py -v > data/slack_bot.log 2>&1 &
pgrep -af '^\.venv/bin/python slack_bot\.py'    # exactly one pid
```

`deploy/trading-signals-slack.service` uses `KillSignal=SIGTERM` and `TimeoutStopSec=30`
(grace 20 s + margin), so `systemctl --user restart` takes the same path.

**In-flight file.** `data/slack_inflight.json` (env `SLACK_INFLIGHT_FILE`) holds every
placeholder the bot has posted but not yet replaced: `{"<channel>:<ts>": {channel, ts, user,
thread_ts, started}}`. An entry is written when the placeholder is posted and removed when
the final `chat_update` succeeds. It is the single source of truth for cleanup - the bot
never scans channel history. At start-up (`startup sweep: ...` log line) whatever is left
from a crash or `kill -9` is updated with the restart warning; entries older than 24 h are
dropped without a Slack call. `--selftest` uses an in-memory copy and never touches the file.

**Start-up order** (all logged): stale sessions -> startup sweep -> `memory store ready in
X s (embeddings: on|keyword only)` -> `Bolt app is running`. The memory store (SQLite open
plus the Vertex embeddings probe) is initialised in a worker thread *before* the socket
connects, so no request ever runs it - the 2026-09-11 incident was a first-use embeddings
probe evaluated on the event loop, which froze the bot with a placeholder dangling for
45 minutes and defeated the request timeout (the loop that would have raised it was the
one stuck).

**Timeouts** (all overridable by env):

| Layer | Default | Env | Behaviour when hit |
|---|---|---|---|
| LLM call (`utils/llm.py`) | 120 s, 1 retry | `LLM_TIMEOUT_S`, `LLM_MAX_RETRIES` | anthropic `APITimeoutError`; the agent turn fails -> ":warning: Something went wrong" |
| Embeddings call (`providers/memory_store.py`) | 20 s | `MEMORY_EMBED_TIMEOUT_S` | that call falls back to keyword search; 3 consecutive failures disable embeddings for 10 min (logged once), then retried |
| Request (`slack_bot.handle`) | 240 s | - | ":warning: That took too long and I gave up…" |
| Failsafe watchdog | 240 + 30 s | - | if the handler is *still* running (e.g. `chat_update` hung) a fresh Slack client with a 15 s cap forces the timeout text onto the placeholder, the stuck task's stack is logged at ERROR and the task is cancelled |
| Shutdown grace | 20 s | `SLACK_SHUTDOWN_GRACE_S` | remaining requests cancelled, placeholders marked as interrupted |

The request deadline only works because `answer()` awaits nothing blocking: model calls
are async, tools run in LangGraph's executor threads, and memory recall / episode writes
resolve the store *inside* `asyncio.to_thread`. Keep it that way - any sync network call
added to the loop can freeze the bot again. Every embeddings call runs on a daemon thread
with a hard deadline, so a hung Vertex call can never block its caller (loop or tool
thread) beyond 20 s.

**Loop-lag watchdog.** A background task sleeps 2 s in a loop and logs
`WARNING event loop stalled for X s` whenever it wakes more than 3 s late - the signature
of blocking code on the loop. Every reply logs its wall time (`replied ... in 9.0s`).
If you see stalls, find the sync call and move it to a thread.

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
| `BQ_MAX_BYTES_BILLED` | `20000000000` (20 GB) | `maximum_bytes_billed` on every job. Raised from 2 GB on 2026-09-11 so Carson Levy's EOW derivatives PnL script (`get_derivs_pnl_eod`, ~15 GB per run, ~$0.08) can run live; 20 GB is ~$0.13 worst case per query at $6.25/TB |
| `BQ_MAX_ROWS` | `200` | `LIMIT` appended / lowered on every query |
| `BQ_TIMEOUT_S` | `60` | query timeout |
| `BQ_CATALOG_PATH` | `data/bq_catalog.json` | schema cache (TTL 24 h) |

**Guards** (`providers.bigquery.validate_sql`, applied before anything reaches BigQuery):
single `SELECT` / `WITH` statement only (DML, DDL, scripting and `;`-separated statements
are refused; the one exception is `DeskBigQuery.query_script` / `validate_declare_script`,
used only by project code for `sql/haruko_eod_pnl.sql`: a script whose statements are
`DECLARE name TYPE DEFAULT <plain literal>;` followed by exactly one `SELECT` / `WITH`, with the
query part going through the same `validate_sql` - never reachable from `query_desk_data`); every table reference - backticked or not, `project.dataset.table`,
`dataset.table`, `INFORMATION_SCHEMA` views - must be in an allowed dataset of the data
project, otherwise the error lists the allowed datasets; trailing `LIMIT` capped at
`BQ_MAX_ROWS`; per-job cost cap. Denied datasets are never queried, so a bad request costs
nothing. Some Haruko convenience views (`fct_otc_haruko_pnl_by_venue*`, `*_by_strategy*`,
`fct_otc_haruko_pnl_summary`, `fct_otc_haruko_futures_positions`, `fct_otc_haruko_position_pnl`,
`fct_otc_haruko_trades_derivs`) scan 1.5-260 GB, the largest are refused by the cap; the tools use the
partitioned `fct_otc_haruko_pnl_position_history_eod` (a few MB per day) instead.
`fct_perps_positions` is stale (last snapshot 2026-07-10) and is not used.

**Tools** (all return compact markdown with the as-of timestamp and Haruko's
`data_quality_flag` / `valid_pricer_pct` where present):

| Tool | Source | Answers |
|---|---|---|
| `get_desk_risk_snapshot()` | `fct_otc_haruko_pnl_portfolio` (snapshot every ~5 min) | positions/venues/assets, gross & net notional, equity, day/WTD/MTD/QTD/YTD/LTD PnL, funding, fees, delta/gamma/vega/theta USD, risk levels, large-change flags, data quality; per entity + combined |
| `get_derivs_pnl_eod(period, include_daily)` | `fct_otc_haruko_position_pnl_history` + `fct_otcderivatives_trades` via `sql/haruko_eod_pnl.sql` (Carson Levy / Nayshil Dalal's EOW query, run live by `providers/haruko_eod.py`) | **authoritative derivatives PnL by period** - `mtd`, `wtd`, `ytd` (Haruko YTD column and LTD change since first snapshot, both), `last_week`, `last_month`, `month:YYYY-MM`, `range:A..B`: 3pm America/Chicago EOD cut, one full-book snapshot/day, PnL = life-to-date differences (never Haruko's month_to_date column, which resets mid-month); monthly table, optional daily rows, staleness warning, coverage note; ~15 GB / ~45 s cold, in-process cache 15 min + BigQuery query cache. August 2026 = $2,174,523, reconciled with the EOW report 2026-09-11 |
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
describe formatting), `tests/test_desk_tools.py` (every tool against canned frames
modelled on real rows) and `tests/test_haruko_eod.py` (the EOW PnL maths against the
2026-06-28..2026-09-10 series, DECLARE-script guard, TTL cache) - no GCP, no network.

## Spot desk PnL (Google Sheet)

The chat agent and the Slack bot can also read the spot desk's own PnL dashboard, the Google
Sheet **"A1 Metrics Dashboard"** (id `1BksNxC2QXHLjFJNCuv-GC9JOBHeb8EyoGwzTuNwqNSY`, owner
Joao Luis, maintained daily; each dashboard tab has a "Data as of" cell). It is the **booked
PnL of the spot business** - **HOLD** (Anchorage's client spot trading; PnL = trading
commissions on client trades, per the sheet's `hold db` feed) vs **A1** (A1 Ltd, the
principal spot desk; weekly PnL split into realised and an "unrealized PNL approximation") -
and therefore a different source from the Haruko mark-to-market PnL of the derivatives book in
BigQuery. The tools and the prompt say which source a number came from. Implementation:
`providers/gsheets.py` (`A1MetricsSheet`, `get_a1_metrics_sheet()` in `providers/factory.py`),
`tools/sheet_tools.py` (`get_sheet_tools()`, registered after the desk tools by
`chat.default_tools()`), prompt section "Spot desk PnL (A1 Metrics Dashboard sheet)".

**Access.** Sheets REST API v4 called directly with `requests` (no Google API client library)
using Application Default Credentials - the same user ADC as Vertex / BigQuery, but the token
must carry the Sheets scope. Log in once with exactly these scopes:

```bash
gcloud auth application-default login --scopes=https://www.googleapis.com/auth/cloud-platform,https://www.googleapis.com/auth/drive.file,https://www.googleapis.com/auth/presentations.readonly,https://www.googleapis.com/auth/spreadsheets.readonly,https://www.googleapis.com/auth/drive.readonly
```

Every request carries `x-goog-user-project: anchorage-corp-eng-playground` (the ADC quota
project; `GSHEETS_QUOTA_PROJECT`). A 403 means the ADC token lacks `spreadsheets.readonly`
(re-run the login above) or the account cannot view the sheet; a 404 means the sheet is not
shared with the account - the tools relay both as plain messages. Google warns at login that
`spreadsheets.readonly` / `drive.readonly` **will soon be blocked for gcloud's default OAuth
client id**. Mitigations: (a) register your own OAuth client id and pass it with
`gcloud auth application-default login --client-id-file=client_secret.json --scopes=...`;
(b) for Cloud Run, use a service account - `gm-bot@anchorage-corp-eng-playground.iam.gserviceaccount.com`
exists, but the sheet **cannot be shared with it** under the current Workspace domain policy,
so it needs domain-wide delegation (impersonate a user with view access) or a policy exception.
Nothing in this module writes to the sheet.

| Env var | Default | Meaning |
|---|---|---|
| `A1_METRICS_SHEET_ID` | `1BksNxC2QXHLjFJNCuv-GC9JOBHeb8EyoGwzTuNwqNSY` | spreadsheet id |
| `GSHEETS_QUOTA_PROJECT` | `anchorage-corp-eng-playground` | `x-goog-user-project` on every call |
| `GSHEETS_CACHE_TTL_S` | `300` | in-process cache per range / tab list |

**Tabs used** (layouts observed 2026-09-11; parsers locate header rows by content because the
tabs are hand-built with merged group headers and blank spacer columns):

| Tab | Layout | Tool |
|---|---|---|
| `Volume & PNL` (and `2025 Volume & PNL`) | row 1 `Data as of`; row 2 HOLD / A1 / TOTAL group labels; row 3 `Month | Volume | PNL | Take Rate` per group (TOTAL adds cumulative volume / PnL, target PnL, % of target); months 1-12; `Total YTD` row. Take rate is stored in bps | `get_spot_pnl_summary(year=None)` |
| `Weekly PNL` | HOLD `Week | PNL`; A1 `Week | Realized PNL | Unrealized PNL Approximation | Total`; `A1 + HOLD` `Week | PNL | Start Date | End Date` (Fri-Thu weeks); `Total` row; future weeks are 0 and dropped | `get_weekly_spot_pnl(weeks=8)` |
| `Financing Fees` | `Month | HOLD Financing Fees | HOLD Delta Sales | Total` + a weekly block; no as-of cell (the tool prints the dashboard's) | `get_financing_fees(months=6)` |
| `Nonclient PNL` | only a title and a link to a separate "HOLD PNL" spreadsheet the agent cannot read | `get_nonclient_pnl(months=6)` (says so) |
| `db` | trade blotter `Date (UTC) | Counterparty | Side | Symbol | Buy QTY | Buy Asset | Sell QTY | Sell Asset | Price | PNL | Currency | bps | Month` | `get_counterparty_pnl(days=30, top_n=15, counterparty=None, by="counterparty"|"symbol"|"side")` |
| `A1 database` (cols R:X only) | row 1 headers `Client Flow PNL | Non Client Flow PNL | Change in Total PNL | Change in Client Flow PNL | Change in non Client Flow PNL` in T:X (R1:S9 is an unrelated asset list); rows 2-224 undated history; from row 225 (2025-07-24) R = snapshot datetime `YYYY-MM-DD HH:MM:SS`, S/T/U = cumulative YTD total / client / non-client realised PnL (reset Jan 1), V/W/X = that day's realised total / client-flow / non-client-flow PnL. Some dates missing (e.g. 2026-09-04), one date with two snapshots, offsetting artefact pairs (e.g. 2025-08-14/15 +/-$12.0M) that the weekly tab includes net | `get_a1_client_flow_split(start_date, end_date=None, include_daily=True)` - window sums with % split, per-day table (monthly subtotals for long windows), missing days, artefact disclosure (a pair inside the window is kept and netted; a lone leg is excluded), cross-check vs `Weekly PNL` A1 Realized when the window is a dashboard week and vs the cumulative S/T/U columns for YTD; shortcuts `last week`, `this week`, `mtd`, `ytd`, `last month`, `last N days` resolved against the data-as-of date. Verified 2026-09-11: Sep 4-10 2026 (week 36) V $296,548 = W $68,187 + X $228,361, weekly tab $296,548 |
| any | raw cells, capped at 200 rows x 30 columns | `list_a1_dashboard_tabs()`, `read_a1_dashboard_range(tab, a1_range)` |

All tools print the tab's "Data as of" date and the source line, format USD with `$` and
thousands separators and bps to two decimals.

**Caveats.** The trade-level blotter (`db`) stops at **2024-12-31**; the fuller `Trades` tab
ends 2025-05-22 and has no PnL column, `Dealer Trades` / `Exchange Trades` are broken
`IMPORTRANGE`s and `Counterparty Trades` is a hand summary - so there is **no trade-level
counterparty PnL for 2025-2026 in this sheet**. `get_counterparty_pnl` counts `days` back from
the latest trade in the blotter and prints the coverage window with a caveat; current totals
come from the monthly / weekly tabs. The Weekly and Monthly tabs are updated by hand and can
differ slightly (2026 YTD: weekly total $13.93M vs monthly $13.62M on 2026-09-11 - the weekly
A1 figure includes the unrealised approximation). Verified live 2026-09-11: `Total YTD` PnL
cell `'Volume & PNL'!N16` = 13,622,125.72 = sum of the monthly TOTAL PnL cells = HOLD
5,019,699.27 + A1 8,602,426.45, matching `get_spot_pnl_summary`.

Tests: `tests/test_gsheets.py` (parsers on canned `values` payloads modelled on the real
rows, auth headers, 401 refresh, 403 / 404 / 429 messages, cache TTL, factory) and
`tests/test_sheet_tools.py` (every tool's markdown, unavailable / error paths, `by` variants,
the range cap, the client-flow split: shortcuts, artefacts, missing days, cross-checks) - no
network, no GCP.

## Threaded answers and the asker mention (Slack)

Since 2026-09-16 the Cloud Run service runs with `SLACK_REPLY_IN_THREAD=1`: a channel answer is
posted as a reply under the question (follow-ups inside that thread continue the conversation;
DMs stay flat), and the first chunk of every channel reply starts with `<@asker>` so the answer
shows up in the asker's Activity list (`SLACK_TAG_ASKER`, default on; DMs never tagged).
Access-control replies ("not enabled here", command results) follow the same rules.

## Access control (who may ask the bot what)

Added 2026-09-16 (`access/`). Every Slack message is checked before the agent runs, and every
tool call is re-checked when it executes, against a two-part policy:

- **Static** (`access/policy.yaml`, loaded at start-up): the roles and what they may use
  (`viewer` = public market data, live prices, news / sectors, ETF, CME, macro, public snapshots,
  personal memory; `desk` = + the full written report, Haruko risk / PnL / positions, the spot PnL
  sheet incl. counterparty PnL; `lead` = + raw BigQuery discovery and SQL, shared memory, channel
  rules; `admin` = everything), the tool groups (every registered tool is in exactly one group;
  `tests/test_access.py` fails when a new tool is left out), the confidential groups, argument
  rules (`remember(scope="shared")` needs lead; snapshot sources `haruko` / `sheet` need desk and a
  confidential place), the denial messages, DM behaviour and the bootstrap admins.
- **Dynamic** (the access store: BigQuery `gmask_bot.access` on Cloud Run, sqlite `data/access.db`
  locally, chosen by `ACCESS_BACKEND` which defaults to the snapshot backend): which users hold
  which role, which channels the bot answers in and whether they are confidential, the default
  role for unlisted users (viewer, decision 2026-09-16) and the pause switch. Rules are cached for
  `refresh_seconds` (60) and invalidated immediately by admin commands.

Admins manage it from Slack, no deploy needed (these messages never reach the model):

```
@GM Bot access help
@GM Bot access whoami                        anyone: your role and this channel's status
@GM Bot access list                          lead / admin
@GM Bot access add user @someone desk        roles: viewer, desk, lead, admin, none
@GM Bot access remove user @someone
@GM Bot access add channel here              enable the bot in this channel (public tools)
@GM Bot access add channel here confidential  ... and allow positions / PnL here
@GM Bot access remove channel here | #name
@GM Bot access default viewer                role for unlisted users
@GM Bot access pause | access resume         everyone but admins
```

Semantics: a channel that has not been added gets a one-line "not enabled here" reply (admins are
exempt so they can enable it); DMs are open to everyone with at least the viewer role and count as
confidential places for desk-role users; confidential tools return "Not permitted ..." to the
model in public channels, which the prompt tells it to relay. Denied and confidential calls are
logged under `access.audit` (Cloud Logging). `ACCESS_CONTROL=off` disables gating (local
experiments; the test suite runs with it off except `tests/test_access.py`). Not built yet: Slack
user-group sync, per-user quotas, the BigQuery audit table.

## Long-term memory

The chat agent and the Slack bot share a small persistent memory (`providers/memory_store.py`,
`tools/memory_tools.py`, `tools/context.py`), separate from the per-conversation checkpoints.

**What is stored, where.** One SQLite file, `data/memory.db` by default (`MEMORY_DB_URL`,
e.g. `sqlite:///data/memory.db`; `postgresql://...` is accepted for Cloud Run and needs
`pip install 'psycopg[binary]'`, otherwise a clear `NotImplementedError`). Table `memories`
(`id, namespace, key, text, meta JSON, created_at, updated_at, expires_at, embedding BLOB`) with
four namespaces, plus the `snapshots` table used by `snapshot_daily.py` when `SNAPSHOT_BACKEND=sqlite`
(the default; production uses the shared BigQuery table, see [Data snapshots](#data-snapshots)):

| Namespace | Content | Written by | TTL |
|---|---|---|---|
| `facts:shared` | desk facts everyone should know ("take rate is quoted in bps", data quirks) | `remember(text, scope="shared")` | none |
| `prefs:<slack_user_id>` | one user's preferences (units, format, tokens followed); terminal user is `local` | `remember(text, scope="me")` | none |
| `rules:<channel_id>` | the channel's single standing instruction (replace on set) | `set_channel_rule(text)` / `clear_channel_rule()` - channels only, refused in DMs | none |
| `episodes:<slack_user_id>` | 2-4 sentence summaries of past conversations (who asked, topics, numbers with as-of dates, follow-ups) | automatic, see below | `MEMORY_EPISODE_TTL_DAYS` (default 90; `purge_expired()`) |

**Recall.** Before every turn `tools.memory_tools.build_context(user, channel, text)` searches
`facts:shared` + `prefs:<user>` + `episodes:<user>` + `rules:<channel>` and renders at most ~1200
chars as a `<memories>` block ("Standing instructions for this channel: ..." first, verbatim, then
"Relevant memories (may be stale): ..."). The block is appended to the **system prompt of that model
call only** (a langchain `dynamic_prompt` middleware in `chat.build_chat_agent` reads
`tools.context.current_memory_context`), so the checkpointed messages stay clean and the
episodic summaries never see recalled memories. Another user's preferences or episodes are never
searched. Search is semantic when embeddings work (`langchain_google_vertexai.VertexAIEmbeddings`,
`text-embedding-005`, project `anchorage-ai-development`, `us-central1`; vectors stored as float32
bytes, cosine in Python) and degrades to BM25-style keyword scoring after one WARNING when the
probe fails or `MEMORY_EMBEDDINGS=off`.

**Identity.** Tools never receive a user id from the model: `slack_bot.handle` and `chat.ChatSession.ask`
set the contextvars `tools.context.current_user_id / current_channel_id / current_is_dm` around the
agent call (LangGraph copies the context into the tool and model nodes). In a DM `remember` defaults
to `scope="me"` unless the text clearly asks for a shared/desk fact.

**Tools** (`chat.default_tools()` registers them after the sheet tools): `remember`, `recall(query, k)`,
`forget(query_or_id, scope)` (id or id prefix, matching text, or `"everything"` to wipe the user's
prefs + episodes), `what_do_you_remember()` (the user's prefs and episodes with ids and dates, the
shared-fact count, the channel rule - never other users' data), `set_channel_rule`, `clear_channel_rule`.
Prompt rules (`prompts/chat_assistant_prompt.md`, "Long-term memory"; Slack addendum in
`prompts/slack_prompt.md`): store durable facts and preferences only, never positions / PnL / prices
/ client names / credentials; confirm what was stored or deleted. A regex guard refuses anything that
looks like a token or key (Slack `xox*`, `sk-`, AWS, Google, GitHub, JWT, PEM, `password=`, long
hex / base64 blobs) and warns - but stores - when the text mentions positions / PnL or a dollar amount
above $1M.

**Episodic memory.** The Slack bot summarises a session with `get_llm(max_tokens=300)` into
`episodes:<user>` (meta: channel, session key, started, ended, replies) when the session closes -
idle timeout detected on the next message, `reset`, or stale sessions found at start-up
(`SessionStore.expire_stale`) - and after every 6th reply. Summaries run as background asyncio tasks
(`slack_bot.spawn_episode`, `flush_background`), so replies are never delayed; failures are logged.
`SessionStore` tracks how many replies of each session are already summarised
(`episodes` in `data/slack_sessions.json`). The terminal chat summarises on `/reset` and `/quit`;
`/memory` prints `what_do_you_remember`.

**How to forget.** Say "forget <id or text>" (ids appear in `what_do_you_remember` / `recall`) or
"forget everything about me" (deletes `prefs:<user>` and `episodes:<user>`; shared facts stay).
Operators: `python -c "from providers.factory import get_memory_store as g; s=g(); print(s.delete_namespace('prefs:U123'))"`
or delete `data/memory.db`.

```bash
python chat.py                                    # "remember that I prefer bps not percent" -> stored in prefs:local
python chat.py -q "what do you remember about me?"
python slack_bot.py --selftest "what do you remember about me?"   # identity USELFTEST / CSELFTEST
```

| Env var | Default | Meaning |
|---|---|---|
| `MEMORY_DB_URL` | `sqlite:///data/memory.db` | store location; `postgresql://` needs psycopg |
| `MEMORY_EPISODE_TTL_DAYS` | `90` | episode expiry |
| `MEMORY_EMBEDDINGS` | `vertex` | `vertex` (text-embedding-005 + keyword fallback) or `off` |
| `MEMORY_EMBED_MODEL` / `MEMORY_EMBED_PROJECT` / `MEMORY_EMBED_LOCATION` | `text-embedding-005` / `anchorage-ai-development` / `us-central1` | embedding endpoint |

**Cloud Run note.** The SQLite file lives on the instance's disk and is lost on redeploy (and not
shared across instances); point `MEMORY_DB_URL` at Cloud SQL (`postgresql://...`, add `psycopg[binary]`
to requirements) or mount a persistent volume. The service account needs `aiplatform.user` on
`anchorage-ai-development` for embeddings; without it the store logs one warning and uses keyword search.

Tests: `tests/test_memory_store.py` (put/search/list/delete/TTL, namespace isolation, keyword fallback,
snapshot upsert idempotency and series, backend selection) and `tests/test_memory_tools.py` (every tool,
DM scoping, channel-rule replace/clear and injection text, recall block cap, guard regexes, summariser
with a fake LLM); `tests/test_chat.py` and `tests/test_slack_bot.py` cover the injection into the system
prompt only, identity contextvars inside the graph, and episodes after the 6th reply / reset / idle
timeout / start-up. No network, no Vertex.

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
- **Memory**: recall is best-effort (top 6 items, ~1200 chars) and the memories block is advisory -
  the model may still ignore a preference. Episode summaries cost one small Vertex call per closed
  session / 6 replies. Embeddings need Vertex access in `us-central1`; otherwise keyword search only.

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
       tools/memory_tools.py                  remember / recall / forget / channel rules + build_context, episode summariser
       tools/context.py                       contextvars: current_user_id / current_channel_id / current_is_dm / current_memory_context
         providers/factory.get_memory_store() MemoryStore (providers/memory_store.py): memories in data/memory.db; snapshots via SNAPSHOT_BACKEND
           providers/snapshot_bq.py             BigQuerySnapshotBackend: MERGE/SELECT on gmask_bot.snapshots (written daily by the Cloud Run job)
  utils/llm.get_llm()                         ChatAnthropicVertex (cached per model/temperature/max_tokens)
  utils/config.Config + utils/secrets         env override -> GCP Secret Manager
  notifiers/slack                             to_mrkdwn / chunk_text, post_via_bot (bot token), post_message (env webhook)
  prompts/*.md                                system, per-token, whole-universe, chat and Slack-addendum prompts
  deploy/                                     Cloud Run notes, deploy_snapshot_job.sh (Cloud Run job + Scheduler), systemd units
```

## Data snapshots

`snapshot_daily.py` records one row per (date, source, entity, metric) so the chat agent and the
Slack bot can answer "how has X moved since ..." from our own history (`tools/snapshot_tools.py`).
Rows are upserted on that key, so re-running a day is a no-op apart from `captured_at`.

### Architecture (since 2026-09-11)

```
Cloud Scheduler trading-signals-snapshot-daily  (23:30 UTC, POST jobs.run as gm-bot)
  -> Cloud Run job trading-signals-snapshot      python snapshot_daily.py --sources signals,etf,cme --verbose
       Coin Metrics + Amberdata + Messari (keys from Secret Manager)  ~85 s signals (28 tokens, 370 rows)
       + etf (~90 rows) + cme (~200 rows)
         -> MERGE into BigQuery anchorage-corp-eng-playground.gmask_bot.snapshots
                                                    ^
local systemd timer (23:30 UTC, best effort)        |   same table, same MERGE
  python snapshot_daily.py --sources haruko,sheet --+
                                                    |
Slack bot / chat.py (SNAPSHOT_BACKEND=bigquery) ----+   parameterised SELECTs, date-filtered
```

The developer machine sleeps, so the market-data source runs in Google Cloud; the two sources that
need the user's own credentials stay on the local timer: the service account **cannot** read
`anc-global-markets` (Haruko tables) and **cannot** be shared the A1 Metrics Google Sheet (domain
policy), so `haruko` and `sheet` are captured whenever the machine is awake at 23:30 UTC.

**Storage switch** (`utils/config.py`, `.env`): `SNAPSHOT_BACKEND=sqlite` (default; the `snapshots`
table inside `MEMORY_DB_URL`) or `SNAPSHOT_BACKEND=bigquery` (`providers/snapshot_bq.py`; table
`SNAPSHOT_BQ_TABLE`, default above; jobs run and are billed in `BQ_BILLING_PROJECT`). Memories always stay
in the local SQLite file; only the snapshot methods of `MemoryStore` are delegated, with identical
return shapes on both backends. The store logs `Snapshot backend: bigquery (...)` at start-up.

Table: `snapshot_date DATE, source, entity, metric STRING, value FLOAT64, value_json STRING,
captured_at TIMESTAMP`, partitioned by `snapshot_date`, clustered by `(source, entity, metric)`.
Writes are one parameterised `MERGE ... USING UNNEST(@rows)` per batch of 500 (duplicates inside a batch
collapse first); reads filter on `snapshot_date` wherever a window is known so scans stay tiny.
`scripts/migrate_snapshots_to_bq.py` copied the first local day (431 rows) into the table once.

```bash
python snapshot_daily.py                          # all three sources for today's UTC date (into the backend from .env)
python snapshot_daily.py --sources haruko,sheet   # what the local timer runs
python snapshot_daily.py --sources signals,etf,cme -v   # what the Cloud Run job runs
python snapshot_daily.py --date 2026-09-10        # store under another date (values are still "now")
python snapshot_daily.py --dry-run -v             # print every row, write nothing
python scripts/migrate_snapshots_to_bq.py --dry-run   # count local sqlite rows that would be MERGEd
```

Exit code is non-zero only when every requested source fails; one failing source never blocks
the others. Each source logs its row count, upstream call count and duration.

| Source | Where it runs | Entities | Metrics |
|---|---|---|---|
| `haruko` - latest `fct_otc_haruko_pnl_portfolio` row per entity (same query shape as `get_desk_risk_snapshot`) | local timer | `20` (A1 Ltd), `86` (ADSD), `combined` (sum) | `delta_usd`, `delta_adjusted_usd`, `gamma_usd`, `gamma_pct_usd`, `vega`, `theta`, `day_pnl`, `ytd_pnl`, `gross_notional`, `equity`, `valid_pricer_pct` (combined = pricer-count weighted), `data_quality_flag` (value 1 = Normal / 0 = flagged; `value_json` carries the flag text and as-of time) |
| `signals` - `fetch_token_metrics(FULL_TOKEN_UNIVERSE, 45 days)` -> `calculate_statistical_signals` | Cloud Run job | one per token (`btc`, `eth`, ...) | `<m>` (latest value) and `<m>_z` (z-score) for `spot_volume`, `perp_volume`, `perp_oi`, `total_liquidations` and, where listed, `dvol_close`, `atm_iv_30d`, `pcr_oi`, `options_notional_volume`, `options_block_notional_volume`; `skew_25d_30d` / `pcr_volume_24h` plus `_chg7d`; `price`, `price_pct_change_1d`, `funding_rate` (annualised %). Values are as of the last complete UTC day (`value_json.as_of`). |
| `sheet` - A1 Metrics Dashboard (`monthly_volume_pnl`, `weekly_pnl`) | local timer | `HOLD`, `A1`, `TOTAL` | `mtd_volume_usd`, `mtd_pnl_usd`, `mtd_take_rate_bps` (latest populated month <= current), `ytd_volume_usd`, `ytd_pnl_usd`, `ytd_take_rate_bps`, `ytd_target_pnl_usd` / `ytd_pct_of_target` (TOTAL), `week_pnl_usd` (latest week), `week_realized_pnl_usd` / `week_unrealized_pnl_usd` (A1) |
| `etf` - Messari `etf_assets()` (Blockworks) + Coin Metrics `etf_onchain_flows("btc")` | Cloud Run job | `bitcoin`, `ethereum`, `solana`, `xrp`, `multi-asset` ... | `spot_aum_usd`, `spot_flow_usd`, `spot_products`, `futures_aum_usd`, `futures_flow_usd`, `futures_products`, `us_/europe_/apac_spot_aum_usd`, `us_/europe_/apac_spot_flow_usd`, `us_spot_products`, `deltaone_aum_usd`, `leveraged_aum_usd`, `total_volume_usd` (null flows skipped: not yet published); `bitcoin` also `onchain_flow_in_usd`, `onchain_flow_out_usd`, `onchain_net_flow_usd`, `etf_supply_btc`, `etf_supply_usd` (`value_json.vendor` names the source) |
| `cme` - Coin Metrics `cme_curve(base, include_micro=True)` for btc, eth, sol, xrp | Cloud Run job | contract symbols (`BTCZ6`, `MBTV6` ...) and the underlying (`btc`) | per contract `close`, `oi_contracts`, `oi_usd`, `volume_usd`, `basis_ann_pct`, `days_to_expiry`; per underlying `cme_oi_usd`, `cme_volume_usd`, `front_basis_ann_pct`, `next_basis_ann_pct`, `spot_ref` (`value_json`: front / next contract, spot market and time, basis convention) |

Chat tools (`tools/snapshot_tools.py`, read-only, registered through `chat.default_tools()`):

- `get_snapshot_history(source, metric, entity=None, days=30)` - date | value table with first -> last
  change and min / max. Default entity: `combined` (haruko), `TOTAL` (sheet), `bitcoin` (etf), `btc` (cme);
  `signals` needs the token; `cme` also takes a contract symbol.
  Haruko entity aliases: `a1` -> `20`, `adsd` -> `86`.
- `compare_to_snapshot(source, metric, entity, date)` - latest vs the snapshot on that date (falls back to
  the closest earlier one, up to 31 days, and says so).
- `list_snapshot_metrics(source="")` - captured entities / metrics and the latest snapshot date per source.

### Cloud job: deploy, schedule, inspect

`deploy/deploy_snapshot_job.sh` is idempotent - run it after any code change (Cloud Build takes 3-5 min):

```bash
deploy/deploy_snapshot_job.sh                 # gcloud run jobs deploy --source . + scheduler create/update
gcloud run jobs execute trading-signals-snapshot --region us-east1 --project anchorage-corp-eng-playground --wait
gcloud run jobs executions list --job trading-signals-snapshot --region us-east1 --project anchorage-corp-eng-playground
gcloud logging read 'resource.type="cloud_run_job" AND resource.labels.job_name="trading-signals-snapshot"' \
  --project anchorage-corp-eng-playground --freshness 1d --limit 100 --order asc --format 'value(timestamp,textPayload)'
gcloud scheduler jobs describe trading-signals-snapshot-daily --location us-east1 --project anchorage-corp-eng-playground
bq --project_id=anchorage-corp-eng-playground query --use_legacy_sql=false \
  'SELECT source, COUNT(*) n, MAX(captured_at) FROM `anchorage-corp-eng-playground.gmask_bot.snapshots`
   WHERE snapshot_date = CURRENT_DATE() GROUP BY source'
```

| | |
|---|---|
| Job | `trading-signals-snapshot`, `us-east1`, 1 task, 1 vCPU / 1 GiB, 20 min timeout, 1 retry; image built from the repo `Dockerfile` by Cloud Build (`--source .`), command `python snapshot_daily.py --sources signals --verbose` |
| Env | `SNAPSHOT_BACKEND=bigquery`, `SNAPSHOT_BQ_TABLE`, `GCP_SECRETS_PROJECT=anchorage-trading-solutions`, `BQ_BILLING_PROJECT=anchorage-corp-eng-playground`, `VERTEX_PROJECT=anchorage-ai-development`, `MEMORY_EMBEDDINGS=off`, `FORCE_IPV4=0` |
| Scheduler | `trading-signals-snapshot-daily`: cron `30 23 * * *` (Etc/UTC), `POST https://run.googleapis.com/v2/projects/anchorage-corp-eng-playground/locations/us-east1/jobs/trading-signals-snapshot:run`, OAuth token of the job's service account (the Admin API is a Google API, so OAuth rather than an OIDC identity token), 30 min attempt deadline |
| Service account | `gm-bot@anchorage-corp-eng-playground.iam.gserviceaccount.com`: `secretmanager.secretAccessor` on `amberdata_key` + `coinmetrics_trial_api` (anchorage-trading-solutions), `bigquery.jobUser` on the project, `bigquery.dataEditor` on dataset `gmask_bot`, `aiplatform.user` on anchorage-ai-development, `run.invoker` on the project (lets the scheduler call `jobs.run`). It has **no** access to `anc-global-markets` or the Google Sheet. |
| First cloud run | 2026-09-11 17:43 UTC (`trading-signals-snapshot-lpcc6`): 28 tokens, 370 rows in 84.8 s, exit 0 |

Costs are negligible: one ~90 s run per day of a 1 vCPU container (well inside the Cloud Run free tier),
Cloud Scheduler's first three jobs are free, and every BigQuery query touches a few MB of a table that
grows by ~400 rows a day (the 10 MB minimum per query applies; a month of bot usage is cents).

Local timer: `deploy/trading-signals-snapshot.service` + `.timer` (installed in
`~/.config/systemd/user/`, 23:30 UTC) run `--sources haruko,sheet` and read `.env`, so they write to the
same BigQuery table. Haruko's own EOD row lands at ~23:55 UTC, so the snapshot holds the latest intraday
portfolio row of the day. First live run 2026-09-11 (all three sources locally, then migrated).
