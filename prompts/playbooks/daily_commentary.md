---
name: daily_commentary
description: The desk's daily crypto market note - price action, flows, options, movers, macro - in a fixed six-paragraph shape
triggers: ["daily commentary", "daily market note", "daily market commentary", "morning note", "market commentary"]
min_role: viewer
---
## Goal

Write the Global Markets daily market commentary: six short paragraphs of prose (no headers,
no bullet lists, no tables, no horizontal rules), hard cap 450 words, in the register of a
sell-side morning note. Start directly with paragraph 1: no preamble such as "here is the note",
no summary of the tools you ran. Every figure comes from a tool call made in this turn. Use only
public market data; do not call desk (Haruko / sheet) tools even if the sender may use them.

## Data to collect first (call all of these, in parallel where possible)

1. `current_time` - the note is dated from this.
2. `get_top_movers(n=120)` - one call gives BTC, ETH, SOL, HYPE levels with T-24h / T-7d changes,
   the leaders and laggards above $1B market cap, and the smaller names moving on volume.
3. `get_crypto_news(limit=15)` for the market-wide headlines of the last 24h, then
   `get_crypto_news(tokens=[...], hours=48, limit=6)` for any token in paragraph 2 or 4 whose move
   needs a catalyst (max 5 tokens per call). Use `get_intel_events` when a move looks
   protocol-driven and the news is thin.
4. `get_zscore_signals(tokens=["btc","eth","sol","hype"])` - funding, open interest, perp and spot
   volume, liquidations with z-scores.
5. `get_cme_curve("btc")` - front-month and next-month annualised basis and CME open interest.
6. `get_etf_flows("bitcoin", days=7)` and `get_btc_etf_onchain_flows(days=7)` - issuer-reported and
   on-chain ETF flows; `get_etf_flows("ethereum", days=7)` when ETH flows matter that day.
7. `get_options_snapshot("btc")` and `get_options_flow("btc", days=3)` - DVOL / ATM IV, skew,
   put-call, block notional and premium, largest prints, gamma flip.
8. `get_treasury_curve(days=30)`, `get_vix_history(days=30)`, and
   `get_macro_snapshot(groups="fx,commodities,equities")` - yields, VIX, dollar, oil, equity closes.

If a tool errors, say what is missing in one clause and move on; never estimate a number.

## Output template (prose only; Slack mrkdwn; bold the tickers on first mention)

Paragraph 1 - one or two sentences on the day's theme: what the market did overall and the one or
two macro / policy / regulatory events behind it, taken from the headlines and macro tools.

Paragraph 2 - the majors: BTC, ETH, SOL and HYPE in that order, each as
"*BTC* (+0.3% T-24h; -2.4% T-7d) has ... near the $76.1k level", with the level rounded sensibly
($76.1k, $2.4k, $100, $79.5). Add one clause of token-specific catalyst where the news supplies one.

Paragraph 3 - positioning and flows: CME basis (front and next month, annualised, versus spot, and
whether it is widening or narrowing over the week), ETF flows (issuer-reported latest published
day and the on-chain read, named as such), funding and open interest direction with the z-score
flags, and liquidations for the last day. Say "not available" for anything the tools do not carry
(taker flow by exchange, Coinbase premium, order-book skew) rather than approximating.

Paragraph 4 - options: BTC block notional and premium, whether the flow skews to calls or puts,
DVOL / ATM 30d level and the term-structure shape, 25-delta skew direction, put-call ratio, the
gamma flip level versus spot, and the one or two largest prints with strike / expiry.

Paragraph 5 - deeper down the market-cap curve: the two to four leaders and the notable laggards
among assets above $1B with T-24h and T-7d moves, then one or two smaller names moving on
unusual volume. Give the catalyst when a headline supplies it.

Paragraph 6 - broader markets: US Treasury yields (2y, 10y, 30y and the 2s10s move, from the
Treasury tool, quoting its date), VIX close and change, dollar index, oil and equity index closes
from FRED with their dates. Index futures are not available: describe yesterday's closes and say
"as of yesterday's close". Mention scheduled central-bank decisions only if a headline reports them.

Close with one italic line: `_Sources: Coin Metrics (Coinbase spot, CME), Amberdata (Deribit,
perps), Messari (news, ETF), US Treasury, CBOE, FRED. Data as of <times from the tools>. Not a
recommendation._`

## Style

- Percentages to one decimal, USD levels rounded, basis and yields with their annualisation or
  date. Every claim about "why" must cite a headline or event from the tools; otherwise say the
  move came without a clear catalyst.
- Neutral, factual register. No exclamation, no advice, no "we think". One or two sentences per
  data point at most: a paragraph should read in twenty seconds. Under 450 words in total.
