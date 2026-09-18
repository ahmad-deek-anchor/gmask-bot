-- Haruko end-of-day derivatives PnL series - the desk's end-of-week (EOW) report method.
--
-- Authors of the method and query: Carson Levy (Global Markets) with Nayshil Dalal.
-- Copied verbatim (apart from this header) into the bot on 2026-09-11 so that the
-- bot's Haruko PnL matches the EOW report; run live by providers/haruko_eod.py via
-- DeskBigQuery.query_script (read-only guard, dataset allow-list, 30 GB cost cap).
--
-- Method:
--   * Source: anc-global-markets.brokerage_a1.fct_otc_haruko_position_pnl_history
--     (position-level Haruko snapshots). Positions are restricted to the desk's portfolios
--     (desk_strategies, '|'-separated strategy_name values: Derivs Risk, ADSD, AD Hedge Co -
--     added to Carson's query on 2026-09-18 per Ahmad Deek; see providers/desk_scope.py),
--     combined, no per-portfolio split.
--   * Uniform EOD cut at eod_time in eod_zone (15:00 America/Chicago). Only full-book
--     snapshots (> 100 position rows) qualify; per day the one snapshot nearest 15:00 CT
--     within max_skew_secs (600 s) is chosen.
--   * Positions are de-duplicated on `key`, preferring is_eod_snapshot rows.
--   * ltd_pnl = SUM(life_to_date_pnl) of the chosen snapshot. daily_pnl = change in
--     ltd_pnl versus the previous EOD row; monthly / period PnL = LTD difference between
--     the period's last EOD row and the last EOD row before the period. ytd_pnl_check =
--     LTD change since the first available row. Haruko's own month_to_date_pnl /
--     week_to_date_pnl columns are summed for reference only (they reset mid-period).
--   * Trades (fct_otcderivatives_trades, excluding CANCELED/VOIDED/STAGED and novation
--     backs) are bucketed to the same EOD date (exec_time + 9h in eod_zone) for notional.
--   * Scans ~21 GB (about $0.13, ~45 s; the strategy_name column adds ~6 GB); one row per EOD date, newest first.
--
-- The DECLARE defaults are the only parameters; providers/haruko_eod.py substitutes them
-- by literal replacement after validation (never by string formatting of user input).
DECLARE eod_time     TIME    DEFAULT TIME '15:00:00';
DECLARE eod_zone     STRING  DEFAULT 'America/Chicago';
DECLARE ytd_base_date DATE    DEFAULT DATE '2025-12-31';
DECLARE max_skew_secs INT64   DEFAULT 600;
DECLARE desk_strategies STRING DEFAULT 'Derivs Risk|ADSD|AD Hedge Co';
WITH snapshots AS (
  SELECT snapshot_timestamp AS ts
  FROM `anc-global-markets.brokerage_a1.fct_otc_haruko_position_pnl_history`
  WHERE snapshot_timestamp >= TIMESTAMP(DATETIME(ytd_base_date, eod_time), eod_zone)
  GROUP BY 1 HAVING COUNT(*) > 100),
candidates AS (
  SELECT ts, DATE(ts, eod_zone) AS eod_date,
    TIMESTAMP_DIFF(ts, TIMESTAMP(DATETIME(DATE(ts, eod_zone), eod_time), eod_zone), SECOND) AS skew_secs
  FROM snapshots),
chosen AS (
  SELECT eod_date, ts, skew_secs FROM (
    SELECT eod_date, ts, skew_secs, ROW_NUMBER() OVER (PARTITION BY eod_date ORDER BY ABS(skew_secs)) AS rn
    FROM candidates WHERE ABS(skew_secs) <= max_skew_secs) WHERE rn = 1),
deduped_positions AS (
  SELECT c.eod_date, c.skew_secs, h.life_to_date_pnl, h.year_to_date_pnl, h.month_to_date_pnl, h.week_to_date_pnl,
    ROW_NUMBER() OVER (PARTITION BY c.eod_date, h.key ORDER BY h.is_eod_snapshot DESC) AS rk
  FROM chosen c JOIN `anc-global-markets.brokerage_a1.fct_otc_haruko_position_pnl_history` h ON h.snapshot_timestamp = c.ts
  WHERE h.strategy_name IN UNNEST(SPLIT(desk_strategies, '|'))),
pnl_daily AS (
  SELECT eod_date, ANY_VALUE(skew_secs) AS skew_secs, COUNT(*) AS n_positions,
    SUM(life_to_date_pnl) AS ltd_pnl, SUM(year_to_date_pnl) AS ytd_pnl, SUM(month_to_date_pnl) AS mtd_pnl, SUM(week_to_date_pnl) AS wtd_pnl
  FROM deduped_positions WHERE rk = 1 GROUP BY 1),
trades_daily AS (
  SELECT DATE(TIMESTAMP_ADD(exec_time, INTERVAL 9 HOUR), eod_zone) AS eod_date,
    SUM(SAFE_CAST(quantity_total_quote AS FLOAT64)) AS notional_quote, COUNT(*) AS n_trades
  FROM `anc-global-markets.brokerage_a1.fct_otcderivatives_trades`
  WHERE (status NOT IN ('CANCELED','VOIDED','STAGED') OR status IS NULL)
    AND exec_time >= TIMESTAMP(DATETIME(ytd_base_date, eod_time), eod_zone)
    AND COALESCE(is_novation_back, FALSE) = FALSE GROUP BY 1),
settled_joined AS (
  SELECT p.eod_date, TRUE AS is_final, p.skew_secs, p.n_positions, p.ytd_pnl, p.mtd_pnl, p.wtd_pnl, p.ltd_pnl,
    COALESCE(t.notional_quote, 0) AS notional_quote, COALESCE(t.n_trades, 0) AS n_trades
  FROM pnl_daily p LEFT JOIN trades_daily t ON p.eod_date = t.eod_date)
SELECT eod_date, is_final, skew_secs, n_positions, ytd_pnl, mtd_pnl, wtd_pnl, ltd_pnl,
  ltd_pnl - FIRST_VALUE(ltd_pnl) OVER (ORDER BY eod_date) AS ytd_pnl_check,
  ltd_pnl - LAG(ltd_pnl) OVER (ORDER BY eod_date) AS daily_pnl,
  ltd_pnl - LAG(ltd_pnl, 7) OVER (ORDER BY eod_date) AS trailing_7d_pnl,
  notional_quote, n_trades
FROM settled_joined WHERE eod_date > ytd_base_date ORDER BY eod_date DESC;
