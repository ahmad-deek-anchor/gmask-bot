"""providers/haruko_eod.py + tools.desk_tools.get_derivs_pnl_eod: Carson Levy's EOW PnL method, offline.

Fixture: the real 2026-06-28..2026-09-10 series produced by sql/haruko_eod_pnl.sql on
2026-09-11 (values rounded to cents). Known answers: August 2026 = $2,174,523 (matches the
EOW report), September MTD at 2026-09-10 = $553,538, YTD column = $10,772,786.
No GCP, no network.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from providers import haruko_eod as he
from providers.bigquery import DeskBigQuery, SQLGuardError, validate_declare_script
from providers.haruko_eod import HarukoEodError, HarukoEodPnl, business_days_between, substitute_declares
from tools import desk_tools as dt

PROJECT = "anc-global-markets"
ALLOWED = ("brokerage_a1", "pricing")

# eod_date, skew_secs, n_positions, ytd_pnl, mtd_pnl, wtd_pnl, ltd_pnl, notional_quote, n_trades
ROWS = [
    ("2026-06-28", -4, 2014, 7697034.85, 3187736.25, 839038.30, 10525002.09, 28123500, 8),
    ("2026-06-29", 28, 2015, 7708804.78, 3199506.18, 11984.43, 10536772.03, 0, 0),
    ("2026-06-30", 30, 2022, 7753628.90, 31064.10, 61108.55, 10581596.15, 4895900, 6),
    ("2026-07-01", -41, 2035, 7852384.45, 48779.52, 159864.10, 10680351.69, 147547000, 16),
    ("2026-07-02", -40, 2045, 7857405.27, 53800.34, 164884.92, 10685372.51, 117519900, 25),
    ("2026-07-03", -40, 2052, 7825260.43, 21655.50, 132740.08, 10653227.67, 129098975, 29),
    ("2026-07-04", -41, 2059, 7896305.31, 92700.38, 203784.96, 10724272.55, 67000000, 12),
    ("2026-07-05", -40, 2062, 7842640.54, 39035.61, 150120.19, 10670607.78, 126555000, 24),
    ("2026-07-06", 16, 2067, 7791043.91, -12561.02, -52006.16, 10619011.16, 190490000, 36),
    ("2026-07-07", 83, 2068, 7805971.40, 2366.47, -37078.67, 10633938.65, 157800000, 30),
    ("2026-07-08", -38, 2075, 7727404.26, -76200.67, -115645.81, 10555371.51, 3884000, 3),
    ("2026-07-09", -35, 2086, 7748465.22, -55139.71, -94584.86, 10576432.46, 5016955, 6),
    ("2026-07-10", -35, 2087, 7669669.47, -133935.46, -173380.61, 10497636.71, 0, 0),
    ("2026-07-11", -35, 2096, 7707125.97, -96478.95, -135924.10, 10535093.22, 69850000, 20),
    ("2026-07-12", -32, 2097, 7690617.85, -112987.08, -152432.22, 10518585.10, 34925000, 10),
    ("2026-07-13", -35, 2102, 7700945.75, -102659.18, 11177.54, 10528912.99, 145635000, 32),
    ("2026-07-14", -34, 2109, 7761767.31, -41837.62, 71999.11, 10589734.56, 212960000, 44),
    ("2026-07-15", -33, 2109, 7886753.49, 80508.48, 196015.00, 10714720.73, 180080000, 12),
    ("2026-07-16", -93, 2118, 7890877.96, 84632.95, 200139.47, 10718845.21, 103513680, 21),
    ("2026-07-17", -92, 2127, 7890931.25, 84686.24, 200192.76, 10718898.50, 111860000, 24),
    ("2026-07-18", -93, 2128, 7924629.92, 118384.91, 233891.43, 10752597.16, 93050000, 18),
    ("2026-07-19", -93, 2131, 7891962.47, 85717.46, 201223.98, 10719929.72, 1075000, 3),
    ("2026-07-20", 98, 2132, 7886448.76, 80203.75, -6047.60, 10714416.01, 0, 0),
    ("2026-07-21", 100, 2149, 8059319.43, 253074.42, 166823.06, 10887286.67, 250676000, 26),
    ("2026-07-22", 69, 2159, 8121799.36, 315554.35, 229302.99, 10949766.61, 103341552, 13),
    ("2026-07-23", 70, 2170, 8126153.43, 319908.42, 233657.06, 10954120.68, 224960000, 21),
    ("2026-07-24", -65, 2173, 8021384.75, 215139.74, 128888.38, 10849351.99, 3938055, 1),
    ("2026-07-25", -71, 2173, 8018287.99, 212042.98, 125791.62, 10846255.24, 0, 0),
    ("2026-07-26", -71, 2173, 8017382.30, 211137.29, 124885.93, 10845349.54, 0, 0),
    ("2026-07-27", 0, 2176, 7993224.20, 186979.19, -24533.13, 10821191.45, 0, 0),
    ("2026-07-28", 0, 2223, 7964061.12, 157816.11, -53696.20, 10792028.37, 30310000, 6),
    ("2026-07-29", 0, 2224, 7975336.32, 169091.31, -42421.01, 10803303.57, 91500000, 18),
    ("2026-07-30", 0, 2240, 8010985.73, 204740.72, -6771.59, 10838952.98, 156134140, 33),
    ("2026-07-31", 0, 2249, 8044724.74, 238479.73, 26967.41, 10872691.98, 79360000, 18),
    ("2026-08-01", 0, 2251, 8069352.52, 24627.78, 51595.19, 10897319.76, 60110000, 12),
    ("2026-08-02", 0, 2252, 8102145.51, 57420.77, 84388.18, 10930112.75, 114000000, 10),
    ("2026-08-03", 0, 2258, 8088652.41, 43927.67, -13493.10, 10916619.65, 34575000, 6),
    ("2026-08-04", 2, 2268, 8117922.41, 73197.67, 15776.90, 10945889.65, 93516400, 15),
    ("2026-08-05", 1, 2276, 8165971.80, 121247.06, 63826.29, 10993939.04, 91373000, 18),
    ("2026-08-06", -91, 2278, 8225040.08, 180315.34, 122894.57, 11053007.32, 75592250, 13),
    ("2026-08-07", 44, 2282, 8232126.06, 187401.32, 129980.55, 11060093.30, 32104998, 7),
    ("2026-08-08", 45, 2291, 8284937.68, 240212.94, 182792.17, 11112904.92, 49143200, 10),
    ("2026-08-09", 60, 2291, 8301387.87, 256663.13, -29851.09, 11129355.11, 0, 0),
    ("2026-08-10", 47, 2295, 8340460.50, 295735.77, 37762.56, 11168427.75, 71405000, 15),
    ("2026-08-11", -26, 2299, 8321300.48, 276575.75, 18602.54, 11149267.73, 31680000, 4),
    ("2026-08-12", -58, 2308, 8398747.04, 354022.31, 96049.10, 11226714.29, 131631475, 20),
    ("2026-08-13", 28, 2316, 8523155.90, 478431.16, 220457.95, 11351123.14, 219164765, 29),
    ("2026-08-14", 30, 2322, 8572163.73, 527439.00, 269465.79, 11400130.98, 141449000, 12),
    ("2026-08-15", 30, 2330, 8636469.59, 591744.85, 333771.64, 11464436.83, 104215000, 21),
    ("2026-08-16", -90, 2330, 8671026.63, 626301.89, 368328.69, 11498993.87, 75620000, 14),
    ("2026-08-17", 117, 2337, 8725509.08, 680784.34, 54456.19, 11553476.32, 93750000, 9),
    ("2026-08-18", 117, 2348, 8776320.82, 731596.08, 106306.46, 11604288.07, 162314325, 28),
    ("2026-08-19", -105, 2361, 9215414.30, 1170689.57, 545399.94, 12043381.55, 98980000, 12),
    ("2026-08-20", -5, 2392, 10170677.62, 2125952.89, 1500663.85, 12998644.87, 537479215, 41),
    ("2026-08-21", 13, 2425, 10230745.23, 2186020.49, 1560731.45, 13058712.47, 444034000, 47),
    ("2026-08-22", 102, 2435, 10377372.57, 2332647.84, 1707358.80, 13205339.82, 54266000, 10),
    ("2026-08-23", -147, 2437, 10379323.22, 2334598.48, 1709309.44, 13207290.46, 30200000, 4),
    ("2026-08-24", 69, 2451, 10239245.05, 2194520.31, -140013.77, 13067212.29, 128050050, 13),
    ("2026-08-25", 58, 2472, 10349820.21, 2305095.48, -29438.60, 13177787.46, 214774006, 18),
    ("2026-08-26", -78, 2475, 10334752.67, 2290027.93, -44506.15, 13162719.91, 8300000, 3),
    ("2026-08-27", -94, 2491, 10489808.01, 2445083.28, 110549.20, 13317775.26, 74680000, 11),
    ("2026-08-28", -34, 2513, 10215177.62, 2170452.89, -164081.20, 13043144.87, 374823000, 44),
    ("2026-08-29", 5, 2521, 10079645.57, 2034920.83, -299613.25, 12907612.81, 139028000, 15),
    ("2026-08-30", 5, 2526, 10244409.08, 2199684.34, 167197.98, 13072376.33, 153601000, 20),
    ("2026-08-31", 5, 2535, 10219247.92, -18923.27, -18923.27, 13047215.17, 315218000, 45),
    ("2026-09-01", 16, 2540, 10382473.60, 167297.34, 144302.41, 13210440.85, 413938000, 47),
    ("2026-09-02", -114, 2542, 10792225.53, 574629.35, 551709.13, 13620192.77, 81800000, 13),
    ("2026-09-03", -118, 2549, 10586931.79, 369335.61, 346415.39, 13414899.03, 31910000, 6),
    ("2026-09-04", -118, 2563, 11357919.74, 1140323.57, 1117403.35, 14185886.99, 198769000, 22),
    ("2026-09-05", -118, 2563, 11296553.66, 1078957.48, 1056037.26, 14124520.90, 0, 0),
    ("2026-09-06", 18, 2563, 11083304.98, 865708.81, -239929.25, 13911272.23, 0, 0),
    ("2026-09-07", 18, 2566, 11066490.45, 848894.27, -16849.90, 13894457.69, 9340000, 4),
    ("2026-09-08", 44, 2568, 11399892.37, 1113261.58, 303471.49, 14227859.61, 0, 0),
    ("2026-09-09", 48, 2579, 11018883.07, 732252.28, -77537.81, 13846850.32, 4384539, 11),
    ("2026-09-10", 48, 2582, 10772785.57, 486154.79, -323635.30, 13600752.82, 4290000, 1),
]


def series() -> pd.DataFrame:
    df = pd.DataFrame(ROWS, columns=["eod_date", "skew_secs", "n_positions", "ytd_pnl", "mtd_pnl", "wtd_pnl", "ltd_pnl",
                                     "notional_quote", "n_trades"])
    df["is_final"] = True
    df["ytd_pnl_check"] = df["ltd_pnl"] - df["ltd_pnl"].iloc[0]
    df["daily_pnl"] = df["ltd_pnl"].diff()
    df["trailing_7d_pnl"] = df["ltd_pnl"].diff(7)
    return df.sort_values("eod_date", ascending=False).reset_index(drop=True)  # newest first, like BigQuery


class FakeBQ(DeskBigQuery):
    """Validates the script exactly like production and returns the canned series."""

    def __init__(self, frame=None):
        super().__init__(client=object(), data_project=PROJECT, allowed_datasets=ALLOWED, max_rows=200,
                         catalog_path="/nonexistent/bq_catalog.json")
        self.frame_fn = frame or series
        self.calls = []

    def query_script(self, sql, max_rows=None, timeout_s=None):
        safe = self.validate_script(sql, max_rows)
        self.calls.append((safe, max_rows, timeout_s))
        df = self.frame_fn()
        df.attrs["bytes_processed"] = 15_300_000_000
        df.attrs["cache_hit"] = False
        return df

    def query(self, sql):  # the tool must never use the single-statement path for the script
        raise AssertionError("query() must not be used for the EOW script")


@pytest.fixture
def clock():
    return {"t": 1_000_000.0, "today": date(2026, 9, 11)}


@pytest.fixture
def h(clock):
    bq = FakeBQ()
    inst = HarukoEodPnl(bq, now=lambda: clock["t"], today=lambda: clock["today"])
    inst.bq_fake = bq
    return inst


# ----------------------------------------------------------------------------
# guard: the DECLARE + SELECT script
# ----------------------------------------------------------------------------

def test_guard_accepts_carsons_script_and_keeps_limit():
    sql = he.SQL_PATH.read_text()
    safe = validate_declare_script(sql, PROJECT, ALLOWED, 1000)
    assert safe.startswith("DECLARE eod_time TIME DEFAULT TIME '15:00:00';")
    assert safe.count("DECLARE") == 4
    assert "fct_otc_haruko_position_pnl_history" in safe and safe.rstrip().endswith("LIMIT 1000")
    assert "-- Haruko end-of-day" not in safe  # header comment dropped, body intact


@pytest.mark.parametrize("sql", [
    "DECLARE x INT64; SELECT 1",                                    # no DEFAULT literal
    "DECLARE x INT64 DEFAULT (SELECT 1); SELECT 1",                 # sub-query default
    "DECLARE x INT64 DEFAULT 1 + 1; SELECT 1",                      # expression
    "DECLARE x STRING DEFAULT 'a; DROP TABLE t; --'; SELECT 1",     # ; inside the literal
    "DECLARE x INT64 DEFAULT 1;",                                   # no query
    "DECLARE x INT64 DEFAULT 1; SET x = 2; SELECT 1",               # SET
    "DECLARE x INT64 DEFAULT 1; SELECT 1; SELECT 2",                # two queries
    "DECLARE x INT64 DEFAULT 1; DECLARE x INT64 DEFAULT 2; SELECT 1",  # duplicate name
    f"DECLARE x INT64 DEFAULT 1; DELETE FROM `{PROJECT}.brokerage_a1.t` WHERE true",
    f"DECLARE x INT64 DEFAULT 1; INSERT INTO `{PROJECT}.brokerage_a1.t` SELECT 1",
    f"DECLARE x INT64 DEFAULT 1; SELECT * FROM `{PROJECT}.hold.balances`",           # denied dataset
    "DECLARE x INT64 DEFAULT 1; SELECT * FROM `other-project.brokerage_a1.t`",       # other project
    "DECLARE x INT64 DEFAULT 1; BEGIN SELECT 1; END",
    "",
])
def test_guard_rejects_bad_scripts(sql):
    with pytest.raises(SQLGuardError):
        validate_declare_script(sql, PROJECT, ALLOWED, 200)


def test_guard_plain_select_still_fine_and_single_statement_path_still_rejects_declare():
    from providers.bigquery import validate_sql
    assert validate_declare_script("SELECT 1", PROJECT, ALLOWED, 5).endswith("LIMIT 5")
    with pytest.raises(SQLGuardError):
        validate_sql("DECLARE x INT64 DEFAULT 1; SELECT 1", PROJECT, ALLOWED, 200)


def test_desk_bigquery_query_script_uses_raised_row_cap_and_timeout():
    class Job:
        total_bytes_processed = 42
        cache_hit = True

        def result(self, timeout=None):
            return self

        def to_dataframe(self, create_bqstorage_client=True):
            return pd.DataFrame({"eod_date": ["2026-09-10"], "ltd_pnl": [1.0]})

    class Client:
        def __init__(self):
            self.calls = []

        def query(self, sql, job_config=None, timeout=None):
            self.calls.append((sql, job_config, timeout))
            return Job()

    client = Client()
    bq = DeskBigQuery(client, data_project=PROJECT, allowed_datasets=ALLOWED, max_rows=200, timeout_s=60,
                      max_bytes_billed=20_000_000_000, catalog_path="/nonexistent")
    df = bq.query_script("DECLARE d DATE DEFAULT DATE '2025-12-31'; SELECT 1 AS x FROM `anc-global-markets.brokerage_a1.t`",
                         max_rows=1000, timeout_s=180)
    sql, cfg, timeout = client.calls[0]
    assert sql.endswith("LIMIT 1000") and timeout == 180 and cfg.maximum_bytes_billed == 20_000_000_000
    assert df.attrs["cache_hit"] is True and df.attrs["bytes_processed"] == 42


# ----------------------------------------------------------------------------
# DECLARE substitution
# ----------------------------------------------------------------------------

def test_substitute_declares_valid(h):
    sql = h.sql(eod_time="16:00:00", eod_zone="Europe/London", base_date="2026-01-31", skew=300)
    assert "DECLARE eod_time     TIME    DEFAULT TIME '16:00:00';" in sql
    assert "DEFAULT 'Europe/London';" in sql and "DEFAULT DATE '2026-01-31';" in sql and "DEFAULT 300;" in sql
    assert validate_declare_script(sql, PROJECT, ALLOWED, 1000)  # still passes the guard
    assert h.sql() == he.SQL_PATH.read_text()                   # no overrides -> verbatim


@pytest.mark.parametrize("kw", [
    dict(eod_time="25:00:00"), dict(eod_time="3pm"), dict(eod_zone="Mars/Base"), dict(eod_zone="America/Chicago'; DROP TABLE x; --"),
    dict(base_date="2026-13-01"), dict(base_date="yesterday"), dict(skew=-1), dict(skew=99999), dict(skew="x"), dict(foo=1),
])
def test_substitute_declares_rejects_bad_values(kw):
    with pytest.raises(HarukoEodError):
        substitute_declares(he.SQL_PATH.read_text(), **kw)


# ----------------------------------------------------------------------------
# period maths (LTD differences)
# ----------------------------------------------------------------------------

def test_frame_is_ascending_dates_and_goes_through_query_script(h):
    df = h.frame()
    assert list(df["eod_date"])[:2] == [date(2026, 6, 28), date(2026, 6, 29)] and df["eod_date"].iloc[-1] == date(2026, 9, 10)
    safe, max_rows, timeout = h.bq_fake.calls[0]
    assert max_rows == he.MAX_ROWS == 1000 and timeout == he.TIMEOUT_S and safe.startswith("DECLARE eod_time")
    assert df.attrs["bytes_processed"] == 15_300_000_000 and df.attrs["from_cache"] is False


def test_monthly_ltd_differences_including_first_month_and_august_reset(h):
    m = h.monthly().set_index("month")
    assert list(m.index) == ["2026-06", "2026-07", "2026-08", "2026-09"]
    # first month: from the first row (2026-06-28) to month end
    assert m.loc["2026-06", "from_first_row"] and m.loc["2026-06", "baseline"] == date(2026, 6, 28)
    assert round(m.loc["2026-06", "pnl"]) == 56594 and m.loc["2026-06", "n_days"] == 3 and not m.loc["2026-06", "partial"]
    assert round(m.loc["2026-07", "pnl"]) == 291096 and m.loc["2026-07", "baseline"] == date(2026, 6, 30)
    # August: LTD 10,872,692 (07-31) -> 13,047,215 (08-31) = 2,174,523 while Haruko's MTD column shows -18,923
    assert round(m.loc["2026-08", "pnl"]) == 2174523
    assert round(m.loc["2026-08", "haruko_mtd"]) == -18923
    assert m.loc["2026-08", "baseline"] == date(2026, 7, 31) and m.loc["2026-08", "end"] == date(2026, 8, 31)
    # September is month-to-date
    assert round(m.loc["2026-09", "pnl"]) == 553538 and m.loc["2026-09", "partial"] and m.loc["2026-09", "n_days"] == 10


@pytest.mark.parametrize("kind,pnl,baseline,end,n_days", [
    ("mtd", 553538, date(2026, 8, 31), date(2026, 9, 10), 10),
    ("wtd", -310519, date(2026, 9, 6), date(2026, 9, 10), 4),
    ("last_week", 838896, date(2026, 8, 30), date(2026, 9, 6), 7),
    ("last_month", 2174523, date(2026, 7, 31), date(2026, 8, 31), 31),
    ("month:2026-08", 2174523, date(2026, 7, 31), date(2026, 8, 31), 31),
    ("2026-07", 291096, date(2026, 6, 30), date(2026, 7, 31), 31),
    ("range:2026-08-15..2026-09-05", 2724390, date(2026, 8, 14), date(2026, 9, 5), 22),
    ("range:2026-09-10..2026-09-30", -246097, date(2026, 9, 9), date(2026, 9, 10), 1),
])
def test_period_forms(h, kind, pnl, baseline, end, n_days):
    r = h.period(kind)
    assert abs(r.pnl - pnl) < 1.0 and r.baseline_date == baseline and r.end == end and r.n_days == n_days
    assert not r.from_first_row and r.label


def test_period_from_first_row_when_no_prior_snapshot(h):
    r = h.period("range:2026-06-01..2026-07-05")
    assert r.from_first_row and r.baseline_date == date(2026, 6, 28) and round(r.pnl) == 145606
    assert any("first available row" in n for n in r.notes)
    assert list(r.months["month"]) == ["2026-06", "2026-07"]


def test_ytd_reports_both_figures_and_monthly_table(h):
    r = h.period("ytd")
    assert round(r.ytd_sum) == 10772786          # Haruko SUM(year_to_date_pnl) at 2026-09-10
    assert round(r.ytd_ltd_change) == 3075751    # LTD change since the first row 2026-06-28
    assert round(r.pnl) == 10772786
    assert list(r.months["month"]) == ["2026-06", "2026-07", "2026-08", "2026-09"]
    assert any("two ways" in n and "partial-year" in n for n in r.notes)


@pytest.mark.parametrize("kind", ["bogus", "month:2026-13", "month:2026-03", "range:2026-09-05..2026-09-01", "range:2027-01-01..2027-01-31"])
def test_period_rejects_bad_or_uncovered(h, kind):
    with pytest.raises(HarukoEodError):
        h.period(kind)


def test_business_days_and_staleness(h, clock):
    assert business_days_between(date(2026, 9, 10), date(2026, 9, 11)) == 1     # Thu -> Fri
    assert business_days_between(date(2026, 9, 11), date(2026, 9, 14)) == 1     # Fri -> Mon
    assert business_days_between(date(2026, 9, 9), date(2026, 9, 11)) == 2
    assert business_days_between(date(2026, 9, 11), date(2026, 9, 11)) == 0
    latest = h.latest()
    assert latest["eod_date"] == date(2026, 9, 10) and latest["business_days_behind"] == 1 and not latest["stale"]
    clock["today"] = date(2026, 9, 15)
    assert h.latest()["stale"] and h.latest()["business_days_behind"] == 3


def test_ttl_cache(h, clock):
    h.frame(); h.frame(); h.period("mtd"); h.period("ytd")
    assert len(h.bq_fake.calls) == 1
    assert h.frame().attrs["from_cache"] is True
    clock["t"] += he.DEFAULT_TTL_S            # exactly at the TTL still served
    h.frame(); assert len(h.bq_fake.calls) == 1
    clock["t"] += 1
    h.frame(); assert len(h.bq_fake.calls) == 2
    h.frame(force=True); assert len(h.bq_fake.calls) == 3
    h.frame(skew=300); assert len(h.bq_fake.calls) == 4   # different parameters -> separate entry
    h.clear_cache(); h.frame(); assert len(h.bq_fake.calls) == 5


# ----------------------------------------------------------------------------
# the tool
# ----------------------------------------------------------------------------

@pytest.fixture
def tool(monkeypatch, clock):
    bq = FakeBQ()
    monkeypatch.setattr(dt, "_get_bq", lambda: bq)
    dt._haruko_eod.clear()
    inst = dt._get_haruko_eod(bq)
    inst._now = lambda: clock["t"]
    inst._today = lambda: clock["today"]
    return bq


def test_tool_registered():
    names = [t.name for t in dt.get_desk_tools()]
    assert "get_derivs_pnl_eod" in names and names.index("get_derivs_pnl_eod") == names.index("get_desk_pnl_history") + 1
    assert "get_derivs_pnl_eod" in dt.DESK_TOOL_NAMES
    assert "Carson Levy" in dt.get_derivs_pnl_eod.description and "month_to_date" in dt.get_derivs_pnl_eod.description


def test_tool_august(tool):
    out = dt.get_derivs_pnl_eod.invoke({"period": "month:2026-08"})
    assert "**August 2026 PnL: $2,174,523**" in out
    assert "Carson Levy EOW method: 3pm America/Chicago EOD cut, LTD differences, one full-book snapshot/day" in out
    assert "$10,872,692 at 2026-07-31 EOD -> $13,047,215 at 2026-08-31 EOD" in out
    assert "MTD -$18,923" in out and "reference only" in out
    assert "A1 Ltd + ADSD combined" in out and "Bytes scanned: 14.25 GB" in out and "BigQuery cache hit: no" in out
    assert "fresh run" in out and "WARNING" not in out
    assert "from 2026-06-28 (first row) to 2026-09-10" in out
    assert "| EOD date |" not in out          # no daily table unless asked


def test_tool_ytd_dual_figures_and_monthly_table(tool):
    out = dt.get_derivs_pnl_eod.invoke({"period": "ytd"})
    assert "SUM(year_to_date_pnl) at 2026-09-10 EOD): $10,772,786" in out
    assert "LTD change since first available snapshot (2026-06-28 -> 2026-09-10): $3,075,751" in out
    assert "| 2026-08 | $2,174,523 | 2026-07-31 -> 2026-08-31 | 31 | -$18,923 |" in out
    assert "| 2026-09 (MTD, partial) | $553,538 |" in out and "2026-06 (from first snapshot)" in out
    assert "fresh run" in out
    out2 = dt.get_derivs_pnl_eod.invoke({"period": "mtd"})   # follow-up within the TTL -> no re-scan
    assert "served from the in-process 15-min cache" in out2 and "**MTD September 2026 (to 2026-09-10 EOD) PnL: $553,538**" in out2
    assert len(tool.calls) == 1


def test_tool_daily_table_capped_and_staleness_warning(tool, clock):
    clock["today"] = date(2026, 9, 15)
    out = dt.get_derivs_pnl_eod.invoke({"period": "range:2026-06-28..2026-09-10", "include_daily": True})
    daily_rows = [l for l in out.splitlines() if l.startswith("| 2026-") and l.count("|") == 7]
    assert len(daily_rows) == 31 and daily_rows[-1].startswith("| 2026-09-10 |")
    assert "last 31 of 75 days" in out
    assert "WARNING: the latest EOD snapshot is 3 business days old" in out
    assert "**Monthly PnL" in out


def test_tool_bad_period_and_unavailable(tool, monkeypatch):
    assert dt.get_derivs_pnl_eod.invoke({"period": "bogus"}).startswith("Bad period 'bogus'")
    monkeypatch.setattr(dt, "_get_bq", lambda: None)
    assert "not available" in dt.get_derivs_pnl_eod.invoke({})


def test_tool_empty_frame(monkeypatch):
    bq = FakeBQ(frame=lambda: pd.DataFrame(columns=["eod_date", "ltd_pnl"]))
    monkeypatch.setattr(dt, "_get_bq", lambda: bq)
    dt._haruko_eod.clear()
    assert "no EOD snapshots" in dt.get_derivs_pnl_eod.invoke({"period": "mtd"})


def test_pnl_history_points_to_eow_tool(monkeypatch):
    from tests.test_desk_tools import FakeBQ as HistFake, _portfolio_eod_frame
    bq = HistFake(frames={"fct_otc_haruko_pnl_portfolio_history_eod": lambda: _portfolio_eod_frame(3)})
    monkeypatch.setattr(dt, "_get_bq", lambda: bq)
    out = dt.get_desk_pnl_history.invoke({"days": 7, "by": "portfolio"})
    assert "use `get_derivs_pnl_eod`" in out and "authoritative" in out
