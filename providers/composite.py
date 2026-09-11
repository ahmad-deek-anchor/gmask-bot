"""Composite provider: routes spot calls to one provider, derivatives to another.

Routing
-------
  get_spot_ohlcv   -> spot        (Coin Metrics)
  get_spot_price   -> spot        (Coin Metrics)
  get_funding_rate -> derivatives (Amberdata)
  get_perp_oi      -> derivatives (Amberdata)
  get_perp_volume  -> derivatives (Amberdata)
  get_liquidations -> derivatives (Amberdata)

Both children implement MarketDataProvider, so the composite is itself a
drop-in MarketDataProvider. No caching: every call goes straight through.
"""

from datetime import datetime
from typing import Optional

import pandas as pd

from providers.base import MarketDataProvider


class CompositeProvider(MarketDataProvider):

    def __init__(self, spot: MarketDataProvider, derivatives: MarketDataProvider):
        self._spot = spot
        self._derivatives = derivatives

    @property
    def spot(self) -> MarketDataProvider:
        return self._spot

    @property
    def derivatives(self) -> MarketDataProvider:
        return self._derivatives

    # --- spot -----------------------------------------------------------

    def get_spot_ohlcv(
        self, token: str, start_date: datetime, end_date: datetime
    ) -> Optional[pd.DataFrame]:
        return self._spot.get_spot_ohlcv(token, start_date, end_date)

    def get_spot_price(
        self, token: str, start_date: datetime, end_date: datetime
    ) -> Optional[pd.DataFrame]:
        return self._spot.get_spot_price(token, start_date, end_date)

    # --- derivatives ----------------------------------------------------

    def get_funding_rate(
        self, token: str, start_date: datetime, end_date: datetime
    ) -> Optional[pd.DataFrame]:
        return self._derivatives.get_funding_rate(token, start_date, end_date)

    def get_perp_oi(
        self, token: str, start_date: datetime, end_date: datetime
    ) -> Optional[pd.DataFrame]:
        return self._derivatives.get_perp_oi(token, start_date, end_date)

    def get_perp_volume(
        self, token: str, start_date: datetime, end_date: datetime
    ) -> Optional[pd.DataFrame]:
        return self._derivatives.get_perp_volume(token, start_date, end_date)

    def get_liquidations(
        self, token: str, start_date: datetime, end_date: datetime
    ) -> Optional[pd.DataFrame]:
        return self._derivatives.get_liquidations(token, start_date, end_date)
