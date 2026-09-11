"""Abstract base class for all market data providers.

Each provider must return DataFrames with standardized column names so the
rest of the codebase is provider-agnostic.

Standard output schemas
-----------------------
get_spot_ohlcv   -> time, open, high, low, close, spot_volume
get_spot_price   -> time, price
get_funding_rate -> time, funding_rate  (annualized %, USD-margin)
get_perp_oi      -> time, perp_oi      (USD)
get_perp_volume  -> time, perp_volume  (USD)
get_liquidations -> time, long_liquidations, short_liquidations, total_liquidations  (USD)
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional

import pandas as pd


class MarketDataProvider(ABC):

    @abstractmethod
    def get_spot_ohlcv(
        self, token: str, start_date: datetime, end_date: datetime
    ) -> Optional[pd.DataFrame]:
        """OHLCV for spot markets. Columns: time, open, high, low, close, spot_volume."""

    @abstractmethod
    def get_spot_price(
        self, token: str, start_date: datetime, end_date: datetime
    ) -> Optional[pd.DataFrame]:
        """Daily closing price. Columns: time, price."""

    @abstractmethod
    def get_funding_rate(
        self, token: str, start_date: datetime, end_date: datetime
    ) -> Optional[pd.DataFrame]:
        """Annualized funding rate. Columns: time, funding_rate (%)."""

    @abstractmethod
    def get_perp_oi(
        self, token: str, start_date: datetime, end_date: datetime
    ) -> Optional[pd.DataFrame]:
        """Perpetual open interest in USD. Columns: time, perp_oi."""

    @abstractmethod
    def get_perp_volume(
        self, token: str, start_date: datetime, end_date: datetime
    ) -> Optional[pd.DataFrame]:
        """Perpetual trading volume in USD. Columns: time, perp_volume."""

    @abstractmethod
    def get_liquidations(
        self, token: str, start_date: datetime, end_date: datetime
    ) -> Optional[pd.DataFrame]:
        """Liquidations in USD. Columns: time, long_liquidations, short_liquidations, total_liquidations."""
