"""재현 가능한 합성 OHLCV (계약 테스트·드라이런용).

profile-sizing은 가격 위치/국면을 다루므로, 합성 데이터는 상승·하락·회복 국면이
모두 등장하도록 구성한다. 같은 (n_bars, seed)면 항상 같은 시계열을 만든다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def make_synthetic_ohlcv(n_bars: int, seed: int, interval: str = "1d") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    index = pd.bdate_range(start="2008-01-02", periods=int(n_bars), name="timestamp")

    # 국면이 번갈아 나오는 드리프트: 장기 상승 + 주기적 하락/회복 사이클.
    t = np.arange(n_bars)
    cycle = np.sin(2 * np.pi * t / max(n_bars / 4.0, 1.0))   # 4 사이클
    regime_drift = 0.0004 + 0.0010 * cycle                   # 상승/하락 번갈아
    noise = rng.normal(0.0, 0.012, size=n_bars)
    daily = regime_drift + noise

    close = 100.0 * np.cumprod(1.0 + daily)
    close = pd.Series(close, index=index)
    intra = np.abs(rng.normal(0.0, 0.008, size=n_bars))
    open_ = close.shift(1).fillna(close.iloc[0]).to_numpy()
    high = np.maximum.reduce([close.to_numpy() * (1 + intra), open_, close.to_numpy()])
    low = np.minimum.reduce([close.to_numpy() * (1 - intra), open_, close.to_numpy()])
    volume = rng.integers(1_000, 10_000, size=n_bars).astype(float)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low,
         "close": close.to_numpy(), "volume": volume},
        index=index,
    )
