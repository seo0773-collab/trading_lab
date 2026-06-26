"""Handler for heatmap2 — 고볼륨 노드(HVN) 지지/저항 롱숏 (단일종목).

heatmap1과 같은 볼륨 프로파일을 쓰되, 신호 입력을 단일 POC/VA가 아니라 프로파일에서
추출한 **여러 고볼륨 노드(HVN)** = '히트맵에서 색 짙은 구간'으로 바꾼다. 각 시점에서
현재가 기준 **바로 아래 HVN = 지지(val)**, **바로 위 HVN = 저항(vah)**으로 매핑해
heatmap1의 시뮬레이션 엔진(next_open 체결·손절·POC목표·시간/반대청산)을 그대로
상속한다. 따라서:

- ``signal_mode='va_reversion'``(기본) = 지지에서 반등 → 롱 / 저항에서 거부 → 숏.
- ``signal_mode='va_breakout'``           = 저항 상향돌파 → 롱 / 지지 하향이탈 → 숏.

가격축은 ``price_scale='log'`` 기본(변동성 큰 구간 저가대 보존). 양방향이 기본
(``long_only=false``)이라 숏도 체결한다.
"""
from __future__ import annotations

import sys
from typing import Any

import numpy as np
import pandas as pd

from trading_lab.paths import ROOT
from trading_lab.strategies.heatmap1 import Heatmap1Handler

SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from volume_profile import rolling_sr_levels  # noqa: E402


class Heatmap2Handler(Heatmap1Handler):
    """HVN 지지/저항 전략. load_data·시뮬·메트릭은 heatmap1 상속, 레벨만 교체."""

    def _levels(self, df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
        return rolling_sr_levels(
            df,
            lookback=int(config.get("lookback", 120)),
            bins=int(config.get("profile_bins", 80)),
            scale=str(config.get("price_scale", "log")),
            top_n=int(config.get("node_top_n", 4)),
            min_strength=float(config.get("node_min_strength", 0.3)),
            min_gap_bins=int(config.get("node_min_gap_bins", 3)),
            va_pct=float(config.get("va_pct", 0.70)),
            cumulative=bool(config.get("cumulative", False)),
            axis=str(config.get("sr_axis", "absolute")),
        )

    @staticmethod
    def _signals(
        df: pd.DataFrame, levels: pd.DataFrame, config: dict[str, Any]
    ) -> tuple[np.ndarray, np.ndarray]:
        """HVN 지지/저항 신호. 지지/저항이 매 봉 종가 기준 동적이라 heatmap1의
        '종가 이탈→복귀' 대신 **고저가의 레벨 터치**로 잡는다 (lookahead 없음).

        - va_reversion: 당봉 저가가 지지를 찍고 종가가 위로 마감 → 롱(지지 방어).
          당봉 고가가 저항을 찍고 종가가 아래로 마감 → 숏(저항 거부).
        - va_breakout: 종가가 전봉 저항을 상향 돌파 → 롱 / 전봉 지지를 하향 이탈 → 숏.
        """
        mode = str(config.get("signal_mode", "va_reversion"))
        cl = df["close"].to_numpy(float)
        op = df["open"].to_numpy(float)
        hi = df["high"].to_numpy(float)
        lo = df["low"].to_numpy(float)
        sup = levels["val"].to_numpy(float)   # 지지
        res = levels["vah"].to_numpy(float)   # 저항
        n = len(df)

        # 반등 확인 필터(거래과다 솎기, 기본 off=기존 동작 불변):
        #  confirm_candle  : 롱=양봉·숏=음봉이어야 신호(실제 방향 확인)
        #  confirm_volume_mult>0 : 터치 봉 거래량 ≥ rolling평균×mult(수급 확인)
        confirm_candle = bool(config.get("confirm_candle", False))
        vmult = float(config.get("confirm_volume_mult", 0.0))
        vol_ok = np.ones(n, dtype=bool)
        if vmult > 0.0 and "volume" in df:
            vwin = int(config.get("confirm_volume_win", 20))
            vol = df["volume"].astype(float)
            vmean = vol.rolling(vwin, min_periods=max(2, vwin // 2)).mean()
            vol_ok = (vol >= vmean * vmult).to_numpy()

        long_sig = np.zeros(n, dtype=bool)
        short_sig = np.zeros(n, dtype=bool)
        for i in range(1, n):
            lg = sh = False
            if mode == "va_breakout":
                if not np.isnan(res[i - 1]):
                    lg = cl[i - 1] <= res[i - 1] and cl[i] > res[i - 1]
                if not np.isnan(sup[i - 1]):
                    sh = cl[i - 1] >= sup[i - 1] and cl[i] < sup[i - 1]
            else:  # va_reversion: 레벨 터치 후 종가 방어/거부
                if not np.isnan(sup[i]):
                    lg = lo[i] <= sup[i] and cl[i] > sup[i]
                if not np.isnan(res[i]):
                    sh = hi[i] >= res[i] and cl[i] < res[i]
            if confirm_candle:
                lg = lg and cl[i] > op[i]   # 롱=양봉
                sh = sh and cl[i] < op[i]   # 숏=음봉
            if not vol_ok[i]:
                lg = sh = False
            long_sig[i] = lg
            short_sig[i] = sh
        return long_sig, short_sig

    @staticmethod
    def _targets(
        mode: str, d: int, poc: float, vah: float, val: float, buf: float
    ) -> tuple[float, float]:
        """진입 방향별 (take_profit, stop). vah=저항·val=지지.

        반대편 노드가 없으면(상단/하단 HVN 부재) POC로 폴백한다. 신호 시점은
        프로파일이 비어있지 않아 poc는 항상 유효(빈 프로파일은 신호가 안 뜸).
        """
        sup = val if val == val else poc   # NaN 폴백
        res = vah if vah == vah else poc
        width = res - sup
        if not (width > 0):
            width = (abs(sup) if sup == sup else 1.0) * 0.05
        if mode == "va_breakout":
            # 돌파 연장 목표 / 돌파한 레벨로 되돌림 손절
            if d == 1:
                return res + buf * width, res - buf * width
            return sup - buf * width, sup + buf * width
        # va_reversion: 지지↔저항 스윙
        if d == 1:  # 지지 반등 롱 → 저항 목표, 지지 아래 손절
            return res, sup - buf * width
        return sup, res + buf * width  # 저항 거부 숏 → 지지 목표, 저항 위 손절

    @staticmethod
    def _entry_reason(
        d: int, mode: str, poc: float, vah: float, val: float
    ) -> str:
        side = "롱" if d == 1 else "숏"
        if mode == "va_breakout":
            edge = "저항 상향돌파" if d == 1 else "지지 하향이탈"
        else:
            edge = "지지 반등" if d == 1 else "저항 거부"
        return f"{side} · {edge} · POC {poc:.2f} 지지 {val:.2f} 저항 {vah:.2f}"
