"""한국투자증권 KIS Developers REST 클라이언트 (해외=미국 주식).

⚠️ 안전·검증 주의 (반드시 읽을 것):
- 자격증명은 환경변수에서만 읽는다(리포에 비밀키 절대 저장 금지):
    KIS_APP_KEY, KIS_APP_SECRET, KIS_ACCOUNT(8자리-2자리, 예 12345678-01),
    KIS_ENV = mock(기본) | real
- **기본은 모의투자(mock) 도메인.** 실계좌(real)는 명시적으로 KIS_ENV=real + 호출부의
  추가 확인 플래그가 있어야 주문이 나간다.
- TR_ID·엔드포인트·응답 필드명은 KIS 버전에 따라 바뀐다. 아래 값은 작성 시점 기준이며
  **사용 전 https://apiportal.koreainvestment.com 최신 문서로 반드시 대조**하라.
  특히 해외주식 주문 TR_ID(매수/매도, 실/모의)와 거래소코드(NASD/NYSE/AMEX)를 확인.
- 토큰은 var/live/kis_token_<env>.json 에 캐시(24h). var/live는 gitignore.
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[3]
TOKEN_DIR = ROOT / "var" / "live"

DOMAINS = {
    "real": "https://openapi.koreainvestment.com:9443",
    "mock": "https://openapivts.koreainvestment.com:29443",
}
# ⚠️ 문서 대조 필요. 해외주식(미국). 실/모의 TR_ID 상이.
TR_IDS = {
    # 해외(미국) 주문·잔고 — 실/모의 상이. ⚠️ 문서 대조 필요.
    "real": {"balance": "TTTS3012R", "buy": "TTTT1002U", "sell": "TTTT1006U"},
    "mock": {"balance": "VTTS3012R", "buy": "VTTT1002U", "sell": "VTTT1001U"},
    # 국내 주문(현금)·잔고 — 실/모의 상이. ⚠️ 문서 대조 필요.
    "dom_real": {"balance": "TTTC8434R", "buy": "TTTC0802U", "sell": "TTTC0801U"},
    "dom_mock": {"balance": "VTTC8434R", "buy": "VTTC0802U", "sell": "VTTC0801U"},
    "price": "HHDFS00000300",          # 해외 현재가(실데이터 도메인)
    "price_domestic": "FHKST01010100",  # 국내 현재가(실데이터 도메인)
    # 과거 기간별시세(실데이터 도메인) — ⚠️ 문서 대조 필요
    "hist_domestic": "FHKST03010100",  # 국내 일/주/월/년봉
    "hist_overseas": "HHDFS76240000",  # 해외 일/주/월봉
}
EXCHANGE = {"NASD": "나스닥", "NYSE": "뉴욕", "AMEX": "아멕스"}

# 해외 기간별시세 EXCD(NAS/NYS/AMS)는 현재가 엔드포인트(NASD/NYSE/AMEX)와 코드가 다르다.
HIST_EXCD = {"NASD": "NAS", "NYSE": "NYS", "AMEX": "AMS",
             "NAS": "NAS", "NYS": "NYS", "AMS": "AMS"}
# 국내 기간별 코드 D/W/M/Y, 해외 GUBN 0/1/2(일/주/월)
_DOM_PERIOD = {"1d": "D", "1wk": "W", "1mo": "M", "d": "D", "w": "W", "m": "M"}
_OVS_GUBN = {"1d": "0", "1wk": "1", "1mo": "2", "d": "0", "w": "1", "m": "2"}


class KISConfigError(RuntimeError):
    pass


class KISSchemaError(RuntimeError):
    """잔고/주문 응답에서 기대 필드를 못 찾음 = TR_ID·스키마 불일치(조용히 0 폴백 금지)."""


def _pick_amount(out2: dict, fields: tuple[str, ...], label: str) -> float:
    """잔고 응답에서 금액 필드를 **명시적으로** 고른다.

    기존 ``a or b or 0.0`` 무음 폴백의 위험: 스키마가 바뀌어 기대 필드가 통째로
    사라지면 cash=0 으로 떨어지고 → account_value 가 보유평가만으로 계산되어 →
    목표비중×틀린자산 = 과/소 주문(전량매도 포함)으로 직결된다. 그래서:
      - fields 중 **키가 존재하는** 첫 필드의 값을 쓴다(값이 0/"0"이어도 유효=정상).
      - 기대 필드가 **하나도 존재하지 않으면** 예외(스키마 불일치를 표면화).
    """
    for f in fields:
        if f in out2:
            v = out2.get(f)
            if v in (None, ""):
                continue
            try:
                return float(v)
            except (TypeError, ValueError):
                continue
        # 키 자체는 있는데 값이 None/빈문자열이면 다음 필드로
    if any(f in out2 for f in fields):
        return 0.0  # 기대 필드는 있으나 모두 빈값 → 정상적 0 으로 간주
    raise KISSchemaError(
        f"{label}: 기대 예수금 필드 {fields} 가 응답에 없음. "
        f"실서버 스키마 변경 가능 — 응답 키: {sorted(out2)[:25]}")


class KISClient:
    def __init__(self, env: str | None = None, *, timeout: float = 10.0,
                 app_key: str | None = None, app_secret: str | None = None,
                 account: str | None = None):
        self.env = (env or os.environ.get("KIS_ENV", "mock")).lower()
        if self.env not in DOMAINS:
            raise KISConfigError(f"KIS_ENV must be mock|real, got {self.env!r}")
        self.app_key = app_key or os.environ.get("KIS_APP_KEY")
        self.app_secret = app_secret or os.environ.get("KIS_APP_SECRET")
        acct = account or os.environ.get("KIS_ACCOUNT", "")
        if not (self.app_key and self.app_secret and acct):
            raise KISConfigError(
                "자격증명 필요: app_key/app_secret/account (env KIS_APP_KEY/SECRET/ACCOUNT)")
        self.cano, _, self.acnt_prdt = acct.partition("-")
        self.acnt_prdt = self.acnt_prdt or "01"
        self.base = DOMAINS[self.env]
        self.timeout = timeout
        self._token: str | None = None

    @classmethod
    def from_env(cls, prefix: str = "KIS", *, env: str | None = None,
                 **kw) -> "KISClient":
        """접두사로 자격증명 묶음 로드. 예) from_env('KIS_MOCK', env='mock')
        → KIS_MOCK_APP_KEY/SECRET/ACCOUNT 사용. 시세=실키, 주문=모의키 분리용."""
        return cls(env=env,
                   app_key=os.environ.get(f"{prefix}_APP_KEY"),
                   app_secret=os.environ.get(f"{prefix}_APP_SECRET"),
                   account=os.environ.get(f"{prefix}_ACCOUNT"), **kw)

    # ----- auth -------------------------------------------------------
    def _token_path(self) -> Path:
        tag = (self.app_key or "")[:6]
        return TOKEN_DIR / f"kis_token_{self.env}_{tag}.json"

    def access_token(self) -> str:
        if self._token:
            return self._token
        p = self._token_path()
        if p.exists():
            cached = json.loads(p.read_text())
            if cached.get("expire_at", 0) > time.time() + 60:
                self._token = cached["token"]
                return self._token
        r = requests.post(f"{self.base}/oauth2/tokenP",
                          json={"grant_type": "client_credentials",
                                "appkey": self.app_key, "appsecret": self.app_secret},
                          timeout=self.timeout)
        r.raise_for_status()
        tok = r.json()["access_token"]
        TOKEN_DIR.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"token": tok, "expire_at": time.time() + 23 * 3600}))
        self._token = tok
        return tok

    def _headers(self, tr_id: str, hashkey: str | None = None) -> dict[str, str]:
        h = {
            "authorization": f"Bearer {self.access_token()}",
            "appkey": self.app_key, "appsecret": self.app_secret,
            "tr_id": tr_id, "content-type": "application/json; charset=utf-8",
        }
        if hashkey:
            h["hashkey"] = hashkey
        return h

    def _hashkey(self, body: dict) -> str:
        r = requests.post(f"{self.base}/uapi/hashkey",
                          headers={"appkey": self.app_key, "appsecret": self.app_secret},
                          json=body, timeout=self.timeout)
        r.raise_for_status()
        return r.json()["HASH"]

    # ----- market data ------------------------------------------------
    def price(self, symbol: str, exchange: str = "NASD") -> float:
        """해외주식 현재가. (정규장 외엔 지연/직전가일 수 있음 — 문서 확인)"""
        r = requests.get(
            f"{self.base}/uapi/overseas-price/v1/quotations/price",
            headers=self._headers(TR_IDS["price"]),
            params={"AUTH": "", "EXCD": exchange, "SYMB": symbol},
            timeout=self.timeout)
        r.raise_for_status()
        return float(r.json()["output"]["last"])

    # ----- 과거 일/주/월봉 (백테스트 데이터 소스) ---------------------
    def _hist_get(self, url: str, tr_id: str, params: dict) -> dict:
        """기간별시세 GET. 간헐적 5xx/429 에 지수백오프 재시도."""
        last: Exception | None = None
        for attempt in range(4):
            r = requests.get(url, headers=self._headers(tr_id),
                             params=params, timeout=self.timeout)
            if r.status_code in (429, 500, 502, 503):
                last = requests.HTTPError(f"{r.status_code} on {url}")
                time.sleep(0.4 * (attempt + 1))
                continue
            r.raise_for_status()
            return r.json()
        raise last or RuntimeError("hist_get failed")

    @staticmethod
    def _is_domestic(symbol: str) -> bool:
        """Yahoo 스타일 심볼이 국내(.KS/.KQ) 인지 판별."""
        s = symbol.upper()
        return s.endswith(".KS") or s.endswith(".KQ") or s.isdigit()

    @staticmethod
    def _to_code6(symbol: str) -> str:
        return symbol.upper().split(".")[0].zfill(6)

    def daily_history(
        self, symbol: str, *, interval: str = "1d",
        start: str | None = None, end: str | None = None,
        adjusted: bool = True, exchange: str = "NASD",
    ) -> pd.DataFrame:
        """과거 OHLCV 를 [start, end] 구간으로 받아 정규화 DataFrame 반환.

        - symbol: Yahoo 스타일(``091160.KS``, ``XLK``). .KS/.KQ/숫자면 국내로 라우팅.
        - interval: 1d|1wk|1mo. start/end: 'YYYY-MM-DD' (None=최대/오늘).
        - adjusted: 수정주가(배당·분할 반영). Yahoo auto_adjust 와 정합 맞추는 핵심 플래그.
        - 반환 컬럼: open/high/low/close/volume, DatetimeIndex(naive, 날짜).
        시세 API 는 실데이터 도메인이므로 KIS_ENV 와 무관하게 self.base 사용(실키 필요).
        """
        end_dt = pd.Timestamp(end) if end else pd.Timestamp.today().normalize()
        start_dt = pd.Timestamp(start) if start else pd.Timestamp("2000-01-01")
        if self._is_domestic(symbol):
            frame = self._hist_domestic(self._to_code6(symbol), interval,
                                        start_dt, end_dt, adjusted)
        else:
            frame = self._hist_overseas(symbol.upper(), exchange, interval,
                                        start_dt, end_dt, adjusted)
        if frame.empty:
            return frame
        frame = frame[~frame.index.duplicated(keep="last")].sort_index()
        return frame.loc[(frame.index >= start_dt) & (frame.index <= end_dt)]

    def _hist_domestic(self, code6, interval, start_dt, end_dt, adjusted) -> pd.DataFrame:
        period = _DOM_PERIOD.get(interval.lower(), "D")
        url = f"{self.base}/uapi/domestic-stock/v1/quotations/inquire-daily-itemchartprice"
        rows: list[dict] = []
        cursor = end_dt
        for _ in range(400):  # 안전 상한(페이지네이션 무한루프 방지)
            params = {
                "FID_COND_MRKT_DIV_CODE": "J", "FID_INPUT_ISCD": code6,
                "FID_INPUT_DATE_1": start_dt.strftime("%Y%m%d"),
                "FID_INPUT_DATE_2": cursor.strftime("%Y%m%d"),
                "FID_PERIOD_DIV_CODE": period,
                "FID_ORG_ADJ_PRC": "0" if adjusted else "1",  # 0=수정주가
            }
            out = self._hist_get(url, TR_IDS["hist_domestic"], params).get("output2") or []
            chunk = [d for d in out if d.get("stck_bsop_date")]
            if not chunk:
                break
            rows.extend(chunk)
            oldest = min(d["stck_bsop_date"] for d in chunk)
            oldest_dt = pd.Timestamp(datetime.strptime(oldest, "%Y%m%d"))
            if oldest_dt <= start_dt or len(chunk) < 2:
                break
            cursor = oldest_dt - timedelta(days=1)
            time.sleep(0.08)  # rate-limit 여유
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame({
            "open": [float(d.get("stck_oprc", 0) or 0) for d in rows],
            "high": [float(d.get("stck_hgpr", 0) or 0) for d in rows],
            "low": [float(d.get("stck_lwpr", 0) or 0) for d in rows],
            "close": [float(d.get("stck_clpr", 0) or 0) for d in rows],
            "volume": [float(d.get("acml_vol", 0) or 0) for d in rows],
        }, index=pd.to_datetime([d["stck_bsop_date"] for d in rows], format="%Y%m%d"))
        return df[df["close"] > 0]

    def _hist_overseas(self, symbol, exchange, interval, start_dt, end_dt, adjusted) -> pd.DataFrame:
        gubn = _OVS_GUBN.get(interval.lower(), "0")
        excd = HIST_EXCD.get(exchange.upper(), "NAS")
        url = f"{self.base}/uapi/overseas-price/v1/quotations/dailyprice"
        rows: list[dict] = []
        cursor = end_dt
        for _ in range(400):
            params = {
                "AUTH": "", "EXCD": excd, "SYMB": symbol, "GUBN": gubn,
                "BYMD": cursor.strftime("%Y%m%d"),
                "MODP": "1" if adjusted else "0",  # 1=수정주가 반영
            }
            out = self._hist_get(url, TR_IDS["hist_overseas"], params).get("output2") or []
            chunk = [d for d in out if d.get("xymd")]
            if not chunk:
                break
            rows.extend(chunk)
            oldest = min(d["xymd"] for d in chunk)
            oldest_dt = pd.Timestamp(datetime.strptime(oldest, "%Y%m%d"))
            if oldest_dt <= start_dt or len(chunk) < 2:
                break
            cursor = oldest_dt - timedelta(days=1)
            time.sleep(0.08)
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame({
            "open": [float(d.get("open", 0) or 0) for d in rows],
            "high": [float(d.get("high", 0) or 0) for d in rows],
            "low": [float(d.get("low", 0) or 0) for d in rows],
            "close": [float(d.get("clos", 0) or 0) for d in rows],
            "volume": [float(d.get("tvol", 0) or 0) for d in rows],
        }, index=pd.to_datetime([d["xymd"] for d in rows], format="%Y%m%d"))
        return df[df["close"] > 0]

    def balance(self, exchange: str = "NASD", currency: str = "USD") -> dict[str, Any]:
        """해외주식 잔고 → {'holdings': {sym: shares}, 'cash': float}.

        ⚠️ 응답 필드명(output1/output2, ovrs_cblc_qty 등)은 문서 대조 필요."""
        tr = TR_IDS[self.env]["balance"]
        r = requests.get(
            f"{self.base}/uapi/overseas-stock/v1/trading/inquire-balance",
            headers=self._headers(tr),
            params={"CANO": self.cano, "ACNT_PRDT_CD": self.acnt_prdt,
                    "OVRS_EXCG_CD": exchange, "TR_CRCY_CD": currency,
                    "CTX_AREA_FK200": "", "CTX_AREA_NK200": ""},
            timeout=self.timeout)
        r.raise_for_status()
        data = r.json()
        holdings = {}
        for row in data.get("output1", []):
            sym = row.get("ovrs_pdno") or row.get("pdno")
            qty = float(row.get("ovrs_cblc_qty", 0) or 0)
            if sym and qty:
                holdings[sym] = qty
        out2 = data.get("output2") or {}
        if isinstance(out2, list):
            out2 = out2[0] if out2 else {}
        cash = _pick_amount(out2, ("frcr_dncl_amt1", "frcr_evlu_amt2"),
                            "해외잔고 예수금")
        return {"holdings": holdings, "cash": cash, "raw": data}

    def order(self, symbol: str, side: str, qty: float, *, exchange: str = "NASD",
              price: float = 0.0, confirm_real: bool = False) -> dict[str, Any]:
        """해외주식 주문. side=BUY|SELL. price=0이면 시장가 성격(문서상 지정가 권장).

        실계좌(real)는 confirm_real=True가 없으면 거부(안전장치)."""
        if self.env == "real" and not confirm_real:
            raise KISConfigError("실계좌 주문은 confirm_real=True 필요(안전장치).")
        tr = TR_IDS[self.env]["buy" if side.upper() == "BUY" else "sell"]
        body = {
            "CANO": self.cano, "ACNT_PRDT_CD": self.acnt_prdt,
            "OVRS_EXCG_CD": exchange, "PDNO": symbol,
            "ORD_QTY": str(int(qty)),
            "OVRS_ORD_UNPR": f"{price:.2f}" if price else "0",
            "ORD_SVR_DVSN_CD": "0", "ORD_DVSN": "00",
        }
        hk = self._hashkey(body)
        r = requests.post(
            f"{self.base}/uapi/overseas-stock/v1/trading/order",
            headers=self._headers(tr, hashkey=hk), json=body, timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    # ----- 국내(KRW) 현재가/잔고/주문 -----------------------------------
    def domestic_price(self, symbol: str) -> float:
        """국내주식 현재가. symbol=6자리 또는 091160.KS. 실데이터 도메인."""
        r = requests.get(
            f"{self.base}/uapi/domestic-stock/v1/quotations/inquire-price",
            headers=self._headers(TR_IDS["price_domestic"]),
            params={"FID_COND_MRKT_DIV_CODE": "J",
                    "FID_INPUT_ISCD": self._to_code6(symbol)},
            timeout=self.timeout)
        r.raise_for_status()
        return float(r.json()["output"]["stck_prpr"])

    def domestic_balance(self) -> dict[str, Any]:
        """국내주식 잔고 → {'holdings': {6자리코드: 주수}, 'cash': 원화예수금}.

        ⚠️ 응답 필드명(output1/output2, hldg_qty, dnca_tot_amt 등) 문서 대조 필요."""
        tr = TR_IDS[f"dom_{self.env}"]["balance"]
        r = requests.get(
            f"{self.base}/uapi/domestic-stock/v1/trading/inquire-balance",
            headers=self._headers(tr),
            params={"CANO": self.cano, "ACNT_PRDT_CD": self.acnt_prdt,
                    "AFHR_FLPR_YN": "N", "OFL_YN": "", "INQR_DVSN": "02",
                    "UNPR_DVSN": "01", "FUND_STTL_ICLD_YN": "N",
                    "FNCG_AMT_AUTO_RDPT_YN": "N", "PRCS_DVSN": "01",
                    "CTX_AREA_FK100": "", "CTX_AREA_NK100": ""},
            timeout=self.timeout)
        r.raise_for_status()
        data = r.json()
        holdings = {}
        for row in data.get("output1", []):
            code = row.get("pdno")
            qty = float(row.get("hldg_qty", 0) or 0)
            if code and qty:
                holdings[code] = qty
        out2 = data.get("output2") or {}
        if isinstance(out2, list):
            out2 = out2[0] if out2 else {}
        cash = _pick_amount(out2, ("dnca_tot_amt", "prvs_rcdl_excc_amt"),
                            "국내잔고 예수금")
        return {"holdings": holdings, "cash": cash, "raw": data}

    def domestic_order(self, symbol: str, side: str, qty: float, *,
                       price: float = 0.0, confirm_real: bool = False) -> dict[str, Any]:
        """국내주식 현금 주문. side=BUY|SELL. price=0 → 시장가(ORD_DVSN=01),
        price>0 → 지정가(00). 실계좌(real)는 confirm_real=True 필요(안전장치)."""
        if self.env == "real" and not confirm_real:
            raise KISConfigError("실계좌 주문은 confirm_real=True 필요(안전장치).")
        tr = TR_IDS[f"dom_{self.env}"]["buy" if side.upper() == "BUY" else "sell"]
        body = {
            "CANO": self.cano, "ACNT_PRDT_CD": self.acnt_prdt,
            "PDNO": self._to_code6(symbol),
            "ORD_DVSN": "00" if price else "01",  # 00=지정가, 01=시장가
            "ORD_QTY": str(int(qty)),
            "ORD_UNPR": str(int(price)) if price else "0",
        }
        hk = self._hashkey(body)
        r = requests.post(
            f"{self.base}/uapi/domestic-stock/v1/trading/order-cash",
            headers=self._headers(tr, hashkey=hk), json=body, timeout=self.timeout)
        r.raise_for_status()
        return r.json()
