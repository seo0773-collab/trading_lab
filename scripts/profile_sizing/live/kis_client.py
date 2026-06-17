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
from pathlib import Path
from typing import Any

import requests

ROOT = Path(__file__).resolve().parents[3]
TOKEN_DIR = ROOT / "var" / "live"

DOMAINS = {
    "real": "https://openapi.koreainvestment.com:9443",
    "mock": "https://openapivts.koreainvestment.com:29443",
}
# ⚠️ 문서 대조 필요. 해외주식(미국). 실/모의 TR_ID 상이.
TR_IDS = {
    "real": {"balance": "TTTS3012R", "buy": "TTTT1002U", "sell": "TTTT1006U"},
    "mock": {"balance": "VTTS3012R", "buy": "VTTT1002U", "sell": "VTTT1001U"},
    "price": "HHDFS00000300",  # 해외주식 현재체결가(실/모의 공통, 실데이터 도메인)
}
EXCHANGE = {"NASD": "나스닥", "NYSE": "뉴욕", "AMEX": "아멕스"}


class KISConfigError(RuntimeError):
    pass


class KISClient:
    def __init__(self, env: str | None = None, *, timeout: float = 10.0):
        self.env = (env or os.environ.get("KIS_ENV", "mock")).lower()
        if self.env not in DOMAINS:
            raise KISConfigError(f"KIS_ENV must be mock|real, got {self.env!r}")
        self.app_key = os.environ.get("KIS_APP_KEY")
        self.app_secret = os.environ.get("KIS_APP_SECRET")
        acct = os.environ.get("KIS_ACCOUNT", "")
        if not (self.app_key and self.app_secret and acct):
            raise KISConfigError(
                "환경변수 KIS_APP_KEY/KIS_APP_SECRET/KIS_ACCOUNT 필요(.env 참고)")
        self.cano, _, self.acnt_prdt = acct.partition("-")
        self.acnt_prdt = self.acnt_prdt or "01"
        self.base = DOMAINS[self.env]
        self.timeout = timeout
        self._token: str | None = None

    # ----- auth -------------------------------------------------------
    def _token_path(self) -> Path:
        return TOKEN_DIR / f"kis_token_{self.env}.json"

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
        cash = float(out2.get("frcr_dncl_amt1") or out2.get("frcr_evlu_amt2") or 0.0)
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
