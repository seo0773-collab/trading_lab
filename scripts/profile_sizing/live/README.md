# yoon1b 실거래 모듈 (KIS)

신호 생성(백테스트/페이퍼)과 분리된 **실행** 레이어. 실돈 안전을 위해 **기본 dry-run·
모의투자**이며, 단계적으로만 실계좌로 올라간다.

## 구성
- `rebalance.py` — 브로커 무관 순수 주문 계산(`rebalance_plan`). 단위 테스트 대상.
- `kis_client.py` — 한국투자증권 KIS REST 클라이언트(해외=미국 주식). 자격증명은 env.
- `kis_rebalance.py` — 오케스트레이터 CLI(목표→잔고→주문표→(선택)제출).

## 토스증권?
토스증권은 (작성 시점 기준) 리테일 공개 매매 API가 없어 **자동매매 불가**. 단 yoon1b는
월 1회 리밸런스라 주문이 몇 건뿐 → `--holdings-file`로 주문표만 뽑아 **토스 앱에서 수동
체결**이 현실적. 자동화를 원하면 KIS(미국주식 지원·모의투자 있음)를 쓴다.

## 설정 (KIS)
1. https://apiportal.koreainvestment.com 에서 앱 등록 → APP_KEY/SECRET 발급, 계좌 개설.
2. 비밀키는 **리포에 저장 금지**. 환경변수로:
   ```bash
   cp .env.example .env   # 채운 뒤
   export KIS_APP_KEY=... KIS_APP_SECRET=... KIS_ACCOUNT=12345678-01 KIS_ENV=mock
   ```
   (`.env`·`var/live/`는 gitignore)

## 사용 순서 (안전 단계)
```bash
# 0) 키 없이 로직만: 계좌 JSON으로 주문표 계산(수동/토스용)
PYTHONPATH=src .venv/bin/python scripts/profile_sizing/live/kis_rebalance.py \
    --holdings-file my_account.json
#    my_account.json = {"cash":1000, "holdings":{"AAPL":2}, "prices":{"AAPL":210,...}}

# 1) KIS 모의투자 잔고로 주문표(제출 안 함)
PYTHONPATH=src .venv/bin/python scripts/profile_sizing/live/kis_rebalance.py

# 2) 모의투자 제출
... --execute

# 3) 실계좌 (충분히 검증 후): 다단 확인 필요
KIS_ENV=real ... --execute --confirm-real
```

## ⚠️ 반드시 검증할 것
- `kis_client.py`의 **TR_ID·엔드포인트·응답 필드명은 KIS 버전에 민감**하다. 사용 전
  최신 문서로 대조하라(특히 해외주식 주문 TR_ID 실/모의, 거래소코드, 잔고 output 필드).
- 이 클라이언트는 **실 KIS 서버 대상으로 자동 테스트되지 않았다**. 반드시 모의투자에서
  소량으로 먼저 검증하라. 순수 계산(`rebalance.py`)만 단위 테스트로 검증됨.
- 시장가/지원 주문유형·정규장 시간·환전(원화↔달러)·결제는 별도 확인 필요.
