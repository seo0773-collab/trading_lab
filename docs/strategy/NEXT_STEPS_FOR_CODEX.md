# yoon1b 핸드오프 — 다음 작업 제안 (for Codex)

작성 2026-06-17. 이 문서는 다음 작업 에이전트(Codex)가 **맥락 없이도 이어받을 수 있게** 쓴
핸드오프다. 먼저 "현재 상태/정직한 결론"을 읽고, 그다음 "제안 작업"을 우선순위대로 진행하라.
권위 있는 최신 현황은 항상 리포지토리 루트 `now.txt` 다.

---

## 0. 한 줄 정리 (오해 금지)

yoon1b는 **"시장을 이기는 알파 전략"이 아니다.** PIT(시점구성) 재검증에서 그 주장은
**손픽 생존편향 산물**로 드러났다(아래 §2). 정직한 포지셔닝은
**"시장과 비슷한 수익을 내며 낙폭을 의미 있게 줄이는 방어형 배분"** 이다.
새 결과를 보고할 때 "시장 초과/알파"로 과장하지 말 것.

---

## 1. 현재 상태 (무엇이 완료됐나)

- **전략**: 다종목 포트폴리오. 매월 점수(profile-percentile 사이징) 상위 K=20을 점수비례로
  담고 노출=mean(top-K 점수). SPY 200MA 시장필터. 운영 확정 변형:
  - `yoon1b` (gain 1.25, 수익 우선) ← **운영 선택**
  - `yoon1c` (SPY∧섹터 하이브리드 레짐, 방어 우선) — 대안 보존
  - `yoon1` (기준선)
- **검증 완료**: 무누수(코드 검증), 거래비용 2배 견고, holdout 개봉, 워크포워드(SPY 대비
  연 21/27·롤링 24/25 — 단 survivor-30 기준), 생존편향 PIT 재검증(§2).
- **실거래 모듈**: `scripts/profile_sizing/live/` (rebalance.py 순수계산+단위테스트,
  kis_client.py KIS REST, kis_rebalance.py CLI, 기본 dry-run·모의투자). 토스는 공개 API
  없어 수동 체결용 주문표 생성 지원.
- **페이퍼 트레이딩**: `paper_trade.py`(목표 적립)+`paper_review.py`(채점). 저널
  `var/paper_trading/*.jsonl`(git 추적). 로컬 cron `0 13 * * 5`(금 22:00 KST)로 적립.
- 전체 테스트 109개 통과: `PYTHONPATH=src python -m unittest discover tests -q`.

### 핵심 파일
| 용도 | 경로 |
| --- | --- |
| 엔진(다종목) | `scripts/profile_sizing/portfolio.py` |
| 핸들러 | `src/trading_lab/strategies/profile_portfolio.py` |
| config | `configs/strategies/{yoon1,yoon1b,yoon1c}.json` |
| PIT | `scripts/profile_sizing/pit_universe.py`, `pit_backtest.py` |
| 실거래 | `scripts/profile_sizing/live/` |
| 리포트 | `reports/profile_sizing/*.md` (walk_forward, pit_backtest, meltup_analysis 등) |

---

## 2. 정직한 결론 (검증으로 확정된 것)

PIT(`reports/profile_sizing/pit_backtest.md`), 2013~ 기준:

| | CAGR | MDD | Sharpe |
| --- | ---: | ---: | ---: |
| 승자30 yoon1b (손픽, 과장) | 20.4% | -22.9% | 1.41 |
| **PIT yoon1b (현실적)** | 14.8% | -23.9% | 1.02 |
| SPY | 14.9% | -33.7% | 0.91 |
| PIT 등가중(공정 벤치) | 15.7% | -38.4% | 0.94 |

- 생존편향이 CAGR을 ~5.6%p, Sharpe를 ~0.39 부풀렸다. **vs SPY 수익우위는 소멸.**
- 살아남은 진짜 가치 = **낙폭 축소**(PIT yoon1b -23.9% vs 등가중 -38.4%, SPY -33.7%)
  + 소폭 위험조정 우위(Sharpe +0.08 vs 등가중). 일부는 노출이 낮아 생기는 기계적 효과.

---

## 3. 알려진 한계 / 함정 (반복하지 말 것)

1. **잔여 생존편향**: PIT도 yfinance에 **현재 S&P500 구성원만** 있어 상폐/편출 종목이
   빠짐 → PIT 수치도 약간 낙관 상한. (`pit_backtest.py` 로드 502/503, 편출 367 제외)
2. **미반영 실거래 비용**: 백테스트는 편도 10bp만. 한국 거주자 미국주식엔 양도세 22%,
   배당 원천징수 15%, 원↔달러 환전 스프레드, 실슬리피지 미반영 → 실수익 과대.
3. **다중검정**: 같은 데이터로 여러 변형을 탐색했다(gain·top_k·floor·섹터·인트라데이·
   크립토). 새 파라미터는 반드시 validation→test(과적합 방지) 규율로 고를 것.
4. **클라우드 데이터 차단**: 클라우드 샌드박스는 Yahoo를 403 차단한다(진단 확인). 클라우드
   자동화는 비-Yahoo 데이터원(Stooq/Tiingo 등)으로만 가능. 현재는 로컬 cron으로 운영.
5. **KIS 클라이언트 미검증**: TR_ID·응답 필드명은 KIS 문서 버전 의존, 실서버 자동테스트
   안 됨 → 모의투자 소량으로 먼저 검증 필수(`live/README.md`).

---

## 4. 제안 작업 (우선순위)

### P1. 잔여 생존편향 제거 — 상폐 종목 가격 확보
- 왜: §3-1. 현재 PIT는 상폐종목 누락으로 여전히 낙관. 이게 가장 큰 미해결 데이터 편향.
- 어떻게: yfinance 외 데이터원 도입(Stooq=무료·일부 상폐 보유, 또는 Norgate/CRSP=유료·
  완전 survivorship-free). `pit_universe.py`의 합집합(현재+편출 367종)으로 가격 확보 시도.
- 진입점: `pit_backtest.py`의 `load_cached` 를 새 데이터원으로 확장, 멤버십 마스크는 그대로.
- 완료기준: 편출종목 ≥100개 포함해 PIT 재실행, 방어우위(MDD/Sharpe)가 유지되는지 재확인.

### P2. PIT 기준으로 파라미터 재선택
- 왜: top_k=20·gain=1.25·floor=1.0은 **승자30에서** 골랐다(§3-3). PIT에선 최적이 다를 수 있음.
- 어떻게: `portfolio_sweep.py`·`return_boost_sweep.py`를 PIT 유니버스(마스킹)로 돌려
  validation 선정→test 확인. 선택기준은 **수익이 아니라 위험조정/낙폭**(전략 정체성).
- 완료기준: PIT val→test에서 견고한 파라미터 확정, 필요시 yoon1b config 갱신(+근거 리포트).

### P3. 현실적 비용·세금 모델
- 왜: §3-2. 실수익 판단의 핵심.
- 어떻게: 엔진 비용에 환전 스프레드·슬리피지 상향 옵션 추가, 사후 세금(양도세 22%·배당
  원천징수) 계산기. `engine.py`/`portfolio.py` 비용부 확장 또는 별도 after-tax 리포트.
- 완료기준: net-of-tax/net-of-FX CAGR을 리포트에 병기. 세후에도 방어가치가 의미 있는지 판단.

### P4. KIS 모의투자 실연동 검증
- 왜: §3-5. 실거래의 마지막 관문.
- 어떻게: KIS Developers 키 발급 후 `live/README.md` 절차대로 모의투자에서 잔고조회→
  주문표→소량 제출 round-trip. `kis_client.py`의 TR_ID·응답 필드를 최신 문서로 대조·수정.
- 완료기준: 모의투자 매수/매도 1건씩 정상 체결, balance 파싱 검증. 단위테스트로 파싱 고정.

### P5. 페이퍼 트레이딩 누적·채점 운영
- 왜: live 최종판정은 전진(out-of-sample) 기록이 필요.
- 어떻게: 로컬 cron 유지(머신 항상 켜기) 또는 비-Yahoo 데이터원으로 클라우드 복구.
  수주~수개월 후 `paper_review.py`로 실제 vs 백테스트 괴리 측정.
- 완료기준: 스냅샷 ≥8주 누적, 페이퍼 NAV가 백테스트 기대경로와 추적오차 작음 확인.

### P6. 포지셔닝/문서 정합
- 왜: §0. specs가 "시장 초과"로 읽히면 오해.
- 어떻게: `docs/strategy/specs/profile-portfolio-v1.md` 등에 PIT 결론(방어형, 시장수익±,
  낮은 낙폭) 반영, 과장 표현 제거.
- 완료기준: 모든 전략 문서가 정직한 포지셔닝으로 일관.

---

## 5. 작업 규약 (필수)

- 리포지토리 루트 `CLAUDE.md`의 전략 추가/수정 계약을 반드시 따를 것(공통 인프라 수정 금지).
- 마무리 전 `PYTHONPATH=src python -m unittest discover tests -q` 통과.
- 비밀키는 `.env`(gitignore)만. `var/`는 페이퍼 저널 외 추적 금지.
- 결과 보고는 §0 원칙대로 정직하게. 손픽/생존편향 수치를 시장초과로 포장하지 말 것.
