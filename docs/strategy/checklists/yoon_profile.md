# 체크리스트: yoon_profile

- [x] 데이터: BTC/USDT 일봉 2017-08-17~2026-07-03 재수집 (ccxt binance, 갭 0)
- [x] 지표 모듈 (`cycle_state.py`) — 플랜 스펙 그대로, expanding percentile bin(무누수)
- [x] 2D 조건부 스터디 (`yoon_profile_study.py`) — NW t-stat·Bonferroni/FDR·walk-forward·H1~H3 판정
- [x] 가설 판정: H1 변형 채택((0,1) 눌림목) / H2 보류(방향 일치) / H3 bin 관점 부분 채택 / 2D 추가성 채택
- [x] 파이프라인 (`yoon_profile.py`) + 핸들러 (`YoonProfileHandler`) + config + registry 등록
- [x] 실데이터 백테스트: all/validation/test 전 phase B&H 대비 위험조정 우위 (명세 §5)
- [x] 명세 작성
- [ ] 계약/전체 테스트 통과 + CLI 합성 run succeeded
- [x] 숏 확장 검증 (`yoon_profile_short_study.md`) — (2,1) 감속 숏 기각(holdout 붕괴), (0,0) 칼 숏 −0.5 보류(엣지=2022 단일 에피소드), long-only 유지
- [x] 숏 스퀴즈 가드 검증 (`yoon_profile_squeeze_guard_study.md`) — **trail15 채택 후보**(test +23.1%/Sharpe 0.487로 long-only 상회, 기여 에피소드 2개로 확대). ma20·fast_exit 기각. config `short` 옵션화는 미적용(기본 off 권장)
- [ ] 신규 데이터 1년 누적 후 스터디 재실행 (셀 부호 안정성 재검증 게이트)
- [ ] 멀티심볼 일반화 (ETH 등) — 후속
