# 체크리스트: yoon_profile_3

- [x] 가설 발굴 — 2D 상태 필드 궤적 실험 (`state_field_trend_analysis.md` §6, 셀×경로 스크리닝)
- [x] 사전 등록 정식 검증 (`yoon_profile_path_study.md`) — PA(M1A1 분리) 채택: t 2.32/2.10, WF 3/3 · PB(M2A1 하강→0) 기각: WF 1/3
- [x] 파이프라인 `path_tilt` 블록 (기본 off → base yoon_profile 동작 불변, 단위 테스트 2개)
- [x] 백테스트 게이트 — BTC holdout 수익/Sharpe/MDD 모두 개선 (+18.5%/0.439/−20.2%)
- [x] config + registry 등록 (핸들러는 yoon_profile과 공유)
- [x] 명세 작성
- [x] 계약/전체 테스트 통과 (174개 OK) + CLI 합성 run succeeded (run 439)
- [x] 멀티에셋(크립토·ETF) 이식 검증 (`yoon_profile_3_multiasset_eval.md`) — BTC 채택 확정 · SOL은 base 이식 성립(tilt 역기여 −4.4%p) · ETH 판정보류(방어 우위만) · ETF(SPY/QQQ/GLD) 기각 → **path_tilt는 BTC 특화**
- [ ] 1년 데이터 누적 후 path 스터디 재실행 (delta 부호 반전 시 폐기)
