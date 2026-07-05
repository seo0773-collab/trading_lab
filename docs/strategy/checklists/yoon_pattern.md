# 체크리스트: yoon_pattern

- [x] 가설 이벤트 스터디 (analog vs self × L{150,300} × H{5,21}, SPY·QQQ) → analog L300/H21 채택
- [x] 무누수 규약: s+H ≤ t 후보 제한 + 매치 간격 gap + 전 필터 인과 (스터디·파이프라인 동일)
- [x] 피처/파이프라인 (`pattern_projection.py`, walk_forward_expectancy 재사용)
- [x] 핸들러 (`YoonPatternHandler`) + config + registry 등록
- [x] 명세 작성
- [x] 계약/전체 테스트 통과 (165개 OK) + CLI 합성 run succeeded (run 400)
- [x] SPY 실데이터 스모크 (all/validation/test 전부 양수, `reports/profile_sizing/yoon_pattern_spy_smoke.md`)
- [x] L(패턴 길이)·M(매치 수) 스윕 — **미확정**(val↔test 순위 역전, L300/M50 유지, `yoon_pattern_sweep.md`)
- [x] MDD 오버레이(변동성 타게팅 15%) `risk_overlay` config 옵션 구현 + 기본 ON
- [x] 멀티심볼 검증 (SPY·QQQ·IWM·EFA) → **L=450 확정** (유일한 전 칸 양수), config 반영
- [x] 확정 기본값 재검증: 계약 테스트 OK, SPY val 0.694/-11.1% · test 0.763/-14.6%
- [ ] holdout test 구간 무효화 조건 점검 (수익<0 또는 IC≤0이면 폐기)
