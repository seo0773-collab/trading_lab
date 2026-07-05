# 체크리스트: yoon_profile_1

- [x] 숏 후보 선별 (`yoon_profile_short_study.md`) — (0,0) 칼 숏만 생존, (2,1) 감속 숏 기각
- [x] 스퀴즈 가드 선정 (`yoon_profile_squeeze_guard_study.md`) — trail15 채택, ma20·trail10·fast_exit 기각
- [x] 파이프라인 숏 확장 (`short` config 블록, 기본 off → yoon_profile 동작 불변 확인: all/val/test 수치 동일)
- [x] 양방향 거래 분해 (`signed_lot_trades`) + `squeeze_guard` exit_reason 라벨 (presentation.py)
- [x] config + registry 등록 (핸들러는 yoon_profile과 공유)
- [x] 실데이터 검증: 스터디 수치 재현 (all +2336%/0.902, test +23.1%/0.484) · 숏 14건 확인
- [x] 명세 작성
- [x] 계약/전체 테스트 통과 (165개 OK) + CLI 합성 run succeeded (run 403)
- [ ] 다음 하락 사이클 OOS에서 숏 슬리브 기여 재판정 (음수면 폐기)
- [ ] 선물 펀딩비용 모델링 (실거래 승격 전 필수)
