"""실거래 보조 모듈 (yoon1b 운영용).

신호 생성(백테스트/페이퍼)과 분리된 '실행' 레이어. 핵심은 브로커 무관 순수 주문
계산(rebalance.py)이고, KIS 연동(kis_client.py)은 그 위의 I/O 어댑터다.
실돈 안전을 위해 기본은 dry-run·모의투자다. README.md 참고.
"""
