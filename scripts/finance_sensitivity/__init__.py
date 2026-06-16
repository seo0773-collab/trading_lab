"""Financial-statement sensitivity strategy research package (finance_plan.txt).

종목별 재무 팩터 변화 → 발표 후 주가 반응의 민감도를 rolling Ridge로 학습해
20일/60일 예상 수익률을 산출한다. 무거운 로직은 이 패키지에, 공통 대시보드와
잇는 얇은 핸들러는 ``src/trading_lab/strategies/fin_sensitivity.py``에 둔다
(DI 전략과 동일한 분리). 현재 단계 = 백테스트 전: 데이터/모델/신호 모듈까지.
"""
from __future__ import annotations

from .config import FinSensitivityConfig, config_from_dict

__all__ = ["FinSensitivityConfig", "config_from_dict"]
