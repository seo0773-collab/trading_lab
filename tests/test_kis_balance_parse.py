"""KIS 잔고 예수금 파싱(_pick_amount) 안전화 회귀 테스트.

기존 ``a or b or 0.0`` 무음 폴백이 스키마 변경 시 cash=0 → 과/소 주문으로
직결되던 위험을 막는 명시 파서. 핵심 계약:
  - 기대 필드가 있으면 그 값(0/"0" 포함)을 쓴다.
  - 기대 필드가 통째로 없으면 KISSchemaError(조용히 0 으로 넘어가지 않음).
"""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from profile_sizing.live.kis_client import (  # noqa: E402
    KISConfigError,
    KISSchemaError,
    _integer_qty,
    _pick_amount,
)


class TestPickAmount(unittest.TestCase):
    FIELDS = ("dnca_tot_amt", "prvs_rcdl_excc_amt")

    def test_first_field_used(self):
        out2 = {"dnca_tot_amt": "1234567", "prvs_rcdl_excc_amt": "999"}
        self.assertEqual(_pick_amount(out2, self.FIELDS, "x"), 1234567.0)

    def test_fallback_to_second(self):
        # 첫 필드 키가 없을 때 두 번째 사용
        out2 = {"prvs_rcdl_excc_amt": "500"}
        self.assertEqual(_pick_amount(out2, self.FIELDS, "x"), 500.0)

    def test_present_zero_is_valid(self):
        # 값이 0 이면 '정상적 0 예수금' — 폴백/예외 없이 0 반환
        out2 = {"dnca_tot_amt": "0", "prvs_rcdl_excc_amt": "0"}
        self.assertEqual(_pick_amount(out2, self.FIELDS, "x"), 0.0)

    def test_empty_string_skips_to_next(self):
        out2 = {"dnca_tot_amt": "", "prvs_rcdl_excc_amt": "777"}
        self.assertEqual(_pick_amount(out2, self.FIELDS, "x"), 777.0)

    def test_all_empty_returns_zero(self):
        # 기대 키는 존재하나 모두 빈값 → 0 (스키마는 맞음)
        out2 = {"dnca_tot_amt": "", "prvs_rcdl_excc_amt": None}
        self.assertEqual(_pick_amount(out2, self.FIELDS, "x"), 0.0)

    def test_schema_mismatch_raises(self):
        # 기대 필드가 통째로 없음 = TR_ID/스키마 불일치 → 예외(조용한 0 금지)
        out2 = {"some_other_amt": "123", "totalAssets": "456"}
        with self.assertRaises(KISSchemaError):
            _pick_amount(out2, self.FIELDS, "국내잔고 예수금")

    def test_schema_error_lists_keys(self):
        out2 = {"foo": "1", "bar": "2"}
        try:
            _pick_amount(out2, self.FIELDS, "해외잔고 예수금")
        except KISSchemaError as e:
            self.assertIn("해외잔고 예수금", str(e))
            self.assertIn("foo", str(e))  # 디버깅용 가용 키 노출
        else:
            self.fail("KISSchemaError 가 발생해야 함")


class TestIntegerQty(unittest.TestCase):
    def test_integer_qty_accepts_whole_number(self):
        self.assertEqual(_integer_qty(3.0), 3)

    def test_integer_qty_rejects_fractional(self):
        with self.assertRaises(KISConfigError):
            _integer_qty(1.25)


if __name__ == "__main__":
    unittest.main()
