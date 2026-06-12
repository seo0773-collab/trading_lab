"""Scaffold a new strategy: config, spec, and gate checklist.

Usage:
    python scripts/new_strategy.py my-strategy-v1 --description "..."
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs" / "strategies"
STRATEGY_DOCS = ROOT / "docs" / "strategy"
SPEC_TEMPLATE = STRATEGY_DOCS / "STRATEGY_SPEC_TEMPLATE.md"
MASTER_CHECKLIST = STRATEGY_DOCS / "STRATEGY_TEST_CHECKLIST.md"

# 모든 전략 config가 갖춰야 하는 키. 실행 어댑터(research_adapter.py)가
# 직접 조회하는 키와 검증 분할에 필요한 키의 합집합이다.
REQUIRED_CONFIG_KEYS = (
    "version",
    "interval",
    "period",
    "horizon",
    "direction",
    "confidence_quantile",
    "quantile_window",
    "cycle_len",
    "fast_window",
    "slow_window",
    "fee_bps_per_side",
    "execution",
    "exit_on_opposite",
    "long_only",
    "identification_frac",
    "validation_frac",
)

DEFAULT_CONFIG = {
    "version": "",
    "interval": "1h",
    "period": "720d",
    "horizon": 72,
    "direction": "PRICE",
    "confidence_quantile": 0.85,
    "quantile_window": 2000,
    "cycle_len": 200,
    "fast_window": 120,
    "slow_window": 720,
    "slope_span": 24,
    "slope_mode": "linear",
    "fee_bps_per_side": 10.0,
    "execution": "next_open",
    "exit_on_opposite": True,
    "long_only": False,
    "identification_frac": 0.4,
    "validation_frac": 0.3,
}

REGISTRY_SNIPPET = '''    "{strategy_id}": StrategyDefinition(
        strategy_id="{strategy_id}",
        version="1",
        description="{description}",
        config_path=ROOT / "configs" / "strategies" / "{config_file}",
        enabled=True,
        live_eligible=False,
    ),'''


def validate_config(config: dict) -> list[str]:
    """Return the list of required keys missing from a strategy config."""
    return [key for key in REQUIRED_CONFIG_KEYS if key not in config]


def config_filename(strategy_id: str) -> str:
    return strategy_id.replace("-", "_") + ".json"


def scaffold(
    strategy_id: str, description: str, *,
    base_config: Path | None = None, force: bool = False,
) -> dict[str, Path]:
    if not re.fullmatch(r"[a-z][a-z0-9]*(-[a-z0-9]+)*", strategy_id):
        raise ValueError(
            "strategy_id must be kebab-case, e.g. my-strategy-v1"
        )

    config = dict(DEFAULT_CONFIG)
    if base_config is not None:
        config.update(json.loads(base_config.read_text(encoding="utf-8")))
    config["version"] = strategy_id
    missing = validate_config(config)
    if missing:
        raise ValueError(f"base config missing required keys: {missing}")

    config_path = CONFIG_DIR / config_filename(strategy_id)
    spec_path = STRATEGY_DOCS / "specs" / f"{strategy_id}.md"
    checklist_path = STRATEGY_DOCS / "checklists" / f"{strategy_id}.md"

    if not force:
        for path in (config_path, spec_path, checklist_path):
            if path.exists():
                raise FileExistsError(f"already exists: {path}")

    spec = SPEC_TEMPLATE.read_text(encoding="utf-8")
    spec = (
        spec.replace("{STRATEGY_ID}", strategy_id)
        .replace("{DESCRIPTION}", description)
        .replace("{DATE}", dt.date.today().isoformat())
        .replace("{CONFIG_FILE}", config_path.name)
        .replace("{IDENT_FRAC}", str(config["identification_frac"]))
        .replace("{VALID_FRAC}", str(config["validation_frac"]))
    )

    checklist = MASTER_CHECKLIST.read_text(encoding="utf-8")
    header = (
        f"# 테스트 체크리스트: {strategy_id}\n\n"
        f"- 생성일: {dt.date.today().isoformat()}\n"
        f"- 명세: specs/{strategy_id}.md\n"
        f"- 각 항목 통과 시 증거(run_id 또는 리포트 경로)를 같은 줄에 기록\n\n"
    )
    checklist = header + checklist.split("\n", 1)[1].lstrip("\n")

    spec_path.parent.mkdir(parents=True, exist_ok=True)
    checklist_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    spec_path.write_text(spec, encoding="utf-8")
    checklist_path.write_text(checklist, encoding="utf-8")
    return {
        "config": config_path,
        "spec": spec_path,
        "checklist": checklist_path,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("strategy_id", help="kebab-case id, e.g. my-strategy-v1")
    parser.add_argument("--description", default="", help="one-line description")
    parser.add_argument(
        "--base-config", type=Path, default=None,
        help="existing config JSON to copy defaults from",
    )
    parser.add_argument("--force", action="store_true", help="overwrite existing files")
    args = parser.parse_args()

    try:
        paths = scaffold(
            args.strategy_id, args.description,
            base_config=args.base_config, force=args.force,
        )
    except (ValueError, FileExistsError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    for name, path in paths.items():
        print(f"{name}: {path.relative_to(ROOT)}")
    print("\nregistry.py 의 _STRATEGIES 에 추가:\n")
    print(REGISTRY_SNIPPET.format(
        strategy_id=args.strategy_id,
        description=args.description or args.strategy_id,
        config_file=paths["config"].name,
    ))
    print("\n다음 단계: 명세의 '미정' 항목을 채운 뒤 Gate 0부터 진행하세요.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
