# Research And Legacy Layout

The platform package lives under `src/trading_lab`.

Existing research code remains in place during the migration:

- `scripts/flat_chart.py`, `scripts/kalman.py`, and
  `scripts/strategy_execution.py` are the validated research engine currently
  used by the platform adapter.
- `indicators/`, `strategies/cycle_reversion.py`, and the old `dashboard/`
  are legacy Wave Viewer research components. They are not registered as
  platform strategies because the full-interval profile has look-ahead bias.
- `scripts/*_1.py` are historical copies. Remove them only after the new
  package-level regression suite fully replaces their reference value.
- Existing `reports/`, `data/processed/`, and `logs/` are preserved. New
  platform runs write only to ignored `var/`.

No existing research artifact is automatically deleted or moved.
