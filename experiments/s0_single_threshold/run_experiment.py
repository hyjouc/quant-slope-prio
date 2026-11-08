"""Stage 1 has no authorized backtest. Prevent accidental research runs."""
raise SystemExit('Stage 1 audit only: research lead review required before implementing or running S0 backtests.')
