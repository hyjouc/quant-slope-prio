"""Execution tests deferred: no backtest engine in audit-only stage."""
import pytest

@pytest.mark.skip(reason='Stage 1 audit only; implement executor after research lead review')
def test_execution_acceptance_pending():
    pass
