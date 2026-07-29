"""Tests for the buffered, failure-tolerant wandb logger (src/wandb_logging.py).

No network and no wandb run: a fake run records the calls a real wandb Run would receive, so the
chunking policy and the "a wandb error must never reach training" contract are checked directly.
"""

import numpy as np
import pytest

from src.wandb_logging import WandbLogger


class FakeRun:
    """Minimal stand-in for a wandb Run: records log()/summary/finish, can be made to fail."""

    def __init__(self, fail_logs: int = 0):
        self.id = "fake123"
        self.rows = []           # [(step, payload)] in the order wandb received them
        self.summaries = {}
        self.finished = False
        self._fail_logs = fail_logs        # fail this many log() calls before succeeding
        self.summary = self._Summary(self)

    class _Summary:
        def __init__(self, run):
            self._run = run

        def update(self, values):
            self._run.summaries.update(values)

    def log(self, payload, step=None, commit=True):
        if self._fail_logs > 0:
            self._fail_logs -= 1
            raise RuntimeError("HTTP 429: rate limit exceeded")
        self.rows.append((step, dict(payload)))

    def finish(self):
        self.finished = True


def _logger(run, **kwargs):
    kwargs.setdefault("flush_every", 100)
    kwargs.setdefault("flush_interval_s", 1e9)   # time-based flush off unless a test asks for it
    kwargs.setdefault("retry_base_s", 0.0)       # no real sleeping in tests
    kwargs.setdefault("retry_max_s", 0.0)
    return WandbLogger(run, **kwargs)


def test_first_row_is_pushed_immediately():
    """A run must show up on the dashboard with data right away, not only after 100 iterations."""
    run = FakeRun()
    wb = _logger(run)
    wb.log({"train/mmd_train": 0.5}, step=0)
    assert run.rows == [(0, {"train/mmd_train": 0.5})]


def test_chunking_holds_then_pushes_every_row():
    """Rows are buffered until the chunk is full, then ALL of them are pushed, in step order."""
    run = FakeRun()
    wb = _logger(run, flush_every=10)
    wb.log({"i": 0}, step=0)                      # first row flushes immediately
    assert len(run.rows) == 1

    for step in range(1, 10):
        wb.log({"i": step}, step=step)
    assert len(run.rows) == 1, "rows must be buffered, not sent one by one"

    wb.log({"i": 10}, step=10)                    # 10 buffered -> chunk is full
    assert len(run.rows) == 11
    assert [step for step, _ in run.rows] == list(range(11)), "steps must arrive in order"
    assert [payload["i"] for _, payload in run.rows] == list(range(11)), "no row may be dropped"


def test_flush_pushes_partial_chunk():
    run = FakeRun()
    wb = _logger(run, flush_every=100)
    for step in range(5):
        wb.log({"i": step}, step=step)
    wb.flush(force=True)
    assert [step for step, _ in run.rows] == [0, 1, 2, 3, 4]


def test_time_based_flush():
    """A run with slow iterations must still appear long before its chunk fills."""
    run = FakeRun()
    wb = _logger(run, flush_every=100, flush_interval_s=0.0)
    for step in range(3):
        wb.log({"i": step}, step=step)
    assert len(run.rows) == 3


def test_transient_failure_is_retried():
    run = FakeRun(fail_logs=2)
    wb = _logger(run, flush_every=1, retries=3)
    wb.log({"i": 0}, step=0)
    assert run.rows == [(0, {"i": 0})], "should succeed on the 3rd attempt"
    assert wb.enabled


def test_persistent_failure_disables_wandb_but_never_raises():
    """The contract: a sustained rate limit costs logging, not the training run."""
    run = FakeRun(fail_logs=10_000)
    wb = _logger(run, flush_every=1, retries=1, max_failures=2)
    for step in range(50):
        wb.log({"i": step}, step=step)           # must not raise, ever
    assert run.rows == []
    assert not wb.enabled, "wandb should be dropped after max_failures consecutive failures"
    wb.summary({"best_mmd_val": 0.1})            # still no raising once disabled
    wb.finish()


def test_disabled_logger_is_a_noop():
    """run=None (wandb disabled, or init failed) -- callers should not have to branch on it."""
    wb = _logger(None)
    assert not wb.enabled and wb.run_id is None
    assert wb.log({"i": 0}, step=0) == 0.0
    assert wb.summary({"x": 1}) is False
    assert wb.log_artifact("a", ["nonexistent.file"]) is False
    wb.flush(force=True)
    wb.finish()


def test_finish_flushes_and_closes():
    run = FakeRun()
    wb = _logger(run, flush_every=100)
    wb.log({"i": 0}, step=0)
    wb.log({"i": 1}, step=1)
    wb.summary({"best_mmd_val": 0.25})
    wb.finish()
    assert [step for step, _ in run.rows] == [0, 1]
    assert run.summaries == {"best_mmd_val": 0.25}
    assert run.finished


def test_payload_values_survive_untouched():
    """Buffering copies rows; NaNs and numpy scalars must pass through unchanged."""
    run = FakeRun()
    wb = _logger(run, flush_every=2)
    wb.log({"a": np.float64(1.5), "b": np.nan}, step=0)
    _, payload = run.rows[0]
    assert payload["a"] == pytest.approx(1.5)
    assert np.isnan(payload["b"])
