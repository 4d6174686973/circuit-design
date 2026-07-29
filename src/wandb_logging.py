"""Rate-limit-tolerant wandb logging, sized for sweeps with hundreds of concurrent runs.

Why this module exists
----------------------
W&B rate-limits requests per API key/project, and every ONLINE run costs requests largely
independently of how much we log: `wandb.init()` (a handful of GraphQL mutations), a periodic
filestream POST carrying whatever history rows have accumulated, a keepalive so the backend doesn't
mark the run crashed, and finish/artifact mutations at the end. The request rate therefore scales
with the number of CONCURRENT runs, which is why a sweep that is healthy at 8 parallel runs starts
returning HTTP 429 ("rate limit exceeded") at 100+: as a hard error out of init/artifact calls
(which used to kill the training run) or as throttled keepalives, which makes the dashboard show
live runs as "crashed".

Three levers are combined here so run-level parallelism can go up to the node's core count:

1. BUFFERED CHUNK LOGGING (`WandbLogger`) -- per-iteration metric rows are accumulated in-process
   and pushed as one chunk (default: every 100 iterations, or after flush_interval_s for slow runs).
   Nothing is downsampled or dropped: every iteration still gets its own history row and shows up on
   the same step it was measured at. What changes is that a run's data leaves the process in a few
   large bursts instead of a trickle, so most of wandb-core's transmit ticks have nothing to send
   and issue no request at all.
2. TUNED TRANSPORT (`build_settings`) -- a long filestream transmit interval (so one chunk becomes
   ~one request instead of one request per tick, and that interval is also the per-run request floor,
   since the same channel carries the run's liveness), generous startup timeouts, and no
   system-stats/metadata traffic. wandb-core's own HTTP retry budgets are left ALONE -- see the note
   in build_settings for why shortening them drops run data.
3. FAILURE TOLERANCE (`init_run`, `WandbLogger._attempt`) -- init is staggered across concurrent
   runs and retried with jittered exponential backoff, and every wandb call is wrapped so a 429 (or
   any other wandb error) degrades logging instead of killing training. After `max_failures`
   consecutive failures the run stops talking to wandb altogether and keeps training: losses,
   per-eval-step benchmarks, parameter history and both checkpoints are written to the run's output
   directory by QCBM.save() regardless of what wandb does.

Nothing in this module may raise into training -- that is the whole point of it.

Sweep lifecycle (`set_sweep_state`)
-----------------------------------
The grid is driven by Hydra, and runs are attached to the Sweep object explicitly (see
`build_settings`), so `wandb agent` is never involved. A Sweep's state, however, is only ever
advanced by an agent/scheduler -- so without an explicit transition the sweep sits in PENDING
forever in the UI, even after all of its runs have finished. `set_sweep_state` performs that
transition (RUNNING at creation, FINISHED when the launching process exits), mirroring what wandb's
own launch scheduler does in wandb/sdk/launch/sweeps/scheduler.py::_set_sweep_state.
"""

import logging
import os
import random
import time


def _log():
    return logging.getLogger("QCBM")


# --------------------------------------------------------------------------------------------------
# run setup
# --------------------------------------------------------------------------------------------------
def build_settings(cfg, sweep_id: str):
    """wandb.Settings for one run: join the sweep, then tune the transport for many parallel runs.

    Joining the sweep is an EXPLICIT `sweep_id` override rather than relying on wandb.init() picking
    up the WANDB_SWEEP_ID env var: wandb.sweep() (in setup.get_or_create_wandb_sweep) triggers a
    login that snapshots os.environ into a process-wide Settings singleton BEFORE we set
    WANDB_SWEEP_ID, so every later wandb.init() in that process (e.g. every sequential Hydra job
    when runs_batch_size == 1, since BasicLauncher reuses one process) would silently reuse that
    stale, sweep-less snapshot -- which is exactly what produced runs that existed but were not
    attached to the sweep. A per-call override is applied on top of that singleton, so it is correct
    regardless of caching.

    The x_* knobs are wandb-core internals; unknown ones are filtered out below so a different wandb
    version degrades to "fewer knobs applied" instead of failing every run at init.
    """
    import wandb

    options = {
        "sweep_id": sweep_id,
        # Overhead control: with many parallel runs, the per-run system-stats monitor (a thread
        # polling CPU/mem plus periodic posts) and the metadata/code/git scans cost N times as much
        # for no benefit -- we only log our own scalar metrics.
        "x_disable_stats": True,
        "x_disable_meta": True,
        "disable_git": True,
        "disable_code": True,
        "quiet": True,                 # one banner per run x hundreds of runs is unreadable anyway
        # Startup: hundreds of near-simultaneous inits make both the local core handshake and the
        # backend slower to answer, so give them more room than the 90s/30s defaults.
        "init_timeout": float(cfg.logging.wandb_init_timeout_s),
        "x_service_wait": float(cfg.logging.wandb_service_wait_s),
        # Transmission: the interval is what turns a buffered chunk into ~one request, and it also
        # sets the per-run request FLOOR -- the filestream posts periodically to keep the backend from
        # marking a live run crashed, so liveness and data ride the same channel. Raising it means the
        # dashboard lags by up to that much, which is fine at chunk granularity.
        # (Settings.heartbeat_seconds is deliberately not set: in wandb-core every heartbeat code path
        # belongs to `wandb leet`, the terminal UI, and the legacy Python filestream takes its
        # keepalive period from the SERVER's dynamic settings, not from this field. It is not the knob
        # it looks like.)
        "x_file_stream_transmit_interval": float(cfg.logging.wandb_transmit_interval_s),
    }

    fields = getattr(wandb.Settings, "model_fields", None)
    if fields:
        unknown = [k for k in options if k not in fields]
        if unknown:
            _log().warning(f"[wandb] ignoring settings unsupported by wandb {wandb.__version__}: "
                           f"{', '.join(unknown)}")
            options = {k: v for k, v in options.items() if k not in unknown}
    return wandb.Settings(**options)


def _reinit_value():
    """The `reinit` argument for wandb.init().

    A pool worker trains several runs in sequence, so any previous run in the process must be
    FINISHED rather than returned. The boolean form still does that but is deprecated -- and warns
    once per run, which is unreadable at this fan-out. The explicit string form landed in wandb 0.19;
    fall back to the boolean on anything older/unparseable.
    """
    import wandb

    try:
        if tuple(int(part) for part in wandb.__version__.split(".")[:2]) >= (0, 19):
            return "finish_previous"
    except Exception:
        pass
    return True


def stagger(cfg, seed: int = 0, what: str = "") -> float:
    """Sleep a random slice of a concurrency-scaled window; returns the seconds slept.

    Used around the two moments where every concurrent run wants the API at the same instant: start
    (init) and end (artifact upload + finish). Runs in a sweep do near-identical work, so without
    this they arrive in lockstep -- W simultaneous request bursts, which is what actually trips the
    rate limit. The window is wandb_init_stagger_s seconds per concurrent run, so an 8-way sweep
    barely waits while a 300-way sweep spreads the same burst over minutes.

    Uses its own RNG: the global numpy/random state is part of a run's reproducibility (see
    extension.random_topology and the minibatch shuffle in QCBM), so jitter must not touch it.
    """
    if cfg.logging.wandb_mode != "online":
        return 0.0
    window = float(cfg.logging.wandb_init_stagger_s) * max(1, int(cfg.sweep.max_parallel_runs or 1))
    if window <= 0:
        return 0.0
    delay = random.Random((int(seed or 0) << 16) ^ os.getpid() ^ hash(what)).uniform(0.0, window)
    time.sleep(delay)
    return delay


def init_run(cfg, sweep_id: str, seed: int):
    """Start one wandb run -- staggered, retried, and never fatal. Returns the Run or None.

    None means "no wandb for this run": either wandb_mode == "disabled", or init kept failing (e.g. a
    sustained 429 at the start of a large sweep). Training proceeds either way; the caller wraps the
    result in a WandbLogger, which is a no-op when the run is None.

    The stagger (see `stagger`) matters as much as the retries: W workers otherwise call init()
    within milliseconds of each other, and that burst of GraphQL mutations is the most 429-prone
    moment of a whole sweep.
    """
    if cfg.logging.wandb_mode == "disabled":
        return None

    import wandb
    from omegaconf import OmegaConf

    logger = _log()
    online = cfg.logging.wandb_mode == "online"
    rng = random.Random((int(seed or 0) << 16) ^ os.getpid())
    stagger(cfg, seed, "init")

    settings = build_settings(cfg, sweep_id)
    entity = cfg.logging.wandb_entity
    config = OmegaConf.to_container(cfg, resolve=True)
    attempts = (max(0, int(cfg.logging.wandb_init_retries)) + 1) if online else 1
    delay = 5.0

    for attempt in range(1, attempts + 1):
        try:
            run = wandb.init(
                project=cfg.logging.wandb_project,
                entity=entity if entity else None,
                # no explicit name/group: wandb's generated name is fine and the swept params live in
                # the run's config, which is what plotting/benchmarking filter on (also sidesteps
                # wandb's 128-char GroupName limit).
                job_type="train",
                config=config,
                mode=cfg.logging.wandb_mode,
                settings=settings,
                reinit=_reinit_value(),
            )
            url = getattr(run, "url", None)
            logger.info(f"wandb run {run.id} ready{f' ({url})' if url else ''}")
            return run
        except Exception as exc:
            if attempt >= attempts:
                logger.warning(f"[wandb] init failed after {attempt} attempt(s) ({exc!r}); this run "
                               f"trains WITHOUT wandb -- metrics and checkpoints are still written "
                               f"to its output directory.")
                return None
            wait = delay + rng.uniform(0.0, delay)
            logger.warning(f"[wandb] init attempt {attempt}/{attempts} failed ({exc!r}); "
                           f"retrying in {wait:.0f}s")
            time.sleep(wait)
            delay = min(delay * 2, float(cfg.logging.wandb_retry_wait_max_s))


# --------------------------------------------------------------------------------------------------
# buffered, failure-tolerant metric logging
# --------------------------------------------------------------------------------------------------
class WandbLogger:
    """Buffers per-iteration metric rows and pushes them in chunks; swallows every wandb error.

    Usage mirrors the parts of the wandb Run API this project needs::

        wb = WandbLogger(run, flush_every=100, flush_interval_s=300)
        wb.log({"train/mmd_train": ...}, step=7)   # buffered; pushed when the chunk is full
        wb.summary({"best_mmd_val": ...})
        wb.log_artifact("qcbm_abc", [f"{d}/circuit.qpy"], metadata={...}, aliases=["seed42"])
        wb.finish()                                 # final flush + run.finish()

    Chunking (`flush_every` rows) is what keeps request volume flat as concurrency grows; the
    `flush_interval_s` fallback exists so a run with slow iterations still appears on the dashboard
    long before its chunk fills, and the very first row is always pushed immediately so a run shows
    up as alive-with-data seconds after it starts (crash visibility, not just "running").

    A `run` of None (wandb disabled, or init failed) makes every method a cheap no-op, so callers
    never need to branch on whether wandb is active.
    """

    def __init__(self, run, *, flush_every: int = 100, flush_interval_s: float = 300.0,
                 max_failures: int = 5, retries: int = 3, retry_base_s: float = 2.0,
                 retry_max_s: float = 60.0, label: str = ""):
        self._run = run
        self._flush_every = max(1, int(flush_every))
        self._flush_interval_s = float(flush_interval_s)
        self._max_failures = max(1, int(max_failures))
        self._retries = max(0, int(retries))
        self._retry_base_s = float(retry_base_s)
        self._retry_max_s = float(retry_max_s)
        self._label = label or (getattr(run, "id", None) or "off")
        self._log = _log()

        self._rows: list[tuple[int, dict]] = []
        self._last_flush = time.time()
        self._flushes = 0
        self._pushed = 0
        self._dropped = 0
        self._failures = 0            # CONSECUTIVE failures; reset by any successful push
        self._disabled = run is None
        self._rng = random.Random(os.getpid())
        self.seconds = 0.0            # cumulative wall-clock spent inside wandb calls

    # -- properties ---------------------------------------------------------------------------
    @property
    def enabled(self) -> bool:
        return not self._disabled

    @property
    def run_id(self):
        return getattr(self._run, "id", None)

    # -- metric logging -----------------------------------------------------------------------
    def log(self, payload: dict, step: int) -> float:
        """Buffer one step's metrics, pushing the chunk if it is full/stale/first.

        Returns the seconds this call took: ~0 for a plain buffer append, the push cost on the steps
        where a flush happened. The caller can log that number as training telemetry (see
        QCBM.stochastic_gradient_descent, which reports it on the following step -- a call cannot
        include a measurement of itself).
        """
        if self._disabled:
            return 0.0
        start = time.time()
        self._rows.append((int(step), dict(payload)))
        if (self._flushes == 0                                          # first row: show up now
                or len(self._rows) >= self._flush_every
                or (start - self._last_flush) >= self._flush_interval_s):
            self.flush()
        return time.time() - start

    def flush(self, force: bool = False) -> float:
        """Push all buffered rows as one chunk. `force` is accepted for call-site clarity; a flush
        always pushes everything buffered. Returns seconds spent; never raises.

        Rows are handed to wandb in ascending step order (wandb requires non-decreasing steps). If a
        row fails even after its retries, the rest of the chunk is dropped rather than paying the
        full retry budget per row: whatever broke (a dead core connection, a sustained rate limit)
        would fail identically for every remaining row, and a run that stalls for minutes per step
        on wandb retries is worse than a run with a gap in its history.
        """
        if self._disabled or not self._rows:
            self._last_flush = time.time()
            return 0.0

        start = time.time()
        rows, self._rows = self._rows, []
        for i, (step, payload) in enumerate(rows):
            if not self._attempt(lambda: self._run.log(payload, step=step, commit=True),
                                 f"log(step={step})"):
                self._dropped += len(rows) - i
                self._note_failure()
                break
            self._pushed += 1
        else:
            self._failures = 0

        self._flushes += 1
        self._last_flush = time.time()
        dt = self._last_flush - start
        self.seconds += dt
        return dt

    # -- run-level extras ---------------------------------------------------------------------
    def summary(self, values: dict) -> bool:
        """One batched summary update (three separate assignments would be three server updates)."""
        if self._disabled:
            return False
        return self._attempt(lambda: self._run.summary.update(dict(values)), "summary.update")

    def log_artifact(self, name: str, files: list, *, type_: str = "model",
                     metadata: dict = None, aliases: list = None) -> bool:
        """Upload one artifact. Failure is logged and swallowed -- the same files stay on disk in the
        run's output directory, and benchmark.load_checkpoint falls back to that path."""
        if self._disabled:
            return False
        import wandb

        def _upload():
            artifact = wandb.Artifact(name, type=type_, metadata=metadata or {})
            for path in files:
                artifact.add_file(path)
            self._run.log_artifact(artifact, aliases=aliases or None)

        return self._attempt(_upload, f"log_artifact({name})")

    def finish(self) -> None:
        """Final flush, then close the run. Reports what wandb logging actually cost/lost."""
        if self._run is None:
            return
        self.flush(force=True)
        if not self._disabled:
            self._attempt(lambda: self._run.finish(), "finish")
        self._log.info(f"[wandb:{self._label}] pushed {self._pushed} metric rows in {self._flushes} "
                       f"chunk(s), dropped {self._dropped}, {self.seconds:.1f}s spent in wandb")
        self._disabled = True

    # -- internals ----------------------------------------------------------------------------
    def _attempt(self, call, what: str) -> bool:
        """Run one wandb call with jittered exponential backoff; True on success, never raises."""
        delay = self._retry_base_s
        error = None
        for attempt in range(self._retries + 1):
            try:
                call()
                return True
            except Exception as exc:
                error = exc
                if attempt == self._retries:
                    break
                # Full jitter: sleep uniformly in [d, 2d] rather than d + up to 1s. Proportional
                # jitter still de-synchronizes concurrent runs, and it keeps the knobs honest --
                # retry_base_s=0 now really means "don't wait" (a fixed +1s made that impossible).
                capped = min(delay, self._retry_max_s)
                time.sleep(capped + self._rng.uniform(0.0, capped))
                delay *= 2
        self._log.warning(f"[wandb:{self._label}] {what} failed after {self._retries + 1} "
                          f"attempt(s): {error!r}")
        return False

    def _note_failure(self) -> None:
        """Count a failed push and give up on wandb entirely once they stop being transient."""
        self._failures += 1
        if self._failures >= self._max_failures:
            self._disabled = True
            self._rows.clear()
            self._log.warning(
                f"[wandb:{self._label}] {self._failures} consecutive failed pushes -- disabling "
                f"wandb for the rest of this run ({self._dropped} metric rows dropped). Training "
                f"continues; losses, per-eval benchmarks and both checkpoints are still written to "
                f"the run's output directory by QCBM.save().")


def logger_for(cfg, run, label: str = "") -> WandbLogger:
    """WandbLogger with this config's chunking/retry policy (see LoggingConfig in config_schema)."""
    return WandbLogger(
        run,
        flush_every=cfg.logging.wandb_flush_every,
        flush_interval_s=cfg.logging.wandb_flush_interval_s,
        max_failures=cfg.logging.wandb_max_failures,
        retries=cfg.logging.wandb_retry_max,
        retry_max_s=cfg.logging.wandb_retry_wait_max_s,
        label=label,
    )


# --------------------------------------------------------------------------------------------------
# sweep lifecycle
# --------------------------------------------------------------------------------------------------
def set_sweep_state(sweep_id: str, state: str, *, entity: str = None, project: str,
                    mode: str = "online") -> bool:
    """Move a Sweep to `state` ("RUNNING" / "FINISHED" / "PAUSED" / "CANCELED"). Never raises.

    Needed because nothing else advances the state in this setup: the grid is Hydra's and runs are
    attached to the sweep directly, so there is no agent/scheduler to do it, and the sweep would
    otherwise stay PENDING in the UI forever (before, during and after the runs). FINISHED means
    "start no new runs, let running ones finish", which is exactly the end-of-launch semantics here.

    Only meaningful online -- offline/disabled sweeps are a local synthetic id with no server object.
    """
    if mode != "online" or not sweep_id:
        return False
    try:
        from wandb.apis.internal import Api

        api = Api()
        if not entity:
            entity = api.settings("entity") or _default_entity()
        api.set_sweep_state(sweep=sweep_id, state=state, entity=entity, project=project)
        _log().info(f"[wandb] sweep {sweep_id} -> {state}")
        return True
    except Exception as exc:
        # Includes the benign "Sweep already finished" when a second launcher process (e.g. another
        # SLURM array task) gets here first.
        _log().warning(f"[wandb] could not set sweep {sweep_id} state to {state}: {exc!r}")
        return False


def _default_entity():
    """The account's default entity, or None -- resolved only when the config leaves it null."""
    try:
        import wandb

        return wandb.Api().default_entity
    except Exception:
        return None
