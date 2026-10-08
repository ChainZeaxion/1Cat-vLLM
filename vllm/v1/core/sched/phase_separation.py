# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Time-credit phase allocator: keep decode and prefill in separate phases.

Why fence *time* rather than *tokens*
-------------------------------------
A mixed batch (decoders + a prefill chunk) can never replay a FULL CUDA graph:
``get_uniform_decode_token_count`` returns ``None`` as soon as ``has_prefill``
is true, so the handful of decode tokens in that step falls back to eager
execution. Measured on this deployment (TP4, 1024-token chunks):

    pure prefill, 1 req, 1024 tok : cpu 232 ms / gpu 268 ms
    mixed,        3 req, 1040 tok : cpu 343 ms / gpu 350 ms
                                    (+108 ms host, +82 ms GPU for 16 decode tokens)

Fencing the *token* budget cannot fix this, because a prefill step costs almost
the same whether it carries 20 or 1024 tokens -- there is a fixed ~230 ms host
dispatch floor. Earlier attempts (the "fence v1-v4" series) all fenced tokens
and all failed for exactly this reason. Fencing *time* -- running whole steps as
one phase or the other -- lets the decode phase replay its FULL graph (measured
23 ms cpu for 16 tokens, vs 343 ms when a prefill shares the step).

This module owns only the bookkeeping. It never touches the model, the GPU, or
the scheduler's data structures, so it is unit-testable in isolation.

Guardrails inherited from the #37a controller
---------------------------------------------
#37a (``MixedPrefillBudget``) collapsed to its 16-token floor and could not
recover. Three specific mistakes are designed out here:

1. **Only the phase's own elapsed time is fed back.** #37a divided a whole step
   that contained concurrent decode work by only the prefill tokens, inflating
   the apparent per-token cost by roughly 10x.
2. **The decision compares against an absolute zero**, never against a quantity
   derived from the controller's own output. #37a's gate
   ``prefill_tokens < budget * 0.75`` compared against its *current* budget, so
   once it reached the floor the gate always passed and the collapse was
   self-sustaining.
3. **Growth and shrink are symmetric, and the floor is an absolute constant.**
   #37a capped growth at ``tokens * 1.25`` but left shrink unbounded, making the
   collapse a one-way ratchet.

The chunk size is deliberately *not* configured here. It follows from the Mamba
state-block retention setting (``--prefix-cache-retention-interval``), so this
controller cannot change the batch geometry underneath itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Phase(Enum):
    DECODE = "decode"
    PREFILL = "prefill"


@dataclass(frozen=True)
class PhaseConfig:
    """Knobs that are safe to tune; everything else is measured at runtime."""

    # Target share of wall-clock time given to the decode phase (0..1).
    #
    # NOTE this is a *share*, not the earn rate. The credit balance is
    # stationary when ``earn_ratio * prefill_ms == decode_ms``, and
    # ``decode_ms / (decode_ms + prefill_ms) == share`` implies
    # ``earn_ratio == share / (1 - share)``. Storing the share (and converting
    # below) keeps this knob in the same units the operator reasons about, and
    # makes the 1/3-share case a 0.5 earn rate rather than a 1/3 one -- using
    # 1/3 directly would settle at a 25% share, not 33%.
    decode_time_share: float = 1.0 / 3.0
    # Hard ceiling on how long a decoder may go unserved, in milliseconds.
    #
    # NOTE this bounds the prefill *streak*, but the wait a decoder actually
    # observes is `max_decode_wait_ms + one prefill step`: the guard can only
    # be evaluated at a step boundary, so a prefill step already in flight has
    # to finish first. Measured on this deployment: 700 + 386 = 1086 ms
    # predicted, 1071 ms observed. Size this against the prefill step, not
    # against the stall target on its own.
    max_decode_wait_ms: float = 700.0
    # Absolute token floor/ceiling for a prefill chunk. Constants on purpose --
    # deriving them from the controller's own output is the #37a failure mode.
    min_prefill_tokens: int = 1024
    max_prefill_tokens: int = 8192
    # Drop a feedback interval longer than this: it means the engine sat idle
    # between steps, which says nothing about phase cost.
    max_feedback_ms: float = 3000.0
    # Cap on banked credit, expressed as a multiple of a measured decode step.
    credit_cap_steps: float = 8.0
    # Absolute floor for that cap. Deliberately *independent* of
    # max_decode_wait_ms: when the cap was derived from it, lowering the wait
    # budget to tighten the stall bound also throttled the earn rate and the
    # realised share fell below the configured one. The cap and the stall
    # bound are different concerns and must not share a knob.
    credit_cap_floor_ms: float = 1500.0
    # Below this many prefill tokens a step is dominated by the fixed ~250 ms
    # host dispatch floor, so dividing its wall-clock by its token count gives
    # a meaningless per-token cost (observed: 161 ms/token on a 3-token replay).
    # Such samples are skipped rather than allowed to poison the average.
    min_prefill_tokens_for_ema: int = 256
    ema_alpha: float = 0.3

    @property
    def earn_ratio(self) -> float:
        """Decode-milliseconds earned per prefill-millisecond."""
        share = min(max(self.decode_time_share, 0.0), 0.95)
        return share / (1.0 - share)


class PhaseCreditAllocator:
    """Decide which phase the next step runs, and bookkeep what it cost."""

    def __init__(self, config: PhaseConfig | None = None) -> None:
        self.config = config or PhaseConfig()
        self.credit_ms = 0.0
        # Wall-clock bookkeeping for the feedback loop (seconds, monotonic).
        self._prev_phase: Phase | None = None
        self._prev_ts: float | None = None
        self._prev_did_work = False
        self._prev_prefill_tokens = 0
        # When the decode phase last actually ran, for the starvation guard.
        self._last_decode_ts: float | None = None
        # Set when a chosen phase scheduled nothing; see note_empty_phase_step.
        self._force_flip: Phase | None = None
        # Realised step counts, for tuning. The configured knob is a *time*
        # share, but operators reason in "N decode steps per prefill step", so
        # these are what make the two views reconcilable. Counted here rather
        # than via the step profiler: the profiler samples one step in forty
        # (too coarse to see the pattern) and forces a device sync when asked
        # for every step, which would perturb the very cadence being measured.
        self.decode_steps = 0
        self.prefill_steps = 0
        # Measured, never fitted. Diagnostics and the chunk cap read these.
        self.decode_step_ms: float | None = None
        self.prefill_ms_per_token: float | None = None

    # ---------------- decision ----------------

    def next_phase(
        self, *, has_decode: bool, has_prefill: bool, now: float
    ) -> Phase | None:
        """Pick the phase for the step about to be scheduled.

        ``None`` means there is nothing to do. Rule order matters:

        * Emptiness is checked before credit, so a deeply negative credit with
          no prefill work available cannot spin an empty step.
        * Starvation is checked before credit, so a decoder cannot be held off
          indefinitely by a long prefill streak.
        """
        if not has_decode and not has_prefill:
            return None
        # Nothing to decode: prefill must run, even with credit exhausted.
        if not has_decode:
            return Phase.PREFILL
        # Nothing to prefill: decode must run, even with credit in debt.
        if not has_prefill:
            return Phase.DECODE
        # The phase we chose last step scheduled nothing (e.g. every decoder is
        # momentarily ineligible). Flip rather than retry, or the engine spins
        # empty steps while real work waits -- and with no work scheduled the
        # credit never moves, so the spin would be permanent.
        if self._force_flip is not None:
            forced = (
                Phase.PREFILL if self._force_flip is Phase.DECODE else Phase.DECODE
            )
            self._force_flip = None
            return forced
        # A decoder has waited at least max_decode_wait_ms: force decode.
        if (
            self._last_decode_ts is not None
            and (now - self._last_decode_ts) * 1000.0 >= self.config.max_decode_wait_ms
        ):
            return Phase.DECODE
        # Otherwise credit decides. Zero is the tie-break in favour of decode,
        # so an idle-credit system alternates rather than starving either side.
        return Phase.DECODE if self.credit_ms >= 0.0 else Phase.PREFILL

    def note_empty_phase_step(self, phase: Phase) -> None:
        """Record that ``phase`` was chosen but scheduled no tokens.

        The scheduler calls this only for a step that really ran (not an idle
        engine), so it is safe to treat as "this phase has nothing to do now".
        """
        self._force_flip = phase

    # ---------------- feedback ----------------

    def charge_elapsed(self, now: float) -> None:
        """Charge the previous step's wall-clock to the phase that ran it.

        Must be called *before* :meth:`next_phase`. Charging afterwards leaves
        the credit one decision stale, and a prefill step's earnings then only
        become visible two decisions later -- which makes a credit balance that
        has just crossed zero need two consecutive prefill steps to recover,
        burning a whole prefill step per cycle (measured: a 23% decode share
        instead of the requested 33%).
        """
        if self._prev_phase is not None and self._prev_ts is not None:
            if self._prev_did_work:
                dt_ms = (now - self._prev_ts) * 1000.0
                # Idle gaps say nothing about phase cost -- drop them.
                if 0.0 < dt_ms <= self.config.max_feedback_ms:
                    self._apply_feedback(self._prev_phase, dt_ms)
        self._prev_ts = now
        self._prev_did_work = False
        self._prev_prefill_tokens = 0

    def begin_phase(self, phase: Phase, now: float) -> None:
        """Record the phase the step about to run belongs to."""
        self._prev_phase = phase
        if phase is Phase.DECODE or self._last_decode_ts is None:
            # Stamp the first step whatever its phase. Without this the
            # starvation guard stays inert until decode happens to run once,
            # so a burst of prefill steps right after startup could hold
            # decoders off for far longer than max_decode_wait_ms.
            self._last_decode_ts = now

    def observe_step_work(self, *, scheduled_tokens: int, prefill_tokens: int) -> None:
        """Record what the step actually did, once the batch is known."""
        self._prev_did_work = scheduled_tokens > 0
        self._prev_prefill_tokens = prefill_tokens
        if scheduled_tokens > 0:
            if prefill_tokens > 0:
                self.prefill_steps += 1
            else:
                self.decode_steps += 1

    def _apply_feedback(self, phase: Phase, dt_ms: float) -> None:
        cfg = self.config
        if phase is Phase.DECODE:
            self.credit_ms -= dt_ms
            self.decode_step_ms = self._ema(self.decode_step_ms, dt_ms)
        else:
            self.credit_ms += cfg.earn_ratio * dt_ms
            # Only prefill tokens divide prefill time. This is guardrail #1:
            # the step was phase-pure, so its whole duration belongs to them.
            if self._prev_prefill_tokens >= cfg.min_prefill_tokens_for_ema:
                self.prefill_ms_per_token = self._ema(
                    self.prefill_ms_per_token, dt_ms / self._prev_prefill_tokens
                )
        # Symmetric clamp (guardrail #3). The upper bound is how much decode
        # time may be banked while only prefill work exists, which is also the
        # longest decode burst that can follow -- so it must never fall below
        # one wait budget, or the cap silently throttles the configured share.
        # (With a bare 8 * decode_step_ms the cap measured 320 ms, just under
        # the ~326 ms a 653 ms prefill step earns at a 1/3 share, and the
        # realised share collapsed to 27%.) The lower bound is how long a
        # prefill streak may run before decode is forced back in.
        measured = (
            cfg.credit_cap_steps * self.decode_step_ms
            if self.decode_step_ms is not None
            else 0.0
        )
        upper = max(measured, cfg.credit_cap_floor_ms)
        lower = -2.0 * cfg.max_decode_wait_ms
        self.credit_ms = min(upper, max(lower, self.credit_ms))

    def _ema(self, current: float | None, sample: float) -> float:
        if current is None:
            return sample
        a = self.config.ema_alpha
        return (1.0 - a) * current + a * sample

    # ---------------- derived ----------------

    def prefill_chunk_cap(self) -> int:
        """Largest prefill chunk that still fits inside one phase.

        Derived from *measured* ms/token, so it shrinks automatically as the
        context grows (per-token cost rises with KV length). A fixed token cap
        would overrun the phase budget at long context; this cannot.
        """
        cfg = self.config
        if not self.prefill_ms_per_token:
            return cfg.max_prefill_tokens
        fits = cfg.max_decode_wait_ms / self.prefill_ms_per_token
        return max(cfg.min_prefill_tokens, min(cfg.max_prefill_tokens, int(fits)))

    def steps_per_prefill(self) -> float:
        """Realised decode steps per prefill step (0 until both have run)."""
        if not self.prefill_steps:
            return 0.0
        return self.decode_steps / self.prefill_steps

    def stats(self) -> str:
        def fmt(v: float | None, nd: int = 2) -> str:
            return "n/a" if v is None else f"{round(v, nd)}"

        return (
            f"D:P={self.decode_steps}:{self.prefill_steps} "
            f"({self.steps_per_prefill():.2f} dec/pre) "
            f"credit={self.credit_ms:.1f}ms "
            f"dec_step={fmt(self.decode_step_ms)}ms "
            f"pre_ms_per_tok={fmt(self.prefill_ms_per_token, 4)} "
            f"chunk_cap={self.prefill_chunk_cap()}"
        )
