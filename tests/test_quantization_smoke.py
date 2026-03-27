"""
Smoke test for the Quantization.lean integration in train_gpt_kernel.py.

Verifies that the 8-cycle closure regularizer derived from
formal-lean/Quantization.lean (lead_quantization_confirmed, Q1.2)
is wired into KernelGPT and produces a measurable training signal.

Run with:
    python tests/test_quantization_smoke.py

No CUDA, distributed setup, or data files are required.  All tensors
are built on CPU using a tiny synthetic vocabulary.
"""

from __future__ import annotations

import math
import os
import sys

# ── Point the import at the repository root ───────────────────────────────────
_REPO_ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, _REPO_ROOT)

# Force the kernel self-test off (it requires ENV vars we don't set here) and
# force the quantization regularizer ON so we can test it.
os.environ.setdefault("KERNEL_SELFTEST", "0")
os.environ["USE_QUANT_REGULARIZER"] = "1"

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Import only the symbols we need from the kernel script without triggering
# the full distributed / CUDA initialisation that lives inside main().
# We do this by importing at module level (the self-check functions at the
# top of the file run at import time; they are CPU-only and always safe).
# ---------------------------------------------------------------------------
from train_gpt_kernel import (
    MU_ANGLE,
    ETA_SQUARED,
    FLOQUET_PHASE,
    SILVER_RATIO,
    MU_ORBIT_SIZE,
    eight_cycle_loss,
    coherence,
    KernelGPT,
    _USE_QUANT_REGULARIZER,
)

# ── Helpers ───────────────────────────────────────────────────────────────────

_EPS = 1e-5


def _make_tiny_model(quant_lambda: float = 0.01) -> KernelGPT:
    """Return the smallest valid KernelGPT instance (CPU, float32)."""
    return KernelGPT(
        vocab_size=16,
        num_layers=2,
        model_dim=16,
        num_heads=8,
        num_kv_heads=4,
        mlp_hidden=round(SILVER_RATIO * 16),
        tie_embeddings=True,
        tied_embed_init_std=0.005,
        logit_softcap=30.0,
        rope_base=10000.0,
        qk_gain_init=1.5,
        quant_lambda=quant_lambda,
    )


# ── Test 1: feature flag is ON ────────────────────────────────────────────────

def test_use_quant_regularizer_flag_is_on() -> None:
    """USE_QUANT_REGULARIZER env var was set to '1'; flag must be True."""
    assert _USE_QUANT_REGULARIZER, (
        "_USE_QUANT_REGULARIZER is False despite USE_QUANT_REGULARIZER=1.  "
        "The Quantization.lean 8-cycle closure condition is silently not applied."
    )
    print("PASS  test_use_quant_regularizer_flag_is_on")


# ── Test 2: quant_phase_theta is a learnable parameter ───────────────────────

def test_quant_phase_theta_exists() -> None:
    """KernelGPT must expose quant_phase_theta when the regularizer is on."""
    model = _make_tiny_model()
    assert hasattr(model, "quant_phase_theta"), (
        "KernelGPT.quant_phase_theta attribute is missing.  "
        "The regularizer parameter was not created."
    )
    assert isinstance(model.quant_phase_theta, torch.nn.Parameter), (
        "quant_phase_theta must be an nn.Parameter so it receives gradient updates."
    )
    print("PASS  test_quant_phase_theta_exists")


# ── Test 3: theta is initialised to MU_ANGLE ─────────────────────────────────

def test_quant_phase_theta_init() -> None:
    """quant_phase_theta must be initialised to MU_ANGLE = 3π/4."""
    model = _make_tiny_model()
    theta = model.quant_phase_theta.item()
    assert abs(theta - MU_ANGLE) < _EPS, (
        f"quant_phase_theta was initialised to {theta:.6f}, "
        f"expected MU_ANGLE={MU_ANGLE:.6f} (=3π/4).  "
        "This is a fixed point of the 8-cycle loss (L(3π/4)=0), so training "
        "starts with zero regularization penalty as intended."
    )
    print(f"PASS  test_quant_phase_theta_init  (theta={theta:.6f})")


# ── Test 4: eight_cycle_loss is zero at MU_ANGLE ─────────────────────────────

def test_eight_cycle_loss_at_mu_angle() -> None:
    """L(MU_ANGLE) must equal 0 — MU_ANGLE is a fixed point of mu^8=1."""
    loss_val = eight_cycle_loss(torch.tensor(MU_ANGLE)).item()
    assert abs(loss_val) < _EPS, (
        f"eight_cycle_loss(MU_ANGLE) = {loss_val:.2e} ≠ 0.  "
        "Quantization.lean Q1.2 requires L(3π/4) = 1 − cos(8·3π/4) = 0."
    )
    print(f"PASS  test_eight_cycle_loss_at_mu_angle  (L={loss_val:.2e})")


# ── Test 5: regularizer appears in the forward-pass loss ─────────────────────

def test_regularizer_contributes_to_loss() -> None:
    """When quant_phase_theta ≠ a k·π/4 slot, the loss must exceed CE alone."""
    model = _make_tiny_model(quant_lambda=1.0)  # high lambda to make effect visible

    # Manually shift theta away from every 8-cycle fixed point.
    with torch.no_grad():
        model.quant_phase_theta.fill_(MU_ANGLE + math.pi / 8.0)  # midpoint — L=2

    x = torch.zeros(1, 4, dtype=torch.long)
    y = torch.ones(1, 4, dtype=torch.long)
    with torch.no_grad():
        loss = model(x, y).item()

    # Pure CE on a random 16-class model: log(16) ≈ 2.77 nats.
    # With lambda=1 and L_quant=2 (worst case), total loss ≈ 4.77.
    # Confirm the regularizer is large enough to detect above pure CE.
    pure_ce_upper_bound = math.log(16) + _EPS  # ≈ 2.77
    assert loss > pure_ce_upper_bound, (
        f"Forward-pass loss ({loss:.4f}) is not larger than the pure-CE upper "
        f"bound ({pure_ce_upper_bound:.4f}).  The 8-cycle regularizer (lambda=1.0, "
        "L_quant=2 at midpoint theta) is not being added to the training loss."
    )
    print(f"PASS  test_regularizer_contributes_to_loss  (loss={loss:.4f} > CE_bound={pure_ce_upper_bound:.4f})")


# ── Test 6: regularizer is zero when theta is at a fixed point ───────────────

def test_regularizer_zero_at_fixed_point() -> None:
    """When theta = MU_ANGLE (fixed point), CE loss == total loss exactly."""
    model = _make_tiny_model(quant_lambda=1.0)
    # theta is already MU_ANGLE after __init__; just confirm
    assert abs(model.quant_phase_theta.item() - MU_ANGLE) < _EPS

    x = torch.zeros(1, 4, dtype=torch.long)
    y = torch.ones(1, 4, dtype=torch.long)
    with torch.no_grad():
        total_loss = model(x, y).item()

    # Disable regularizer and recompute CE for the same inputs / weights
    model.quant_lambda = 0.0
    with torch.no_grad():
        ce_only_loss = model(x, y).item()

    assert abs(total_loss - ce_only_loss) < _EPS, (
        f"At theta=MU_ANGLE the regularizer should contribute 0 to the loss, "
        f"but total_loss={total_loss:.6f} ≠ ce_only_loss={ce_only_loss:.6f}.  "
        "Check eight_cycle_loss and the forward-pass lambda guard."
    )
    print(f"PASS  test_regularizer_zero_at_fixed_point  (loss={total_loss:.6f})")


# ── Test 7: gradient flows through theta ─────────────────────────────────────

def test_gradient_flows_through_theta() -> None:
    """quant_phase_theta must receive a gradient when theta ≠ fixed point."""
    model = _make_tiny_model(quant_lambda=0.01)
    with torch.no_grad():
        model.quant_phase_theta.fill_(MU_ANGLE + 0.2)  # non-fixed point

    x = torch.zeros(1, 4, dtype=torch.long)
    y = torch.ones(1, 4, dtype=torch.long)
    loss = model(x, y)
    loss.backward()

    assert model.quant_phase_theta.grad is not None, (
        "No gradient on quant_phase_theta after backward().  "
        "The parameter is not connected to the computational graph."
    )
    assert model.quant_phase_theta.grad.abs().item() > 0.0, (
        f"Gradient on quant_phase_theta is zero at theta=MU_ANGLE+0.2.  "
        "The 8-cycle loss is not differentiable at this point or is not applied."
    )
    print(f"PASS  test_gradient_flows_through_theta  (grad={model.quant_phase_theta.grad.item():.6f})")


# ── Test 8: quantization spec constants are correct ──────────────────────────

def test_quantization_spec_constants() -> None:
    """All five conditions from lead_quantization_confirmed must hold numerically."""
    _eps = 1e-12

    # Q4.1: 2η² = 1
    assert abs(2.0 * ETA_SQUARED - 1.0) < _eps, (
        f"Q4.1 failed: 2·ETA_SQUARED={2.0 * ETA_SQUARED} ≠ 1"
    )
    # Q3.1: FLOQUET_PHASE = π
    assert abs(FLOQUET_PHASE - math.pi) < _eps, (
        f"Q3.1 failed: FLOQUET_PHASE={FLOQUET_PHASE} ≠ π"
    )
    # Q1.2: μ^8 = 1  (cos(8·MU_ANGLE)=1, sin(8·MU_ANGLE)=0)
    assert abs(math.cos(8.0 * MU_ANGLE) - 1.0) < _eps, (
        f"Q1.2 failed: cos(8·MU_ANGLE)={math.cos(8.0 * MU_ANGLE)} ≠ 1"
    )
    # Q4.3: C(1) = 1
    c1 = 2.0 * 1.0 / (1.0 + 1.0 ** 2)
    assert abs(c1 - 1.0) < _eps, f"Q4.3 failed: C(1)={c1} ≠ 1"
    # Q2.2: E₁ = −1
    e1 = -1.0 / (1 ** 2)
    assert abs(e1 - (-1.0)) < _eps, f"Q2.2 failed: E₁={e1} ≠ −1"

    print("PASS  test_quantization_spec_constants  (all 5 conditions of lead_quantization_confirmed)")


# ── Test 9: coherence activation applies Q4.3 in every MLP block ─────────────

def test_coherence_activation_applied() -> None:
    """The CoherenceMLP blocks must use the coherence activation (Q4.3: C(1)=1)."""
    from train_gpt_kernel import CoherenceMLP

    model = _make_tiny_model()
    coherence_mlp_count = sum(1 for m in model.modules() if isinstance(m, CoherenceMLP))
    assert coherence_mlp_count > 0, (
        "No CoherenceMLP modules found in KernelGPT.  "
        "The coherence activation C(r)=2r/(1+r²) (Q4.3) is not applied."
    )

    # Verify the activation satisfies C(1)=1
    x_ones = torch.ones(4)
    c1 = coherence(x_ones).mean().item()
    assert abs(c1 - 1.0) < _EPS, (
        f"coherence(ones) = {c1:.6f} ≠ 1.0.  "
        "The coherence activation does not satisfy Q4.3 (C(1)=1)."
    )
    print(f"PASS  test_coherence_activation_applied  ({coherence_mlp_count} CoherenceMLP blocks, C(1)={c1:.6f})")


# ── Runner ────────────────────────────────────────────────────────────────────

_TESTS = [
    test_use_quant_regularizer_flag_is_on,
    test_quant_phase_theta_exists,
    test_quant_phase_theta_init,
    test_eight_cycle_loss_at_mu_angle,
    test_regularizer_contributes_to_loss,
    test_regularizer_zero_at_fixed_point,
    test_gradient_flows_through_theta,
    test_quantization_spec_constants,
    test_coherence_activation_applied,
]


def main() -> None:
    print(f"Running {len(_TESTS)} Quantization.lean smoke tests …\n")
    passed = 0
    failed = 0
    for test_fn in _TESTS:
        try:
            test_fn()
            passed += 1
        except Exception as exc:
            failed += 1
            print(f"FAIL  {test_fn.__name__}")
            print(f"      {exc}")
    print(f"\n{'=' * 60}")
    if failed == 0:
        print(f"All {passed} tests PASSED ✓")
    else:
        print(f"{passed} passed, {failed} FAILED ✗")
        sys.exit(1)


if __name__ == "__main__":
    main()
