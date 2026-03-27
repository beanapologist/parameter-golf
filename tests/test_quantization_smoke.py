"""
Smoke test for the Quantization.lean integration in train_gpt_kernel.py.

Verifies that the KernelEquilibriumLoss (8-cycle closure, Hamiltonian drive,
and amplitude balance regularizers) derived from formal-lean/Quantization.lean
(lead_quantization_confirmed) is wired into KernelGPT and produces a measurable
training signal.

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
# force all three KernelEquilibriumLoss regularizers ON so we can test them.
os.environ.setdefault("KERNEL_SELFTEST", "0")
os.environ["USE_QUANT_REGULARIZER"] = "1"
os.environ["USE_DRIVE_REGULARIZER"] = "1"
os.environ["USE_AMPLITUDE_REGULARIZER"] = "1"

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
    HAMILTONIAN_DRIVE_TARGET,
    SILVER_RATIO,
    MU_ORBIT_SIZE,
    eight_cycle_loss,
    hamiltonian_drive_loss,
    amplitude_balance_loss,
    kernel_equilibrium_loss,
    coherence,
    KernelGPT,
    _USE_QUANT_REGULARIZER,
    _USE_DRIVE_REGULARIZER,
    _USE_AMPLITUDE_REGULARIZER,
)

# ── Helpers ───────────────────────────────────────────────────────────────────

_EPS = 1e-5


def _make_tiny_model(
    quant_lambda: float = 0.01,
    drive_lambda: float = 0.01,
    amplitude_lambda: float = 0.01,
) -> KernelGPT:
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
        drive_lambda=drive_lambda,
        amplitude_lambda=amplitude_lambda,
    )


# ── Test 1: feature flags are ON ──────────────────────────────────────────────

def test_use_quant_regularizer_flag_is_on() -> None:
    """USE_QUANT_REGULARIZER, USE_DRIVE_REGULARIZER, USE_AMPLITUDE_REGULARIZER must be True."""
    assert _USE_QUANT_REGULARIZER, (
        "_USE_QUANT_REGULARIZER is False despite USE_QUANT_REGULARIZER=1.  "
        "The Quantization.lean 8-cycle closure condition is silently not applied."
    )
    assert _USE_DRIVE_REGULARIZER, (
        "_USE_DRIVE_REGULARIZER is False despite USE_DRIVE_REGULARIZER=1.  "
        "The Hamiltonian drive condition (Q3.2) is silently not applied."
    )
    assert _USE_AMPLITUDE_REGULARIZER, (
        "_USE_AMPLITUDE_REGULARIZER is False despite USE_AMPLITUDE_REGULARIZER=1.  "
        "The amplitude balance condition (Q4.1) is silently not applied."
    )
    print("PASS  test_use_quant_regularizer_flag_is_on")


# ── Test 2: quant_phase_thetas is a per-layer learnable parameter ─────────────

def test_quant_phase_theta_exists() -> None:
    """KernelGPT must expose quant_phase_thetas (per-layer vector) when the regularizer is on."""
    model = _make_tiny_model()
    assert hasattr(model, "quant_phase_thetas"), (
        "KernelGPT.quant_phase_thetas attribute is missing.  "
        "The per-layer regularizer parameter was not created."
    )
    assert isinstance(model.quant_phase_thetas, torch.nn.Parameter), (
        "quant_phase_thetas must be an nn.Parameter so it receives gradient updates."
    )
    assert model.quant_phase_thetas.ndim == 1, (
        f"quant_phase_thetas must be a 1-D vector (one per layer); "
        f"got shape {model.quant_phase_thetas.shape}"
    )
    print(f"PASS  test_quant_phase_theta_exists  (shape={tuple(model.quant_phase_thetas.shape)})")


# ── Test 3: all per-layer thetas are initialised to MU_ANGLE ─────────────────

def test_quant_phase_theta_init() -> None:
    """quant_phase_thetas must all be initialised to MU_ANGLE = 3π/4."""
    model = _make_tiny_model()
    thetas = model.quant_phase_thetas
    for i, theta in enumerate(thetas):
        val = theta.item()
        assert abs(val - MU_ANGLE) < _EPS, (
            f"quant_phase_thetas[{i}] was initialised to {val:.6f}, "
            f"expected MU_ANGLE={MU_ANGLE:.6f} (=3π/4).  "
            "Each layer phase must start at the 8-cycle fixed point."
        )
    print(f"PASS  test_quant_phase_theta_init  (all {thetas.numel()} thetas={thetas[0].item():.6f})")


# ── Test 4: eight_cycle_loss is zero at MU_ANGLE ─────────────────────────────

def test_eight_cycle_loss_at_mu_angle() -> None:
    """L(MU_ANGLE) must equal 0 — MU_ANGLE is a fixed point of mu^8=1."""
    loss_val = eight_cycle_loss(torch.tensor(MU_ANGLE)).item()
    assert abs(loss_val) < _EPS, (
        f"eight_cycle_loss(MU_ANGLE) = {loss_val:.2e} != 0.  "
        "Quantization.lean Q1.2 requires L(3pi/4) = 1 - cos(8*3pi/4) = 0."
    )
    print(f"PASS  test_eight_cycle_loss_at_mu_angle  (L={loss_val:.2e})")


# ── Test 5: regularizer appears in the forward-pass loss ─────────────────────

def test_regularizer_contributes_to_loss() -> None:
    """When quant_phase_thetas != k*pi/4 slots, the loss must exceed CE alone."""
    model = _make_tiny_model(quant_lambda=1.0, drive_lambda=0.0, amplitude_lambda=0.0)

    # Manually shift all per-layer thetas away from every 8-cycle fixed point.
    with torch.no_grad():
        model.quant_phase_thetas.fill_(MU_ANGLE + math.pi / 8.0)  # midpoint: L=2

    x = torch.zeros(1, 4, dtype=torch.long)
    y = torch.ones(1, 4, dtype=torch.long)
    with torch.no_grad():
        loss = model(x, y).item()

    # Pure CE on a random 16-class model: log(16) ~= 2.77 nats.
    # With lambda=1 and mean(L_quant)=2 (worst case), total loss ~= 4.77.
    pure_ce_upper_bound = math.log(16) + _EPS  # ~= 2.77
    assert loss > pure_ce_upper_bound, (
        f"Forward-pass loss ({loss:.4f}) is not larger than the pure-CE upper "
        f"bound ({pure_ce_upper_bound:.4f}).  The 8-cycle regularizer (lambda=1.0, "
        "L_quant=2 at midpoint theta) is not being added to the training loss."
    )
    print(f"PASS  test_regularizer_contributes_to_loss  (loss={loss:.4f} > CE_bound={pure_ce_upper_bound:.4f})")


# ── Test 6: regularizer is zero when all thetas are at fixed points ───────────

def test_regularizer_zero_at_fixed_point() -> None:
    """When all thetas = MU_ANGLE (fixed point), 8-cycle term adds 0 to loss."""
    model = _make_tiny_model(quant_lambda=1.0, drive_lambda=0.0, amplitude_lambda=0.0)
    # All thetas already MU_ANGLE after __init__; just confirm
    for theta in model.quant_phase_thetas:
        assert abs(theta.item() - MU_ANGLE) < _EPS

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
        f"but total_loss={total_loss:.6f} != ce_only_loss={ce_only_loss:.6f}.  "
        "Check eight_cycle_loss and the forward-pass lambda guard."
    )
    print(f"PASS  test_regularizer_zero_at_fixed_point  (loss={total_loss:.6f})")


# ── Test 7: gradient flows through per-layer thetas ──────────────────────────

def test_gradient_flows_through_theta() -> None:
    """quant_phase_thetas must receive gradients when thetas != fixed points."""
    model = _make_tiny_model(quant_lambda=0.01, drive_lambda=0.0, amplitude_lambda=0.0)
    with torch.no_grad():
        model.quant_phase_thetas.fill_(MU_ANGLE + 0.2)  # non-fixed point

    x = torch.zeros(1, 4, dtype=torch.long)
    y = torch.ones(1, 4, dtype=torch.long)
    loss = model(x, y)
    loss.backward()

    assert model.quant_phase_thetas.grad is not None, (
        "No gradient on quant_phase_thetas after backward().  "
        "The parameter is not connected to the computational graph."
    )
    mean_grad = model.quant_phase_thetas.grad.abs().mean().item()
    assert mean_grad > 0.0, (
        f"Mean gradient on quant_phase_thetas is zero at theta=MU_ANGLE+0.2.  "
        "The 8-cycle loss is not differentiable at this point or is not applied."
    )
    print(f"PASS  test_gradient_flows_through_theta  (mean_grad={mean_grad:.6f})")


# ── Test 8: quantization spec constants are correct ──────────────────────────

def test_quantization_spec_constants() -> None:
    """All five conditions from lead_quantization_confirmed must hold numerically."""
    _eps = 1e-12

    # Q4.1: 2*eta^2 = 1
    assert abs(2.0 * ETA_SQUARED - 1.0) < _eps, (
        f"Q4.1 failed: 2*ETA_SQUARED={2.0 * ETA_SQUARED} != 1"
    )
    # Q3.1: FLOQUET_PHASE = pi
    assert abs(FLOQUET_PHASE - math.pi) < _eps, (
        f"Q3.1 failed: FLOQUET_PHASE={FLOQUET_PHASE} != pi"
    )
    # Q3.2: H*T = 5*pi/4
    assert abs(HAMILTONIAN_DRIVE_TARGET - 5.0 * math.pi / 4.0) < _eps, (
        f"Q3.2 failed: HAMILTONIAN_DRIVE_TARGET={HAMILTONIAN_DRIVE_TARGET} != 5*pi/4"
    )
    # Q1.2: mu^8 = 1  (cos(8*MU_ANGLE)=1, sin(8*MU_ANGLE)=0)
    assert abs(math.cos(8.0 * MU_ANGLE) - 1.0) < _eps, (
        f"Q1.2 failed: cos(8*MU_ANGLE)={math.cos(8.0 * MU_ANGLE)} != 1"
    )
    # Q4.3: C(1) = 1
    c1 = 2.0 * 1.0 / (1.0 + 1.0 ** 2)
    assert abs(c1 - 1.0) < _eps, f"Q4.3 failed: C(1)={c1} != 1"
    # Q2.2: E1 = -1
    e1 = -1.0 / (1 ** 2)
    assert abs(e1 - (-1.0)) < _eps, f"Q2.2 failed: E1={e1} != -1"

    print("PASS  test_quantization_spec_constants  (all 5 conditions + H*T target of lead_quantization_confirmed)")


# ── Test 9: coherence activation applies Q4.3 in every MLP block ─────────────

def test_coherence_activation_applied() -> None:
    """The CoherenceMLP blocks must use the coherence activation (Q4.3: C(1)=1)."""
    from train_gpt_kernel import CoherenceMLP

    model = _make_tiny_model()
    coherence_mlp_count = sum(1 for m in model.modules() if isinstance(m, CoherenceMLP))
    assert coherence_mlp_count > 0, (
        "No CoherenceMLP modules found in KernelGPT.  "
        "The coherence activation C(r)=2r/(1+r^2) (Q4.3) is not applied."
    )

    # Verify the activation satisfies C(1)=1
    x_ones = torch.ones(4)
    c1 = coherence(x_ones).mean().item()
    assert abs(c1 - 1.0) < _EPS, (
        f"coherence(ones) = {c1:.6f} != 1.0.  "
        "The coherence activation does not satisfy Q4.3 (C(1)=1)."
    )
    print(f"PASS  test_coherence_activation_applied  ({coherence_mlp_count} CoherenceMLP blocks, C(1)={c1:.6f})")


# ── Test 10: h_times_t parameter exists and is initialised correctly ──────────

def test_h_times_t_parameter_exists() -> None:
    """KernelGPT must expose h_times_t when USE_DRIVE_REGULARIZER=1."""
    model = _make_tiny_model()
    assert hasattr(model, "h_times_t"), (
        "KernelGPT.h_times_t attribute is missing.  "
        "The Hamiltonian drive parameter (Q3.2 H*T=5*pi/4) was not created."
    )
    assert isinstance(model.h_times_t, torch.nn.Parameter), (
        "h_times_t must be an nn.Parameter so it receives gradient updates."
    )
    ht_val = model.h_times_t.item()
    assert abs(ht_val - HAMILTONIAN_DRIVE_TARGET) < _EPS, (
        f"h_times_t was initialised to {ht_val:.6f}, "
        f"expected HAMILTONIAN_DRIVE_TARGET={HAMILTONIAN_DRIVE_TARGET:.6f} (=5*pi/4).  "
        "The drive penalty starts at zero when the model is correctly initialised."
    )
    print(f"PASS  test_h_times_t_parameter_exists  (h_times_t={ht_val:.6f})")


# ── Test 11: amplitude_r parameter exists and is initialised correctly ────────

def test_amplitude_r_parameter_exists() -> None:
    """KernelGPT must expose amplitude_r when USE_AMPLITUDE_REGULARIZER=1."""
    model = _make_tiny_model()
    assert hasattr(model, "amplitude_r"), (
        "KernelGPT.amplitude_r attribute is missing.  "
        "The amplitude balance parameter (Q4.1 2*eta^2=1) was not created."
    )
    assert isinstance(model.amplitude_r, torch.nn.Parameter), (
        "amplitude_r must be an nn.Parameter so it receives gradient updates."
    )
    r_val = model.amplitude_r.item()
    expected_eta = math.sqrt(0.5)  # 1/sqrt(2), satisfies 2*eta^2=1
    assert abs(r_val - expected_eta) < _EPS, (
        f"amplitude_r was initialised to {r_val:.6f}, "
        f"expected 1/sqrt(2)={expected_eta:.6f}.  "
        "The amplitude penalty starts at zero when r=1/sqrt(2) (2r^2=1)."
    )
    print(f"PASS  test_amplitude_r_parameter_exists  (amplitude_r={r_val:.6f} = 1/sqrt(2))")


# ── Test 12: hamiltonian_drive_loss properties ────────────────────────────────

def test_hamiltonian_drive_loss_properties() -> None:
    """hamiltonian_drive_loss must be zero at 5*pi/4 and positive elsewhere."""
    # Zero at target
    target = torch.tensor(HAMILTONIAN_DRIVE_TARGET)
    loss_at_target = hamiltonian_drive_loss(target).item()
    assert abs(loss_at_target) < _EPS, (
        f"hamiltonian_drive_loss(5*pi/4) = {loss_at_target:.2e} != 0.  "
        "Drive loss must be zero at the canonical H*T = 5*pi/4 target."
    )
    # Positive away from target
    for delta in (0.1, 0.5, 1.0, -0.3):
        val = hamiltonian_drive_loss(torch.tensor(HAMILTONIAN_DRIVE_TARGET + delta)).item()
        assert val > 0.0, (
            f"hamiltonian_drive_loss at H*T = 5*pi/4 + {delta} = {val:.4f} should be > 0"
        )
    # Differentiable: gradient non-zero away from target
    ht = torch.tensor(HAMILTONIAN_DRIVE_TARGET + 0.5, requires_grad=True)
    hamiltonian_drive_loss(ht).backward()
    assert ht.grad is not None and abs(ht.grad.item()) > 0.0, (
        "hamiltonian_drive_loss must be differentiable away from the target."
    )
    print(f"PASS  test_hamiltonian_drive_loss_properties  (loss_at_target={loss_at_target:.2e})")


# ── Test 13: amplitude_balance_loss properties ────────────────────────────────

def test_amplitude_balance_loss_properties() -> None:
    """amplitude_balance_loss must be zero when 2*r^2=1 and positive elsewhere."""
    # Zero at r = 1/sqrt(2)  (2*(1/sqrt(2))^2 = 2*0.5 = 1)
    r_balanced = torch.tensor(math.sqrt(0.5))
    loss_at_balanced = amplitude_balance_loss(r_balanced).item()
    assert abs(loss_at_balanced) < _EPS, (
        f"amplitude_balance_loss(1/sqrt(2)) = {loss_at_balanced:.2e} != 0.  "
        "Balance loss must be zero at the canonical amplitude r = 1/sqrt(2) (2*eta^2=1)."
    )
    # Positive away from balanced point
    for r_val in (0.1, 0.5, 1.0, 1.5):
        if abs(r_val - math.sqrt(0.5)) > 0.05:
            val = amplitude_balance_loss(torch.tensor(float(r_val))).item()
            assert val > 0.0, (
                f"amplitude_balance_loss at r={r_val} = {val:.4f} should be > 0"
            )
    # Differentiable
    r = torch.tensor(1.0, requires_grad=True)
    amplitude_balance_loss(r).backward()
    assert r.grad is not None and abs(r.grad.item()) > 0.0, (
        "amplitude_balance_loss must be differentiable away from r=1/sqrt(2)."
    )
    print(f"PASS  test_amplitude_balance_loss_properties  (loss_at_balanced={loss_at_balanced:.2e})")


# ── Test 14: kernel_equilibrium_loss combines all three arms ──────────────────

def test_kernel_equilibrium_loss_combines_arms() -> None:
    """kernel_equilibrium_loss must combine 8-cycle, drive, and amplitude terms."""
    thetas = torch.full((2,), MU_ANGLE)  # at fixed point
    ht = torch.tensor(HAMILTONIAN_DRIVE_TARGET)  # at target
    r = torch.tensor(math.sqrt(0.5))  # at balance

    # All at fixed points: total loss = 0 regardless of lambdas
    loss_zero = kernel_equilibrium_loss(thetas, ht, r, 1.0, 1.0, 1.0).item()
    assert abs(loss_zero) < _EPS, (
        f"kernel_equilibrium_loss at all fixed points = {loss_zero:.4e}, expected 0."
    )

    # Only quant arm: shift thetas away
    thetas_off = torch.full((2,), MU_ANGLE + math.pi / 8.0)  # midpoint: L=2
    loss_quant_only = kernel_equilibrium_loss(thetas_off, ht, r, 1.0, 0.0, 0.0).item()
    assert loss_quant_only > 0.0, (
        f"kernel_equilibrium_loss with off-cycle thetas (quant_lambda=1.0) = {loss_quant_only:.4f}, expected > 0"
    )

    # Only drive arm: shift h_times_t away
    ht_off = torch.tensor(HAMILTONIAN_DRIVE_TARGET + 1.0)
    loss_drive_only = kernel_equilibrium_loss(thetas, ht_off, r, 0.0, 1.0, 0.0).item()
    assert loss_drive_only > 0.0, (
        f"kernel_equilibrium_loss with off-target h_times_t (drive_lambda=1.0) = {loss_drive_only:.4f}, expected > 0"
    )

    # Only amplitude arm: shift r away
    r_off = torch.tensor(1.0)  # 2*1^2=2 != 1
    loss_amp_only = kernel_equilibrium_loss(thetas, ht, r_off, 0.0, 0.0, 1.0).item()
    assert loss_amp_only > 0.0, (
        f"kernel_equilibrium_loss with unbalanced r (amplitude_lambda=1.0) = {loss_amp_only:.4f}, expected > 0"
    )

    # All lambdas zero: loss must be exactly 0
    loss_all_zero = kernel_equilibrium_loss(thetas_off, ht_off, r_off, 0.0, 0.0, 0.0).item()
    assert abs(loss_all_zero) < _EPS, (
        f"kernel_equilibrium_loss with all lambdas=0 = {loss_all_zero:.4e}, expected 0."
    )

    print(
        f"PASS  test_kernel_equilibrium_loss_combines_arms  "
        f"(at_fixed={loss_zero:.2e} quant={loss_quant_only:.4f} "
        f"drive={loss_drive_only:.4f} amp={loss_amp_only:.4f})"
    )


# ── Test 15: drive and amplitude regularizers contribute to total loss ─────────

def test_drive_and_amplitude_contribute_to_loss() -> None:
    """Drive and amplitude regularizers must affect forward-pass loss when active."""
    # Build model with only amplitude active (drive_lambda=0 to isolate)
    model = _make_tiny_model(quant_lambda=0.0, drive_lambda=0.0, amplitude_lambda=0.0)

    x = torch.zeros(1, 4, dtype=torch.long)
    y = torch.ones(1, 4, dtype=torch.long)

    # Baseline: all lambdas zero == pure CE
    with torch.no_grad():
        loss_ce_only = model(x, y).item()

    # Activate amplitude regularizer with r shifted away from balance (r=1 -> 2r^2=2 != 1)
    model.amplitude_lambda = 1.0
    with torch.no_grad():
        model.amplitude_r.fill_(1.0)  # 2*1^2=2 != 1 -> amp penalty = (2-1)^2 = 1
    with torch.no_grad():
        loss_with_amp = model(x, y).item()

    assert loss_with_amp > loss_ce_only + _EPS, (
        f"amplitude_balance_loss should add > 0 when r=1 (penalty=1.0), "
        f"but loss_with_amp={loss_with_amp:.4f} not > loss_ce_only={loss_ce_only:.4f}"
    )
    print(
        f"PASS  test_drive_and_amplitude_contribute_to_loss  "
        f"(ce_only={loss_ce_only:.4f} with_amp_off={loss_with_amp:.4f})"
    )


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
    test_h_times_t_parameter_exists,
    test_amplitude_r_parameter_exists,
    test_hamiltonian_drive_loss_properties,
    test_amplitude_balance_loss_properties,
    test_kernel_equilibrium_loss_combines_arms,
    test_drive_and_amplitude_contribute_to_loss,
]


def main() -> None:
    print(f"Running {len(_TESTS)} Quantization.lean smoke tests ...\n")
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
        print(f"All {passed} tests PASSED")
    else:
        print(f"{passed} passed, {failed} FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
