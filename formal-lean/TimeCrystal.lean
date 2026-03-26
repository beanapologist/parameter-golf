/-
  TimeCrystal.lean — Lean 4 formalization of Floquet/time-crystal quantization.

  Defines the Floquet quasi-energy ε_F = π/T and the one-period time-evolution
  operator U(H,T) = exp(−I·H·T), then proves the fundamental quantization
  condition ε_F · T = π and related Floquet properties.

  These definitions are synthesized in Quantization.lean, where the connection
  to the critical eigenvalue μ is established.

  Proof status
  ────────────
  All definitions and lemmas have complete machine-checked proofs.
  No `sorry` placeholders remain.
-/

import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
import Mathlib.Data.Complex.Exponential

open Complex Real

noncomputable section

-- ════════════════════════════════════════════════════════════════════════════
-- Floquet Quasi-Energy
-- ════════════════════════════════════════════════════════════════════════════

/-- The Floquet quasi-energy ε_F = π/T.

    For a time-crystal with period T, the quasi-energy is defined as π/T,
    ensuring the phase accumulated per period equals exactly π:
       ε_F · T = (π/T) · T = π.

    The hypothesis T ≠ 0 is carried to allow downstream rewriting. -/
noncomputable def timeCrystalQuasiEnergy (T : ℝ) (_ : T ≠ 0) : ℝ := Real.pi / T

/-- The Floquet time-evolution operator U(H,T) = exp(−I·H·T).

    For a Hamiltonian H acting over period T, the one-period unitary is
       U(H,T) = exp(−i·H·T).
    When H·T = 5π/4 this equals the critical eigenvalue μ (proved in
    Quantization.lean). -/
noncomputable def floquetOperator (H T : ℝ) : ℂ :=
  Complex.exp (-(Complex.I * ↑H * ↑T))

-- ════════════════════════════════════════════════════════════════════════════
-- Basic Floquet Properties
-- ════════════════════════════════════════════════════════════════════════════

/-- The Floquet phase condition: ε_F · T = π for any T ≠ 0.

    Direct from the definition ε_F = π/T. -/
theorem timeCrystal_phase (T : ℝ) (hT : T ≠ 0) :
    timeCrystalQuasiEnergy T hT * T = Real.pi := by
  unfold timeCrystalQuasiEnergy
  exact div_mul_cancel₀ Real.pi hT

/-- The Floquet quasi-energy is positive when T > 0. -/
theorem timeCrystal_pos (T : ℝ) (hT : 0 < T) :
    0 < timeCrystalQuasiEnergy T (ne_of_gt hT) := by
  unfold timeCrystalQuasiEnergy
  exact div_pos Real.pi_pos hT

/-- Period doubling: ε_F · (2T) = 2π for any T ≠ 0.

    Two periods accumulate a full 2π phase — the characteristic signature
    of period-doubling in a time crystal. -/
theorem timeCrystal_period_double (T : ℝ) (hT : T ≠ 0) :
    timeCrystalQuasiEnergy T hT * (2 * T) = 2 * Real.pi := by
  unfold timeCrystalQuasiEnergy
  field_simp [hT]
  ring

end -- noncomputable section
