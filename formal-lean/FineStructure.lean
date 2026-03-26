/-
  FineStructure.lean — Lean 4 formalization of Bohr–Rydberg energy quantization.

  Defines the Rydberg energy levels E_n = −1/n² and establishes three
  fundamental quantization properties:
    • ground state:   E₁ = −1
    • negativity:     E_n < 0  for all n ≥ 1
    • strict ascent:  E_n < E_{n+1}  (levels increase toward zero)

  These are synthesized in Quantization.lean as part of Theorem Q.

  Proof status
  ────────────
  All definitions and theorems have complete machine-checked proofs.
  No `sorry` placeholders remain.
-/

import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
import Mathlib.Data.Real.Basic

open Real

noncomputable section

-- ════════════════════════════════════════════════════════════════════════════
-- Rydberg Energy Definition
-- ════════════════════════════════════════════════════════════════════════════

/-- The Rydberg energy levels E_n = −1/n² for n ≥ 1.

    This is the Bohr–Rydberg formula in natural units (ℏ = m_e = e = 1,
    Rydberg energy = 1):
       E_n = −1 / n²,    n = 1, 2, 3, …

    The ground state is E₁ = −1 and levels ascend monotonically to 0.
    The hypothesis hn : n ≠ 0 ensures the denominator is non-zero. -/
noncomputable def rydbergEnergy (n : ℕ) (_ : n ≠ 0) : ℝ := -(1 : ℝ) / (n : ℝ) ^ 2

-- ════════════════════════════════════════════════════════════════════════════
-- Basic Rydberg Properties
-- ════════════════════════════════════════════════════════════════════════════

/-- The Rydberg formula: E_n = −1/n². -/
theorem rydberg_formula (n : ℕ) (hn : n ≠ 0) :
    rydbergEnergy n hn = -(1 : ℝ) / (n : ℝ) ^ 2 := rfl

/-- The ground-state energy: E₁ = −1. -/
theorem rydberg_ground_state : rydbergEnergy 1 one_ne_zero = -1 := by
  unfold rydbergEnergy
  norm_num

/-- All Rydberg energy levels are negative: E_n < 0 for n ≥ 1. -/
theorem rydberg_negative (n : ℕ) (hn : n ≠ 0) : rydbergEnergy n hn < 0 := by
  unfold rydbergEnergy
  have hn' : (0 : ℝ) < (n : ℝ) := Nat.cast_pos.mpr (Nat.pos_of_ne_zero hn)
  have hnsq : (0 : ℝ) < (n : ℝ) ^ 2 := by positivity
  simp only [neg_div]
  linarith [div_pos one_pos hnsq]

/-- Rydberg levels strictly ascend: E_n < E_{n+1} for all n ≥ 1.

    Proof: −1/n² < −1/(n+1)² because (n+1)² > n² > 0. -/
theorem rydberg_strict_ascent (n : ℕ) (hn : n ≠ 0) :
    rydbergEnergy n hn < rydbergEnergy (n + 1) (Nat.succ_ne_zero n) := by
  unfold rydbergEnergy
  have hn' : (0 : ℝ) < (n : ℝ) := Nat.cast_pos.mpr (Nat.pos_of_ne_zero hn)
  -- Normalise the cast ↑(n + 1) = ↑n + 1 in ℝ
  have hcast : (↑(n + 1) : ℝ) = (n : ℝ) + 1 := by push_cast; ring
  rw [hcast]
  have hnsq : (0 : ℝ) < (n : ℝ) ^ 2 := by positivity
  have hn1sq : (0 : ℝ) < ((n : ℝ) + 1) ^ 2 := by positivity
  -- Rewrite using neg_div: (-1)/x = -(1/x)
  simp only [neg_div]
  -- Goal: -(1/n²) < -(1/(n+1)²)
  -- i.e., 1/(n+1)² < 1/n² (by neg_lt_neg)
  apply neg_lt_neg
  rw [div_lt_div_iff hn1sq hnsq]
  -- Goal: 1 * n² < 1 * (n+1)²
  simp only [one_mul]
  nlinarith

end -- noncomputable section
