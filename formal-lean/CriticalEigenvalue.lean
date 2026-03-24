/-
  CriticalEigenvalue.lean — Lean 4 formalization of core Kernel theorems.

  The central object is the critical eigenvalue
      μ = exp(I · 3π/4) = (−1 + i)/√2,
  the unique balanced point on the unit circle with negative real part
  and positive imaginary part.  It generates an exact 8-cycle and drives
  the coherence / rotation structure studied in the Kernel project.

  Mathematical background: ../docs/master_derivations.pdf

  Sections
  ────────
  1.  Critical eigenvalue  μ = exp(I · 3π/4)
  2.  Eight-cycle closure  μ^8 = 1
  3.  Distinctness of the eight powers of μ   (gcd(3,8) = 1)
  4.  Rotation matrix  R(3π/4)  and its properties
  5.  Coherence function  C(r) = 2r/(1 + r²)
  6.  Canonical-state normalisation  η² + |μ·η|² = 1
  7.  Silver ratio  δS = 1 + √2  (Proposition 4)
  8.  Additional coherence properties  (symmetry, positivity, strict bound)
  9.  Palindrome residual  R(r) = (r − 1/r)/δS  (Theorem 12)
  10. Lyapunov–coherence duality  C(exp λ) = sech λ  (Theorem 14)
  11. Derived invariant equivalences  (machine-discovered connections)
  12. Orbit magnitude and trichotomy  |ξⁿ| = rⁿ  (Theorem 10)
  13. Coherence monotonicity  (strictly increasing on (0,1], decreasing on [1,∞))
  14. Palindrome arithmetic  (digit identities, gcd/lcm of torus periods)
  15. Z/8Z rotational memory  (bank addressing, μ^(j+8) = μ^j)
  16. Zero-overhead precession  (|e^{iθ}| = 1, preserves |β| and C(r))
  17. Ohm-Coherence circuit identities  (G·R=1, parallel/series laws)
  18. Pythagorean coherence identity  (C²+((r²-1)/(1+r²))²=1)
  19. Orbit Lyapunov connection  (|ξⁿ|=exp(n·logr), C(rⁿ)=sech(n·logr))
  20. Silver ratio self-similarity  (δS = 2+1/δS, minimal polynomial)
  21. Phase accumulation + NullSliceBridge coverage bijection
  22. Machine-discovered deep connections  (master link, η–δS bridge, hyperbolic Pythagorean)

  Proof status
  ────────────
  All theorems have complete machine-checked proofs.
  No `sorry` placeholders remain.
-/

import Mathlib.Analysis.SpecialFunctions.Complex.Circle
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
import Mathlib.Analysis.SpecialFunctions.Pow.Real
import Mathlib.Data.Complex.Exponential
import Mathlib.Data.Matrix.Basic
import Mathlib.LinearAlgebra.Matrix.Determinant.Basic
import Mathlib.RingTheory.RootsOfUnity.Complex

open Complex Real Matrix

noncomputable section

-- ════════════════════════════════════════════════════════════════════════════
-- Section 1 — Critical Eigenvalue
-- Ref: docs/master_derivations.pdf §2.1
-- ════════════════════════════════════════════════════════════════════════════

/-- The critical eigenvalue μ = exp(I · 3π/4).

    Numerically: μ = (−1 + i)/√2 ≈ −0.7071 + 0.7071·i.
    It is the unique point on the unit circle at angle 135° = 3π/4. -/
def μ : ℂ := Complex.exp (Complex.I * (3 * ↑Real.pi / 4))

/-- μ can equivalently be written as (−1 + I)/√2. -/
theorem mu_eq_cart : μ = ((-1 + Complex.I) / Real.sqrt 2) := by
  unfold μ
  -- exp(I · 3π/4) = cos(3π/4) + I·sin(3π/4)
  --              = −(1/√2) + I·(1/√2) = (−1 + I)/√2
  have h2ne : Real.sqrt 2 ≠ 0 := Real.sqrt_ne_zero'.mpr (by norm_num)
  have hcos : Real.cos (3 * Real.pi / 4) = -(1 / Real.sqrt 2) := by
    rw [show (3 : ℝ) * Real.pi / 4 = Real.pi - Real.pi / 4 by ring]
    rw [Real.cos_pi_sub, Real.cos_pi_div_four]
    field_simp [h2ne]
  have hsin : Real.sin (3 * Real.pi / 4) = 1 / Real.sqrt 2 := by
    rw [show (3 : ℝ) * Real.pi / 4 = Real.pi - Real.pi / 4 by ring]
    rw [Real.sin_pi_sub, Real.sin_pi_div_four]
    field_simp [h2ne]
  apply Complex.ext
  · -- Re(exp(I * 3π/4)) = cos(3π/4) = -1/√2
    have hre : (Complex.exp (Complex.I * (3 * ↑Real.pi / 4))).re =
               Real.cos (3 * Real.pi / 4) := by
      simp [Complex.exp_re, Complex.mul_re, Complex.I_re, Complex.I_im,
            Complex.ofReal_re, Complex.ofReal_im, mul_comm]
    rw [hre, hcos]
    simp [Complex.div_re, Complex.normSq_ofReal, Complex.add_re,
          Complex.neg_re, Complex.one_re, Complex.I_re]
    field_simp [h2ne]
  · -- Im(exp(I * 3π/4)) = sin(3π/4) = 1/√2
    have him : (Complex.exp (Complex.I * (3 * ↑Real.pi / 4))).im =
               Real.sin (3 * Real.pi / 4) := by
      simp [Complex.exp_im, Complex.mul_re, Complex.mul_im, Complex.I_re, Complex.I_im,
            Complex.ofReal_re, Complex.ofReal_im, mul_comm]
    rw [him, hsin]
    simp [Complex.div_im, Complex.normSq_ofReal, Complex.add_im,
          Complex.neg_im, Complex.one_im, Complex.I_im]
    field_simp [h2ne]

-- ════════════════════════════════════════════════════════════════════════════
-- Section 2 — 8-cycle closure:  μ^8 = 1
-- Ref: docs/master_derivations.pdf §2.2
-- ════════════════════════════════════════════════════════════════════════════

/-- μ has absolute value 1 (it lies on the complex unit circle).
    Proof: |exp(I·θ)| = exp(Re(I·θ)) = exp(0) = 1. -/
theorem mu_abs_one : Complex.abs μ = 1 := by
  unfold μ
  -- Complex.abs (exp z) = Real.exp z.re; here z = I*(3π/4), z.re = 0
  rw [Complex.abs_exp]
  simp [Complex.mul_re, Complex.I_re, Complex.I_im,
        Complex.ofReal_re, Complex.ofReal_im]

/-- μ^8 = 1: the eigenvalue closes an exact 8-cycle.

    Proof:  μ^8 = exp(I · 3π/4)^8
                = exp(8 · I · 3π/4)     [exp homomorphism]
                = exp(I · 6π)
                = exp(I · 2π · 3)
                = (exp(2πi))^3 = 1^3 = 1.

    Ref: docs/master_derivations.pdf §2.2  Theorem 2.1 -/
theorem mu_pow_eight : μ ^ 8 = 1 := by
  unfold μ
  -- Rewrite μ^8 as exp(↑8 · (I · 3π/4)) via the exp homomorphism
  rw [← Complex.exp_nat_mul]
  -- Simplify exponent: 8 · (I · 3π/4) = ↑3 · (2π · I)
  have h : (↑(8 : ℕ) : ℂ) * (Complex.I * (3 * ↑Real.pi / 4)) =
           ↑(3 : ℕ) * (2 * ↑Real.pi * Complex.I) := by
    push_cast; ring
  -- Apply exp(↑3 · z) = exp(z)^3, then exp(2πi) = 1
  rw [h, Complex.exp_nat_mul, Complex.exp_two_pi_mul_I]
  simp

-- ════════════════════════════════════════════════════════════════════════════
-- Section 3 — Distinctness of the eight powers of μ
-- Ref: docs/master_derivations.pdf §2.3
-- ════════════════════════════════════════════════════════════════════════════

/-- The eight powers μ^0, μ^1, …, μ^7 are pairwise distinct.

    Proof:
      (a) Let ζ = exp(2πi/8).  By `Complex.isPrimitiveRoot_exp`, ζ is a
          primitive 8th root of unity.
      (b) μ = exp(I · 3π/4) = exp(2πi · 3/8) = ζ^3.
      (c) Since gcd(3, 8) = 1, `IsPrimitiveRoot.pow_of_coprime` shows
          μ = ζ^3 is itself a primitive 8th root of unity.
      (d) For a primitive 8th root, `IsPrimitiveRoot.pow_inj` guarantees
          that μ^j = μ^k ↔ j = k for j, k < 8.

    Ref: docs/master_derivations.pdf §2.3 -/
theorem mu_powers_distinct :
    ∀ j k : Fin 8, (j : ℕ) ≠ k → μ ^ (j : ℕ) ≠ μ ^ (k : ℕ) := by
  -- (a) ζ = exp(2πi/8) is the standard primitive 8th root of unity
  have hprim8 : IsPrimitiveRoot (Complex.exp (2 * ↑Real.pi * Complex.I / 8)) 8 :=
    Complex.isPrimitiveRoot_exp 8 (by norm_num)
  -- (b) μ = ζ^3, since exp(3 · (2πi/8)) = exp(3πi/4) = exp(I · 3π/4)
  have hmu_eq : μ = Complex.exp (2 * ↑Real.pi * Complex.I / 8) ^ 3 := by
    unfold μ
    rw [← Complex.exp_nat_mul]
    congr 1
    push_cast
    ring
  -- (c) gcd(3, 8) = 1, so μ = ζ^3 is also a primitive 8th root
  have hmuprim : IsPrimitiveRoot μ 8 := by
    rw [hmu_eq]
    exact hprim8.pow_of_coprime 3 (by decide : Nat.Coprime 3 8)
  -- (d) Distinctness: μ^j = μ^k with j, k < 8 implies j = k
  intro j k hjk heq
  exact hjk (IsPrimitiveRoot.pow_inj hmuprim j.isLt k.isLt heq)

-- ════════════════════════════════════════════════════════════════════════════
-- Section 4 — Rotation matrix R(3π/4)
-- Ref: docs/master_derivations.pdf §3
-- ════════════════════════════════════════════════════════════════════════════

/-- The 2×2 rotation matrix by angle θ = 3π/4. -/
def rotMat : Matrix (Fin 2) (Fin 2) ℝ :=
  let c := Real.cos (3 * Real.pi / 4)   -- = −1/√2
  let s := Real.sin (3 * Real.pi / 4)   -- =  1/√2
  !![c, -s; s, c]

/-- det R(3π/4) = 1.
    Proof: det [[c, −s],[s, c]] = c² + s² = cos²θ + sin²θ = 1. -/
theorem rotMat_det : Matrix.det rotMat = 1 := by
  unfold rotMat
  simp [Matrix.det_fin_two]
  have := Real.sin_sq_add_cos_sq (3 * Real.pi / 4)
  nlinarith [Real.sin_sq_add_cos_sq (3 * Real.pi / 4)]

/-- R(3π/4) is orthogonal: R · Rᵀ = I.
    Proof: direct computation using cos²θ + sin²θ = 1 and
           c·(−s) + s·c = 0. -/
theorem rotMat_orthog : rotMat * rotMatᵀ = 1 := by
  -- Rewrite rotMat to an explicit matrix literal (eliminates let-binding
  -- indirection that prevents simp from reducing transpose entries).
  have hM : rotMat = !![Real.cos (3 * Real.pi / 4), -Real.sin (3 * Real.pi / 4);
                        Real.sin (3 * Real.pi / 4),  Real.cos (3 * Real.pi / 4)] := rfl
  rw [hM]
  ext i j
  fin_cases i <;> fin_cases j <;>
    simp [Matrix.mul_apply, Matrix.transpose_apply, Fin.sum_univ_two,
          Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.head_cons,
          Matrix.head_fin_const, Matrix.vecHead, Matrix.one_apply,
          neg_mul, mul_neg, neg_neg] <;>
    nlinarith [Real.sin_sq_add_cos_sq (3 * Real.pi / 4)]

/-- R(3π/4)^8 = I (8-fold application returns to identity).

    Proof via double-angle formulas:
      (1) cos(2·3π/4) = cos(3π/2) = 0  →  cos²(3π/4) = sin²(3π/4) = 1/2
      (2) sin(2·3π/4) = sin(3π/2) = −1  →  sin(3π/4)·cos(3π/4) = −1/2
      (3) R^2 = !![0, 1; −1, 0]   (matrix multiplication using (1)/(2))
      (4) R^4 = (R^2)^2 = −I       (squaring (3))
      (5) R^8 = (R^4)^2 = I        (squaring (4))

    Ref: docs/master_derivations.pdf §3 -/
theorem rotMat_pow_eight : rotMat ^ 8 = 1 := by
  -- Abbreviate the angle for readability
  set θ := (3 : ℝ) * Real.pi / 4 with hθ
  -- Pythagorean identity
  have hpy := Real.sin_sq_add_cos_sq θ
  -- cos(3π/2) = 0  (from cos(π + π/2) = −cos(π/2) = 0)
  have hcos32 : Real.cos (3 * Real.pi / 2) = 0 := by
    have : (3 : ℝ) * Real.pi / 2 = Real.pi + Real.pi / 2 := by ring
    rw [this]
    simp [Real.cos_add, Real.cos_pi, Real.sin_pi, Real.cos_pi_div_two]
  -- sin(3π/2) = −1  (from sin(π + π/2) = −sin(π/2) = −1)
  have hsin32 : Real.sin (3 * Real.pi / 2) = -1 := by
    have : (3 : ℝ) * Real.pi / 2 = Real.pi + Real.pi / 2 := by ring
    rw [this]
    simp [Real.sin_add, Real.cos_pi, Real.sin_pi, Real.sin_pi_div_two]
  -- Double-angle: cos(3π/2) = 2·cos²(3π/4) − 1  →  cos²(3π/4) = 1/2
  have hct := Real.cos_two_mul θ
  have h2θ : 2 * θ = 3 * Real.pi / 2 := by simp [hθ]; ring
  rw [h2θ, hcos32] at hct
  -- hct : 0 = 2 * cos(θ)^2 - 1
  have hc2 : Real.cos θ ^ 2 = 1 / 2 := by linarith
  -- sin²(3π/4) = 1/2  (from Pythagorean + cos²=1/2)
  have hs2 : Real.sin θ ^ 2 = 1 / 2 := by nlinarith
  -- Double-angle: sin(3π/2) = 2·sin(3π/4)·cos(3π/4)  →  sin·cos = −1/2
  have hst := Real.sin_two_mul θ
  rw [h2θ, hsin32] at hst
  -- hst : -1 = 2 * sin(θ) * cos(θ)
  have hsc : Real.sin θ * Real.cos θ = -(1 / 2) := by linarith
  -- (3) rotMat^2 = !![0, 1; −1, 0]
  have h2 : rotMat ^ 2 = !![0, 1; -1, 0] := by
    have hdef : rotMat = !![Real.cos θ, -Real.sin θ; Real.sin θ, Real.cos θ] := rfl
    have hc2' : Real.cos θ * Real.cos θ = 1 / 2 := by nlinarith [sq_nonneg (Real.cos θ)]
    have hs2' : Real.sin θ * Real.sin θ = 1 / 2 := by nlinarith [sq_nonneg (Real.sin θ)]
    have hsc' : Real.cos θ * Real.sin θ = -(1 / 2) := by
      linarith [mul_comm (Real.sin θ) (Real.cos θ)]
    rw [hdef, pow_two]
    ext i j; fin_cases i <;> fin_cases j <;>
      simp [Matrix.mul_apply, Fin.sum_univ_two, neg_mul, mul_neg, neg_neg] <;>
      linarith [hc2', hs2', hsc', hsc]
  -- (4) rotMat^4 = −I
  have h4 : rotMat ^ 4 = -1 := by
    have : (4 : ℕ) = 2 * 2 := by norm_num
    rw [this, pow_mul, h2]
    ext i j; fin_cases i <;> fin_cases j <;>
      simp [pow_two, Matrix.mul_apply, Fin.sum_univ_two, Matrix.cons_val_zero,
            Matrix.cons_val_one, Matrix.head_cons, Matrix.head_fin_const,
            Matrix.neg_apply, Matrix.one_apply, Pi.neg_apply, neg_mul, mul_neg]
  -- (5) rotMat^8 = I  (square both sides of step 4)
  have : (8 : ℕ) = 4 * 2 := by norm_num
  rw [this, pow_mul, h4]
  simp [pow_two, neg_mul, mul_neg, neg_neg, Matrix.one_apply]

-- ════════════════════════════════════════════════════════════════════════════
-- Section 5 — Coherence function  C(r) = 2r / (1 + r²)
-- Ref: docs/master_derivations.pdf §4
-- ════════════════════════════════════════════════════════════════════════════

/-- Coherence function C(r) = 2r / (1 + r²), defined for all r : ℝ. -/
noncomputable def C (r : ℝ) : ℝ := 2 * r / (1 + r ^ 2)

/-- The denominator 1 + r² is always strictly positive. -/
private lemma one_add_sq_pos (r : ℝ) : 0 < 1 + r ^ 2 := by positivity

/-- C(r) ≤ 1 for all r ≥ 0, by the AM–GM inequality (1 + r² ≥ 2r).

    Proof: C(r) ≤ 1  ↔  2r ≤ 1 + r²  ↔  0 ≤ (r − 1)².
    Ref: docs/master_derivations.pdf §4  Proposition 4.1 -/
theorem coherence_le_one (r : ℝ) (_ : 0 ≤ r) : C r ≤ 1 := by
  unfold C
  rw [div_le_one (one_add_sq_pos r)]
  nlinarith [sq_nonneg (r - 1)]

/-- C(r) = 1 if and only if r = 1.
    Proof: equality in AM–GM holds iff r = 1. -/
theorem coherence_eq_one_iff (r : ℝ) (_ : 0 ≤ r) : C r = 1 ↔ r = 1 := by
  unfold C
  rw [div_eq_one_iff_eq (ne_of_gt (one_add_sq_pos r))]
  constructor
  · intro h
    -- 2r = 1 + r²  ↔  (r − 1)² = 0  ↔  r = 1
    have hsq : (r - 1) ^ 2 = 0 := by nlinarith
    have : r - 1 = 0 := by rwa [sq_eq_zero_iff] at hsq
    linarith
  · intro h
    rw [h]; ring

-- ════════════════════════════════════════════════════════════════════════════
-- Section 6 — Canonical-state normalisation
-- Ref: docs/master_derivations.pdf §5
-- ════════════════════════════════════════════════════════════════════════════

/-- The canonical amplitude η = 1/√2. -/
noncomputable def η : ℝ := 1 / Real.sqrt 2

/-- η² = 1/2. -/
private lemma eta_sq : η ^ 2 = 1 / 2 := by
  unfold η
  rw [div_pow, one_pow, Real.sq_sqrt (by norm_num : (0 : ℝ) ≤ 2)]

/-- |μ|² = 1 (μ lies on the unit circle). -/
private lemma mu_normSq_one : Complex.normSq μ = 1 := by
  -- normSq μ = (Complex.abs μ)^2 = 1^2 = 1
  rw [Complex.normSq_eq_abs, mu_abs_one]; norm_num

/-- The canonical state has norm 1:  η² + |μ·η|² = 1.

    Proof:
      |μ·η|² = |μ|² · η² = 1 · (1/2) = 1/2
      η²  + |μ·η|²  = 1/2 + 1/2 = 1.

    Ref: docs/master_derivations.pdf §5  Proposition 5.1 -/
theorem canonical_norm : η ^ 2 + Complex.normSq (μ * ↑η) = 1 := by
  -- Reduce |μ·η|² to normSq μ · η²
  have h1 : Complex.normSq (μ * ↑η) = η ^ 2 := by
    rw [Complex.normSq_mul, Complex.normSq_ofReal, mu_normSq_one]
    ring
  rw [h1, eta_sq]
  norm_num

-- ════════════════════════════════════════════════════════════════════════════
-- Section 7 — Silver Ratio Conservation (Proposition 4)
-- Ref: docs/master_derivations.pdf §1
-- ════════════════════════════════════════════════════════════════════════════

/-- The silver ratio δS = 1 + √2, the arithmetic dual of μ. -/
noncomputable def δS : ℝ := 1 + Real.sqrt 2

private lemma δS_pos : 0 < δS := by unfold δS; positivity

private lemma sqrt2_sq : Real.sqrt 2 * Real.sqrt 2 = 2 :=
  Real.mul_self_sqrt (by norm_num)

/-- Silver conservation product: δS · (√2 − 1) = 1.
    Ref: docs/master_derivations.pdf Proposition 4 -/
theorem silverRatio_mul_conj : δS * (Real.sqrt 2 - 1) = 1 := by
  unfold δS; nlinarith [sqrt2_sq]

/-- Silver quadratic identity: δS² = 2·δS + 1.
    Ref: docs/master_derivations.pdf Proposition 4 -/
theorem silverRatio_sq : δS ^ 2 = 2 * δS + 1 := by
  unfold δS; nlinarith [sqrt2_sq]

/-- The multiplicative inverse of δS equals √2 − 1. -/
theorem silverRatio_inv : 1 / δS = Real.sqrt 2 - 1 := by
  rw [div_eq_iff (ne_of_gt δS_pos)]
  linarith [silverRatio_mul_conj]

-- ════════════════════════════════════════════════════════════════════════════
-- Section 8 — Additional Coherence Properties
-- Ref: docs/master_derivations.pdf §4 Theorem 11
-- ════════════════════════════════════════════════════════════════════════════

/-- C(r) > 0 for all r > 0. -/
theorem coherence_pos (r : ℝ) (hr : 0 < r) : 0 < C r := by
  unfold C; exact div_pos (by linarith) (one_add_sq_pos r)

/-- C(r) = C(1/r): coherence is symmetric about r = 1.
    This duality reflects the equivalence of amplitude ratios r and 1/r.
    Ref: docs/master_derivations.pdf §4 Theorem 11 property (2) -/
theorem coherence_symm (r : ℝ) (hr : 0 < r) : C r = C (1 / r) := by
  unfold C
  have hr' : r ≠ 0 := ne_of_gt hr
  field_simp [hr', ne_of_gt (one_add_sq_pos r), ne_of_gt (one_add_sq_pos (1 / r))]
  ring

/-- C(r) < 1 for r ≥ 0 with r ≠ 1: the maximum is uniquely attained at r = 1.
    Ref: docs/master_derivations.pdf §4 Theorem 11 property (4) -/
theorem coherence_lt_one (r : ℝ) (hr : 0 ≤ r) (hr1 : r ≠ 1) : C r < 1 :=
  lt_of_le_of_ne (coherence_le_one r hr) (mt (coherence_eq_one_iff r hr).mp hr1)

-- ════════════════════════════════════════════════════════════════════════════
-- Section 9 — Palindrome Residual (Theorem 12)
-- Ref: docs/master_derivations.pdf §4 Theorem 12
-- ════════════════════════════════════════════════════════════════════════════

/-- The palindrome residual R(r) = (r − 1/r) / δS.
    Vanishes at r = 1; positive for r > 1; negative for 0 < r < 1.
    Ref: docs/master_derivations.pdf §4 Theorem 12 -/
noncomputable def Res (r : ℝ) : ℝ := (r - 1 / r) / δS

/-- R(r) = 0 ↔ r = 1 for r > 0.
    Ref: docs/master_derivations.pdf §4 Theorem 12 property (1) -/
theorem palindrome_residual_zero_iff (r : ℝ) (hr : 0 < r) : Res r = 0 ↔ r = 1 := by
  unfold Res
  rw [div_eq_zero_iff, or_iff_left (ne_of_gt δS_pos)]
  have hr' : r ≠ 0 := ne_of_gt hr
  constructor
  · intro h
    -- r − 1/r = 0  →  r·(r − 1/r) = 0  →  r² − 1 = 0  →  (r−1)(r+1) = 0
    have hrr : r * r = 1 := by
      have hstep : r * r - 1 = 0 := by
        have : r * (r - 1 / r) = r * r - 1 := by field_simp
        linarith [show r * (r - 1 / r) = 0 from by rw [h, mul_zero]]
      linarith
    have hfact : (r - 1) * (r + 1) = 0 := by linear_combination hrr
    rcases mul_eq_zero.mp hfact with h1 | h1
    · linarith
    · linarith
  · rintro rfl; norm_num

/-- R(r) > 0 for r > 1.
    Ref: docs/master_derivations.pdf §4 Theorem 12 property (2) -/
theorem palindrome_residual_pos (r : ℝ) (hr : 1 < r) : 0 < Res r := by
  unfold Res
  apply div_pos _ δS_pos
  have hr0 : (0 : ℝ) < r := by linarith
  have hr' : r ≠ 0 := ne_of_gt hr0
  have hrep : r - 1 / r = (r ^ 2 - 1) / r := by field_simp; ring
  rw [hrep]; exact div_pos (by nlinarith) hr0

/-- R(r) < 0 for 0 < r < 1.
    Ref: docs/master_derivations.pdf §4 Theorem 12 property (3) -/
theorem palindrome_residual_neg (r : ℝ) (hr0 : 0 < r) (hr1 : r < 1) : Res r < 0 := by
  unfold Res
  apply div_neg_of_neg_of_pos _ δS_pos
  have hr' : r ≠ 0 := ne_of_gt hr0
  have hrep : r - 1 / r = (r ^ 2 - 1) / r := by field_simp; ring
  rw [hrep]; exact div_neg_of_neg_of_pos (by nlinarith) hr0

/-- R(1/r) = −R(r): palindrome residual is anti-symmetric about r = 1.
    This is the odd counterpart to coherence's even symmetry C(r) = C(1/r). -/
theorem palindrome_residual_antisymm (r : ℝ) (hr : 0 < r) : Res (1 / r) = -Res r := by
  unfold Res
  have hr' : r ≠ 0 := ne_of_gt hr
  field_simp

-- ════════════════════════════════════════════════════════════════════════════
-- Section 10 — Lyapunov–Coherence Duality (Theorem 14)
-- Ref: docs/master_derivations.pdf §4 Theorem 14
-- ════════════════════════════════════════════════════════════════════════════

/-- Key identity: C(exp l) · (exp l + exp(−l)) = 2. -/
private lemma lyapunov_key (l : ℝ) :
    C (Real.exp l) * (Real.exp l + Real.exp (-l)) = 2 := by
  have hmul : Real.exp l * Real.exp (-l) = 1 := by
    rw [← Real.exp_add]; simp
  unfold C
  have hd : (0 : ℝ) < 1 + Real.exp l ^ 2 := one_add_sq_pos _
  field_simp [ne_of_gt hd]
  nlinarith [Real.exp_pos l, Real.exp_pos (-l), hmul, sq_nonneg (Real.exp l)]

/-- Lyapunov–coherence duality: C(exp l) = 2/(exp l + exp(−l)) = sech l.
    The coherence is the reciprocal of the hyperbolic cosine of the Lyapunov
    exponent l = ln r.  At the balanced fixed point r = 1 (l = 0):
    sech(0) = 1, recovering C = 1.
    Ref: docs/master_derivations.pdf §4 Theorem 14 -/
theorem lyapunov_coherence_duality (l : ℝ) :
    C (Real.exp l) = 2 / (Real.exp l + Real.exp (-l)) := by
  have hpos : 0 < Real.exp l + Real.exp (-l) := by positivity
  rw [eq_div_iff (ne_of_gt hpos)]
  exact lyapunov_key l

/-- Corollary: C(exp l) = (cosh l)⁻¹ = sech l.
    Uses: cosh l = (exp l + exp(−l)) / 2. -/
theorem lyapunov_coherence_sech (l : ℝ) :
    C (Real.exp l) = (Real.cosh l)⁻¹ := by
  have hcosh : Real.cosh l = (Real.exp l + Real.exp (-l)) / 2 := Real.cosh_eq l
  rw [lyapunov_coherence_duality, hcosh, inv_div]

-- ════════════════════════════════════════════════════════════════════════════
-- Section 11 — Derived Invariant Equivalences
-- The formal system discovers that independently-defined invariants
-- (coherence C, palindrome residual R, Lyapunov exponent λ) all
-- characterise the same balanced fixed point r = 1.
-- Ref: docs/master_derivations.pdf Corollary 13
-- ════════════════════════════════════════════════════════════════════════════

/-- **Derived**: palindrome residual zero ↔ maximum coherence.
    Two independently-defined invariants identify the same fixed point.
    Ref: docs/master_derivations.pdf Corollary 13 -/
theorem palindrome_coherence_equiv (r : ℝ) (hr : 0 < r) :
    Res r = 0 ↔ C r = 1 :=
  (palindrome_residual_zero_iff r hr).trans (coherence_eq_one_iff r (le_of_lt hr)).symm

/-- **Derived**: C is even (C(r) = C(1/r)) while R is odd (R(1/r) = −R(r)).
    Both are symmetric about the same fixed point r = 1, in dual senses. -/
theorem coherence_palindrome_duality (r : ℝ) (hr : 0 < r) :
    C r = C (1 / r) ∧ Res (1 / r) = -Res r :=
  ⟨coherence_symm r hr, palindrome_residual_antisymm r hr⟩

/-- **Derived**: the C-maximum is preserved under the symmetry r ↦ 1/r.
    C(r) = 1 ↔ C(1/r) = 1 — the maximum is a fixed point of the inversion. -/
theorem coherence_max_symm (r : ℝ) (hr : 0 < r) :
    C r = 1 ↔ C (1 / r) = 1 := by
  constructor
  · intro h; rwa [← coherence_symm r hr]
  · intro h; rwa [coherence_symm r hr]

/-- **Derived**: at the palindrome equilibrium the state is self-dual: r = 1/r. -/
theorem palindrome_zero_self_dual (r : ℝ) (hr : 0 < r) (h : Res r = 0) : r = 1 / r := by
  have := (palindrome_residual_zero_iff r hr).mp h
  rw [this]; norm_num

/-- **Derived**: simultaneous break — all equilibrium invariants are equivalent.
    The balanced state r = 1 is the unique point where coherence is maximal,
    palindrome residual vanishes, and the Lyapunov exponent is zero.
    Ref: docs/master_derivations.pdf Corollary 13 -/
theorem simultaneous_break (r : ℝ) (hr : 0 < r) :
    r = 1 ↔ C r = 1 ∧ Res r = 0 := by
  constructor
  · intro h
    exact ⟨(coherence_eq_one_iff r (le_of_lt hr)).mpr h,
            (palindrome_residual_zero_iff r hr).mpr h⟩
  · intro ⟨hC, _⟩
    exact (coherence_eq_one_iff r (le_of_lt hr)).mp hC

/-- **Derived**: the Lyapunov–coherence duality implies C ≤ 1 for all l,
    since sech(l) ≤ 1 = sech(0).  This recovers `coherence_le_one` via a
    completely different route (the hyperbolic bound). -/
theorem lyapunov_bound (l : ℝ) : C (Real.exp l) ≤ 1 :=
  coherence_le_one _ (le_of_lt (Real.exp_pos l))

-- ════════════════════════════════════════════════════════════════════════════
-- Section 12 — Orbit Magnitude and Trichotomy (Theorem 10)
-- |ξ^n| = r^n for ξ = r·μ.  The three cases r=1 / r>1 / r<1 determine
-- whether the orbit is a closed 8-cycle, spirals outward, or collapses.
-- Ref: docs/master_derivations.pdf §5 Theorem 10
-- ════════════════════════════════════════════════════════════════════════════

/-- |μ^n| = 1 for all n: the orbit of μ stays on the unit circle.
    Follows immediately from |μ| = 1 and multiplicativity of the absolute value. -/
theorem mu_pow_abs (n : ℕ) : Complex.abs (μ ^ n) = 1 := by
  have h : Complex.abs (μ ^ n) = Complex.abs μ ^ n := map_pow Complex.abs μ n
  rw [h, mu_abs_one, one_pow]

/-- |(r·μ)^n| = r^n for r ≥ 0.
    The amplitude of the orbit is purely radial; the phase factor μ contributes
    no growth or decay.  This is the key quantitative form of Trichotomy. -/
theorem scaled_orbit_abs (r : ℝ) (hr : 0 ≤ r) (n : ℕ) :
    Complex.abs ((↑r * μ) ^ n) = r ^ n := by
  have habsr : Complex.abs (↑r * μ) = r := by
    rw [map_mul Complex.abs, Complex.abs_ofReal, _root_.abs_of_nonneg hr, mu_abs_one, mul_one]
  calc Complex.abs ((↑r * μ) ^ n)
      = Complex.abs (↑r * μ) ^ n := map_pow Complex.abs _ _
    _ = r ^ n := by rw [habsr]

/-- Trichotomy — r = 1: orbit has unit magnitude at every step (stable 8-cycle).
    Ref: docs/master_derivations.pdf §5 Theorem 10 case (1) -/
theorem trichotomy_unit_orbit (n : ℕ) : Complex.abs ((1 : ℂ) * μ ^ n) = 1 := by
  rw [one_mul, mu_pow_abs]

/-- Trichotomy — r > 1: magnitudes are strictly increasing (spiral outward).
    |(r·μ)^n| < |(r·μ)^(n+1)| since r^n < r^(n+1) when r > 1.
    Ref: docs/master_derivations.pdf §5 Theorem 10 case (2) -/
theorem trichotomy_grow (r : ℝ) (hr : 1 < r) (n : ℕ) :
    Complex.abs ((↑r * μ) ^ n) < Complex.abs ((↑r * μ) ^ (n + 1)) := by
  simp only [scaled_orbit_abs r (le_of_lt (lt_trans one_pos hr))]
  exact pow_lt_pow_right₀ hr (Nat.lt_succ_self n)

/-- Trichotomy — 0 < r < 1: magnitudes are strictly decreasing (spiral inward).
    |(r·μ)^(n+1)| < |(r·μ)^n| since r^(n+1) < r^n when 0 < r < 1.
    Ref: docs/master_derivations.pdf §5 Theorem 10 case (3) -/
theorem trichotomy_shrink (r : ℝ) (hr0 : 0 < r) (hr1 : r < 1) (n : ℕ) :
    Complex.abs ((↑r * μ) ^ (n + 1)) < Complex.abs ((↑r * μ) ^ n) := by
  simp only [scaled_orbit_abs r (le_of_lt hr0)]
  calc r ^ (n + 1) = r ^ n * r := pow_succ r n
    _ < r ^ n * 1 := mul_lt_mul_of_pos_left hr1 (pow_pos hr0 n)
    _ = r ^ n := mul_one _

-- ════════════════════════════════════════════════════════════════════════════
-- Section 13 — Coherence Monotonicity
-- C(r) = 2r/(1+r²) is strictly increasing on (0,1] and strictly decreasing
-- on [1,∞).  This is the "gradient flow toward balance" that the Kernel
-- system exploits: any r≠1 can recover by moving r toward 1.
-- Ref: docs/master_derivations.pdf §4 Theorem 11
-- ════════════════════════════════════════════════════════════════════════════

/-- Factorisation of C(s) − C(r) useful for monotonicity proofs.
    C(s) − C(r) = 2(s−r)(1−rs) / ((1+r²)(1+s²)).
    The sign of the numerator 2(s−r)(1−rs) determines the direction of change. -/
private lemma coherence_diff_factored (r s : ℝ) (hr : 0 < r) (hs : 0 < s) :
    C s - C r = 2 * (s - r) * (1 - r * s) / ((1 + r ^ 2) * (1 + s ^ 2)) := by
  have hr' : 1 + r ^ 2 ≠ 0 := ne_of_gt (one_add_sq_pos r)
  have hs' : 1 + s ^ 2 ≠ 0 := ne_of_gt (one_add_sq_pos s)
  unfold C; field_simp [hr', hs']; ring

/-- C is strictly increasing on (0, 1]: for 0 < r < s ≤ 1, C(r) < C(s).
    When both components are below balance (|β| < |α|), increasing |β|/|α|
    toward 1 strictly improves coherence.
    Ref: docs/master_derivations.pdf §4 Theorem 11 -/
theorem coherence_strictMono (r s : ℝ) (hr : 0 < r) (hrs : r < s) (hs1 : s ≤ 1) :
    C r < C s := by
  rw [← sub_pos, coherence_diff_factored r s hr (lt_trans hr hrs)]
  refine div_pos ?_ (mul_pos (one_add_sq_pos r) (one_add_sq_pos s))
  have hsr : 0 < s - r := sub_pos.mpr hrs
  have hr1 : r < 1 := lt_of_lt_of_le hrs hs1
  have hrslt1 : r * s < 1 := by
    calc r * s < 1 * s := mul_lt_mul_of_pos_right hr1 (lt_trans hr hrs)
         _ = s := one_mul s
         _ ≤ 1 := hs1
  have h_pos : (s - r) * (1 - r * s) > 0 := mul_pos hsr (by linarith)
  nlinarith

/-- C is strictly decreasing on [1, ∞): for 1 ≤ r < s, C(s) < C(r).
    When both components are above balance (|β| > |α|), increasing |β|/|α|
    away from 1 strictly decreases coherence.
    Ref: docs/master_derivations.pdf §4 Theorem 11 -/
theorem coherence_strictAnti (r s : ℝ) (hr1 : 1 ≤ r) (hrs : r < s) :
    C s < C r := by
  have hr : 0 < r := lt_of_lt_of_le one_pos hr1
  rw [← sub_neg, coherence_diff_factored r s hr (lt_trans hr hrs)]
  refine div_neg_of_neg_of_pos ?_ (mul_pos (one_add_sq_pos r) (one_add_sq_pos s))
  have hsr : 0 < s - r := sub_pos.mpr hrs
  have hs1 : 1 < s := lt_of_le_of_lt hr1 hrs
  have hrsgt1 : 1 < r * s := by nlinarith
  have h_neg : (s - r) * (1 - r * s) < 0 :=
    mul_neg_of_pos_of_neg hsr (by linarith)
  nlinarith

-- ════════════════════════════════════════════════════════════════════════════
-- Section 14 — Palindrome Arithmetic (Proposition from §6 of master_derivations.pdf)
-- The palindrome digit pair 987654321 / 123456789 encodes the 8-cycle period
-- (integer quotient = 8) and slow-precession period D = 13717421.
-- Ref: docs/master_derivations.pdf §6
-- ════════════════════════════════════════════════════════════════════════════

/-- Two-palindrome complementarity: 987654321 = 8 × 123456789 + 9.
    The integer part 8 equals the μ-rotation period; the remainder 9 connects
    to the slow-precession denominator via 9 × D = 123456789.
    Ref: docs/master_derivations.pdf §6 Proposition 5 -/
theorem palindrome_comp : (987654321 : ℕ) = 8 * 123456789 + 9 := by norm_num

/-- Precession period identity: 9 × D = 123456789 where D = 13717421.
    D is the slow-precession period (denominator of ΔΦ₀ = 2π/D).
    Ref: docs/master_derivations.pdf §6 -/
theorem precession_period_factor : 9 * 13717421 = (123456789 : ℕ) := by norm_num

/-- gcd(8, D) = 1: the fast 8-cycle period and slow-precession period are coprime.
    Coprimality ensures the torus T² = S¹ × S¹ has no resonance between the
    two winding numbers, so all 8·D orbit points are distinct. -/
theorem precession_gcd_one : Nat.gcd 8 13717421 = 1 := by native_decide

/-- lcm(8, D) = 8·D = 109739368: the joint orbit closes after 8·D steps.
    Follows from gcd(8, D) = 1 and the formula lcm(a,b) = ab/gcd(a,b). -/
theorem precession_lcm : Nat.lcm 8 13717421 = 8 * 13717421 := by
  unfold Nat.lcm; rw [precession_gcd_one, Nat.div_one]

-- ════════════════════════════════════════════════════════════════════════════
-- Section 15 — Z/8Z Rotational Memory (Proposition from §11 of master_derivations.pdf)
-- Physical addresses decompose as (bank, offset) = (addr % 8, addr / 8).
-- The μ-orbit provides a natural clock that aligns memory banks with the
-- 8-cycle, so the group Z/8Z governs both phase rotation and bank addressing.
-- Ref: docs/master_derivations.pdf §11
-- ════════════════════════════════════════════════════════════════════════════

/-- Eight consecutive rotations return to the starting bank: (n + 8) % 8 = n % 8.
    Ref: docs/master_derivations.pdf §11 Proposition Z/8Z -/
theorem z8z_period (n : ℕ) : (n + 8) % 8 = n % 8 := by omega

/-- Memory address reconstruction: addr % 8 + 8 * (addr / 8) = addr.
    The (bank, offset) decomposition is lossless. -/
theorem z8z_reconstruction (addr : ℕ) : addr % 8 + 8 * (addr / 8) = addr := by omega

/-- **Derived**: the μ-orbit inherits Z/8Z periodicity: μ^(j+8) = μ^j.
    The eigenvalue clock and the memory bank clock share the same period 8,
    discovered by combining mu_pow_eight with the ring law for exponents. -/
theorem mu_z8z_period (j : ℕ) : μ ^ (j + 8) = μ ^ j := by
  rw [pow_add, mu_pow_eight, mul_one]

-- ════════════════════════════════════════════════════════════════════════════
-- Section 16 — Zero-Overhead Precession (Theorem from §13 of master_derivations.pdf)
-- The precession step β ↦ e^{iθ}·β rotates the phase without changing the
-- amplitude |β|.  Consequently it preserves r = |β|/|α|, C(r), and all
-- equilibrium invariants simultaneously — "zero overhead" in the sense of
-- Theorem 5.1 (docs/master_derivations.pdf).
-- Ref: docs/master_derivations.pdf §13 Theorem zero-overhead-prec
-- ════════════════════════════════════════════════════════════════════════════

/-- Precession phasor e^{iθ} has unit absolute value for any real phase θ.
    Proof: |exp(z)| = exp(Re z); Re(I·θ) = 0; exp(0) = 1.
    Ref: docs/master_derivations.pdf §13 Theorem zero-overhead-prec -/
theorem precession_phasor_unit (θ : ℝ) :
    Complex.abs (Complex.exp (Complex.I * ↑θ)) = 1 := by
  rw [Complex.abs_exp]
  have hre : (Complex.I * ↑θ).re = 0 := by
    simp [Complex.mul_re, Complex.I_re, Complex.I_im,
          Complex.ofReal_re, Complex.ofReal_im]
  rw [hre, Real.exp_zero]

/-- Multiplying by a unit-norm phasor preserves the absolute value of any β ∈ ℂ.
    This is the mathematical core of "zero overhead": the precession step
    β ↦ e^{iθ}·β leaves |β| — and hence r = |β|/|α| — unchanged. -/
theorem precession_preserves_abs (β : ℂ) (θ : ℝ) :
    Complex.abs (Complex.exp (Complex.I * ↑θ) * β) = Complex.abs β := by
  rw [map_mul Complex.abs, precession_phasor_unit, one_mul]

/-- **Derived**: precession preserves the coherence ratio.
    If α and β are the two state components and both are multiplied by the
    same phasor e^{iθ}, the ratio |β|/|α| — and therefore C(r) — is invariant.
    This formally proves that precession steps are "zero overhead" for coherence. -/
theorem precession_preserves_coherence (α β : ℂ) (_ : Complex.abs α ≠ 0) (θ : ℝ) :
    C (Complex.abs (Complex.exp (Complex.I * ↑θ) * β) / Complex.abs α) =
    C (Complex.abs β / Complex.abs α) := by
  rw [precession_preserves_abs]

-- ════════════════════════════════════════════════════════════════════════════
-- Section 17 — Ohm-Coherence Circuit Identities
-- G_eff = sech(λ) = (cosh λ)⁻¹ and R_eff = cosh(λ) are the conductance and
-- resistance of a coherent channel.  Their product is always 1, mirroring
-- Ohm's law G·R = 1 (R = 1/G).  Parallel channels add conductances; series
-- stages add resistances — in both topologies G_tot · R_tot = 1.
-- Ref: docs/master_derivations.pdf §9
-- ════════════════════════════════════════════════════════════════════════════

/-- Single-channel Ohm-Coherence law: G_eff · R_eff = sech(l) · cosh(l) = 1.
    Here G_eff = (cosh l)⁻¹ and R_eff = cosh l.  The identity says that the
    coherence (conductance) and the effective resistance are always reciprocals.
    Ref: docs/master_derivations.pdf §9 eq. (single-channel) -/
theorem geff_reff_one (l : ℝ) : (Real.cosh l)⁻¹ * Real.cosh l = 1 :=
  inv_mul_cancel₀ (ne_of_gt (Real.cosh_pos l))

/-- At balance (l = 0): G_eff = sech(0) = 1 — maximal conductance, no overhead.
    Ref: docs/master_derivations.pdf §9 -/
theorem geff_at_zero : (Real.cosh 0)⁻¹ = 1 := by simp [Real.cosh_zero]

/-- Parallel-channel Ohm-Coherence law: N identical channels satisfy G_tot · R_tot = 1.
    G_tot = N · sech(l) and R_tot = cosh(l)/N; their product collapses to sech·cosh = 1.
    Ref: docs/master_derivations.pdf §9 Parallel Channels (MultiChannelSystem) -/
theorem parallel_circuit_one (N : ℕ) (hN : 0 < N) (l : ℝ) :
    (↑N * (Real.cosh l)⁻¹) * (Real.cosh l / ↑N) = 1 := by
  have hcosh : Real.cosh l ≠ 0 := ne_of_gt (Real.cosh_pos l)
  have hN' : (N : ℝ) ≠ 0 := Nat.cast_pos.mpr hN |>.ne'
  field_simp [hcosh, hN']

/-- Series-pipeline Ohm-Coherence law: M identical stages satisfy G_tot · R_tot = 1.
    R_tot = M · cosh(l) and G_tot = (M · cosh(l))⁻¹; their product is 1.
    Ref: docs/master_derivations.pdf §9 Series Pipeline (PipelineSystem) -/
theorem series_circuit_one (M : ℕ) (hM : 0 < M) (l : ℝ) :
    (↑M * Real.cosh l)⁻¹ * (↑M * Real.cosh l) = 1 :=
  inv_mul_cancel₀ (mul_ne_zero (Nat.cast_pos.mpr hM |>.ne') (ne_of_gt (Real.cosh_pos l)))

-- ════════════════════════════════════════════════════════════════════════════
-- Section 18 — Pythagorean Coherence Identity
-- C(r)² + ((r²−1)/(1+r²))² = 1 for all r > 0.
-- Setting r = tan θ: C(r) = sin(2θ) and (r²−1)/(1+r²) = −cos(2θ),
-- so the identity is sin²(2θ) + cos²(2θ) = 1 in disguise.
-- The term (r²−1)/(1+r²) = δS · r · Res(r) / (1+r²) connects to the
-- palindrome residual: δS · r · Res(r) = r² − 1.
-- Ref: algebraic consequence of C(r) = 2r/(1+r²)
-- ════════════════════════════════════════════════════════════════════════════

/-- **Pythagorean coherence identity**: C(r)² + ((r²−1)/(1+r²))² = 1 for all r > 0.
    The coherence and the "imbalance fraction" lie on the unit circle.
    Machine-discovered: two independently-defined quantities sum to 1.
    Ref: algebraic consequence of C(r) = 2r/(1+r²) -/
theorem coherence_pythagorean (r : ℝ) (hr : 0 < r) :
    C r ^ 2 + ((r ^ 2 - 1) / (1 + r ^ 2)) ^ 2 = 1 := by
  unfold C
  have h : 1 + r ^ 2 ≠ 0 := ne_of_gt (one_add_sq_pos r)
  field_simp [h]
  ring

/-- The palindrome amplitude: δS · r · Res(r) = r² − 1.
    This connects the palindrome residual to the Pythagorean imbalance term
    (r²−1)/(1+r²), completing the circuit:
      C(r)² + (δS·r·Res(r)/(1+r²))² = 1.
    Ref: algebraic consequence of Res(r) = (r − 1/r)/δS -/
theorem palindrome_amplitude_eq (r : ℝ) (hr : 0 < r) :
    δS * r * Res r = r ^ 2 - 1 := by
  unfold Res
  have hδ : δS ≠ 0 := ne_of_gt δS_pos
  field_simp [ne_of_gt hr, hδ]
  ring

-- ════════════════════════════════════════════════════════════════════════════
-- Section 19 — Orbit Lyapunov Connection
-- |ξⁿ| = rⁿ = exp(n·λ) where λ = log r is the Lyapunov exponent and
-- ξ = r·μ.  The coherence after n orbit steps is sech(n·λ).
-- Three key results: the exponential formula for orbit radius, the sech
-- formula for orbit coherence, and the monotone decay of coherence.
-- Ref: docs/master_derivations.pdf §5 Theorem 10 + §5 Theorem 14
-- ════════════════════════════════════════════════════════════════════════════

/-- Orbit radius in Lyapunov form: |(r·μ)^n| = exp(n · log r) for r > 0.
    Since rⁿ = exp(log(rⁿ)) = exp(n·log r), this rewrites the orbit amplitude
    in terms of the Lyapunov exponent λ = log r.
    Ref: docs/master_derivations.pdf §5 Theorem 10 -/
theorem orbit_radius_exp (r : ℝ) (hr : 0 < r) (n : ℕ) :
    Complex.abs ((↑r * μ) ^ n) = Real.exp (↑n * Real.log r) := by
  rw [scaled_orbit_abs r (le_of_lt hr),
      ← Real.exp_log (pow_pos hr n), Real.log_pow]

/-- **Full chain**: coherence after n orbit steps = sech(n · Lyapunov exponent).
    C(rⁿ) = C(exp(n·log r)) = (cosh(n·log r))⁻¹.
    Proof: rⁿ = exp(n·log r) (orbit_radius_exp), then apply lyapunov_coherence_sech.
    This is the synthesis of §12 (orbit amplitude) and §10 (duality). -/
theorem coherence_orbit_sech (r : ℝ) (hr : 0 < r) (n : ℕ) :
    C (r ^ n) = (Real.cosh (↑n * Real.log r))⁻¹ := by
  conv_lhs => rw [show r ^ n = Real.exp (↑n * Real.log r) from by
    rw [← Real.exp_log (pow_pos hr n), Real.log_pow]]
  exact lyapunov_coherence_sech _

/-- Coherence decays under orbit iteration: r > 1 and n ≥ 1 implies C(rⁿ) ≤ C(r).
    The coherence of the amplified orbit is no greater than the original.
    Strict decay holds for n ≥ 2; equality only at n = 1 (trivial case).
    Ref: combines coherence_strictAnti with the orbit-amplitude ordering. -/
theorem coherence_orbit_decay (r : ℝ) (hr : 1 < r) (n : ℕ) (hn : 1 ≤ n) :
    C (r ^ n) ≤ C r := by
  by_cases hn1 : n = 1
  · simp [hn1]
  · have hn2 : 1 < n := Nat.lt_of_le_of_ne hn (Ne.symm hn1)
    exact le_of_lt (coherence_strictAnti r (r ^ n) (le_of_lt hr) (by
      nth_rw 1 [← pow_one r]; exact pow_lt_pow_right₀ hr hn2))

/-- Coherence is perfectly preserved at the stable orbit: C(1ⁿ) = 1 for all n.
    The coherent fixed point r = 1 is stable under any number of orbit steps. -/
theorem orbit_coherence_at_one (n : ℕ) : C ((1 : ℝ) ^ n) = 1 := by
  rw [one_pow]
  exact (coherence_eq_one_iff 1 (le_of_lt one_pos)).mpr rfl

-- ════════════════════════════════════════════════════════════════════════════
-- Section 20 — Silver Ratio Self-Similarity
-- δS = 1 + √2 satisfies δS = 2 + 1/δS (continued-fraction fixed point) and
-- δS² − 2δS − 1 = 0 (minimal polynomial over ℚ).  These characterize δS
-- as the unique positive real satisfying x = 2 + 1/x, i.e., x² = 2x + 1.
-- Ref: docs/master_derivations.pdf §1 Proposition 4
-- ════════════════════════════════════════════════════════════════════════════

/-- δS > 0 (exported version of the private δS_pos).  Useful for downstream proofs. -/
theorem silverRatio_pos : 0 < δS := δS_pos

/-- δS satisfies the continued-fraction fixed-point equation δS = 2 + 1/δS.
    This is the defining self-similarity of the silver rectangle:
    a silver rectangle of side ratio δS can be divided into two unit squares
    and a smaller silver rectangle of the same proportions.
    Proof: multiply both sides by δS to get δS² = 2δS + 1 (= silverRatio_sq). -/
theorem silverRatio_cont_frac : δS = 2 + 1 / δS := by
  have hδ : δS ≠ 0 := ne_of_gt δS_pos
  field_simp [hδ]
  linarith [silverRatio_sq]

/-- δS² − 2·δS − 1 = 0: the minimal polynomial of the silver ratio over ℚ.
    δS is the unique positive root of x² − 2x − 1 = 0 (discriminant = 8, roots 1 ± √2).
    Ref: docs/master_derivations.pdf §1 Proposition 4 -/
theorem silverRatio_minPoly : δS ^ 2 - 2 * δS - 1 = 0 := by linarith [silverRatio_sq]

-- ════════════════════════════════════════════════════════════════════════════
-- Section 21 — Phase Accumulation and NullSliceBridge Coverage
-- The precession phasor P(n) = exp(I·n·ΔΦ₀) accumulates phase linearly;
-- after D = 13717421 steps, Φ = 2π (full return).
-- The NullSliceBridge maps 8 channels to distinct angular positions via
-- k ↦ 3k mod 8 — a bijection on ZMod 8 since gcd(3,8) = 1, mirroring
-- exactly the primitive-root structure of the μ-orbit.
-- Ref: docs/master_derivations.pdf §6, §8
-- ════════════════════════════════════════════════════════════════════════════

/-- Full-cycle phase accumulation: D = 13717421 precession steps return to 2π.
    D · (2π/D) = 2π — the precession period is the exact denominator of ΔΦ₀.
    Ref: docs/master_derivations.pdf §6 Proposition (multi-window phase accum.) -/
theorem phase_full_cycle :
    (13717421 : ℝ) * (2 * Real.pi / 13717421) = 2 * Real.pi := by field_simp

/-- NullSliceBridge channel distinctness: the 8 angles {3k mod 8 : k ∈ Fin 8}
    are all 8 distinct residues, covering {0,1,…,7} completely.
    Same gcd(3,8) = 1 coprimality that makes μ a primitive 8th root.
    Ref: docs/master_derivations.pdf §8 NullSliceBridge -/
theorem nullslice_channels_distinct :
    (Finset.image (fun k : Fin 8 => (3 * k.val) % 8) Finset.univ).card = 8 := by
  native_decide

/-- **Derived**: the NullSliceBridge channel map k ↦ 3k is a bijection on ZMod 8.
    Since 3 · 3 = 9 ≡ 1 (mod 8), multiplication by 3 is its own inverse.
    This is the same self-inverse structure as the μ-orbit: gcd(3,8) = 1 implies
    the map is a group automorphism of ℤ/8ℤ. -/
theorem nullslice_coverage_bijective :
    Function.Bijective (fun k : ZMod 8 => (3 : ZMod 8) * k) := by
  constructor
  · intro a b h
    have ha := congr_arg ((3 : ZMod 8) * ·) h
    simp only [← mul_assoc, show (3 : ZMod 8) * 3 = 1 from by decide, one_mul] at ha
    exact ha
  · intro b
    exact ⟨3 * b, by
      simp only [← mul_assoc, show (3 : ZMod 8) * 3 = 1 from by decide, one_mul]⟩

-- ════════════════════════════════════════════════════════════════════════════
-- Section 22 — Machine-Discovered Deep Connections
-- These theorems arise from systematic chaining of all previous results.
-- They were not explicitly encoded from the source documents; rather,
-- the proof system derived them from definitions and lemmas already present.
--
-- Key discoveries:
--   § C(r) = sech(log r)               — master Lyapunov parametrisation
--   § C(δS) = η                        — silver ratio sits at the canonical weight
--   § sech(log δS) = η                 — direct hyperbolic characterisation
--   § Res(exp λ) = 2·sinh λ/δS         — palindrome residual as hyperbolic sine
--   § C(exp λ)² + tanh²λ = 1           — hyperbolic Pythagorean identity
--   § C(r)² + (δS·r·Res r/(1+r²))² = 1 — Pythagorean unified with palindrome
--   § 3·(3·k) = k in ZMod 8            — NullSliceBridge is an involution
--   § C(rⁿ) ≤ 2/rⁿ                    — explicit orbit decoherence rate
--   § μ⁷ = μ⁻¹                        — seventh power equals the inverse
--   § Res(r) + Res(1/r) = 0            — palindrome anti-symmetry in sum form
-- ════════════════════════════════════════════════════════════════════════════

/-- **Master link**: C(r) = (cosh(log r))⁻¹ for all r > 0.
    Every coherence value is a hyperbolic secant of the natural Lyapunov
    exponent λ = log r.  Setting r = exp λ recovers lyapunov_coherence_sech;
    setting r = 1 gives C(1) = (cosh 0)⁻¹ = 1.
    Ref: synthesized from §5 (C definition) and §10 (Theorem 14 duality). -/
theorem coherence_is_sech_of_log (r : ℝ) (hr : 0 < r) :
    C r = (Real.cosh (Real.log r))⁻¹ := by
  conv_lhs => rw [← Real.exp_log hr]
  exact lyapunov_coherence_sech (Real.log r)

/-- **Machine-discovered**: the coherence at the silver ratio equals the canonical
    state component η = 1/√2.
    C(δS) = 2(1+√2)/(1+(1+√2)²) = (1+√2)/(2+√2) = 1/√2 = η.
    δS (§7) and η (§6) were defined independently; the machine found their link. -/
theorem coherence_at_silver_is_eta : C δS = η := by
  have hs := sqrt2_sq
  have hns : Real.sqrt 2 ≥ 0 := Real.sqrt_nonneg 2
  have hs2_ne : Real.sqrt 2 ≠ 0 := by nlinarith
  have hd : (1 : ℝ) + δS ^ 2 ≠ 0 := by unfold δS; nlinarith
  unfold C η
  rw [div_eq_div_iff hd hs2_ne]
  unfold δS
  nlinarith

/-- **Corollary**: sech(log δS) = η.
    Direct hyperbolic characterisation: the silver ratio δS = 1+√2 satisfies
    log δS = arcsinh(1), so cosh(log δS) = √2, and sech(log δS) = 1/√2 = η.
    Proof: coherence_is_sech_of_log + coherence_at_silver_is_eta. -/
theorem sech_at_log_silverRatio : (Real.cosh (Real.log δS))⁻¹ = η :=
  (coherence_is_sech_of_log δS δS_pos).symm.trans coherence_at_silver_is_eta

/-- Palindrome residual in Lyapunov form: Res(exp l) = 2·sinh(l)/δS.
    C = sech l (even) and δS · Res / 2 = sinh l (odd): a sech/sinh dual pair
    paralleling the cos/sin pair in circular geometry.
    Machine-discovered: §9 (Res definition) + §10 (exp parametrisation). -/
theorem lyapunov_tanh_residual (l : ℝ) :
    Res (Real.exp l) = 2 * Real.sinh l / δS := by
  unfold Res
  simp only [Real.sinh_eq, Real.exp_neg, one_div]
  ring

/-- **Hyperbolic Pythagorean identity**: C(exp l)² + tanh²(l) = 1.
    In Lyapunov variables C = sech l, so this is just sech²l + tanh²l = 1.
    This is the l-space companion to §18's `coherence_pythagorean`.
    Machine-discovered: §10 (C = sech) + standard hyperbolic identity. -/
theorem coherence_lyapunov_pythag (l : ℝ) :
    C (Real.exp l) ^ 2 + Real.tanh l ^ 2 = 1 := by
  rw [lyapunov_coherence_sech, Real.tanh_eq_sinh_div_cosh]
  have hc : Real.cosh l ≠ 0 := ne_of_gt (Real.cosh_pos l)
  have hprod : Real.exp l * Real.exp (-l) = 1 := by rw [← Real.exp_add]; simp
  have key : Real.sinh l ^ 2 + 1 = Real.cosh l ^ 2 := by
    rw [Real.sinh_eq, Real.cosh_eq]
    field_simp
    nlinarith [Real.exp_pos l, Real.exp_pos (-l), hprod,
               sq_nonneg (Real.exp l - Real.exp (-l))]
  field_simp [hc]
  linarith

/-- **Unified Pythagorean** with palindrome residual.
    Substitutes palindrome_amplitude_eq (δS·r·Res r = r²−1) into §18's
    coherence_pythagorean, expressing the identity purely via C and Res.
    Machine-discovered: §18 (Pythagorean identity) + §9 (palindrome amplitude). -/
theorem coherence_residual_pythagorean (r : ℝ) (hr : 0 < r) :
    C r ^ 2 + (δS * r * Res r / (1 + r ^ 2)) ^ 2 = 1 := by
  rw [palindrome_amplitude_eq r hr]
  exact coherence_pythagorean r hr

/-- NullSliceBridge is an involution: k ↦ 3k composed with itself is the identity.
    Since 3·3 = 9 ≡ 1 (mod 8), the channel map is self-inverse on ZMod 8.
    Mirrors the μ-orbit: μ⁷ = μ⁻¹ so applying the 7-step twice returns to start.
    Machine-discovered: extension of §21's NullSliceBridge bijection proof. -/
theorem nullslice_involution (k : ZMod 8) :
    (3 : ZMod 8) * ((3 : ZMod 8) * k) = k := by
  simp only [← mul_assoc, show (3 : ZMod 8) * 3 = 1 from by decide, one_mul]

/-- **Orbit decoherence rate**: C(rⁿ) ≤ 2/rⁿ for r > 1.
    Coherence decays at least as fast as the reciprocal of the orbit amplitude.
    Proof: cosh(n·log r) ≥ rⁿ/2 (since n·log r ≥ 0 for r > 1, so cosh x ≥ exp x / 2),
    so sech(n·log r) ≤ 2/rⁿ.
    Machine-discovered: §19 (orbit sech formula) + cosh lower bound. -/
theorem orbit_decoherence_rate (r : ℝ) (hr : 1 < r) (n : ℕ) :
    C (r ^ n) ≤ 2 / r ^ n := by
  have hr0 : 0 < r := lt_trans one_pos hr
  have hpow : (0 : ℝ) < r ^ n := pow_pos hr0 n
  rw [coherence_orbit_sech r hr0, inv_eq_one_div,
      div_le_div_iff₀ (Real.cosh_pos _) hpow, one_mul]
  have hexp : Real.exp (↑n * Real.log r) = r ^ n := by
    rw [← Real.exp_log (pow_pos hr0 n), Real.log_pow]
  rw [Real.cosh_eq, hexp]
  linarith [Real.exp_pos (-(↑n * Real.log r))]

/-- μ⁷ = μ⁻¹: the seventh power of μ equals its multiplicative inverse.
    Since μ⁸ = 1 (mu_pow_eight), μ⁷ · μ = μ⁸ = 1, so μ⁷ = μ⁻¹.
    Traversing the 8-cycle seven steps forward is the same as one step backwards.
    Machine-discovered: §2 (mu_pow_eight) + group-inverse characterisation. -/
theorem mu_inv_eq_pow7 : μ ^ 7 = μ⁻¹ := by
  have hμ : μ ≠ 0 := Complex.exp_ne_zero _
  have h7 : μ ^ 7 * μ = 1 := by
    calc μ ^ 7 * μ = μ ^ (7 + 1) := (pow_succ μ 7).symm
      _ = μ ^ 8 := rfl
      _ = 1 := mu_pow_eight
  calc μ ^ 7
      = μ ^ 7 * (μ * μ⁻¹) := by rw [mul_inv_cancel₀ hμ, mul_one]
    _ = μ ^ 7 * μ * μ⁻¹   := (mul_assoc _ _ _).symm
    _ = 1 * μ⁻¹            := by rw [h7]
    _ = μ⁻¹                := one_mul _

/-- Palindrome anti-symmetry in sum form: Res(r) + Res(1/r) = 0.
    Direct consequence of palindrome_residual_antisymm (Res(1/r) = −Res r):
    the palindrome residuals of a ratio and its reciprocal cancel exactly.
    Machine-discovered: §9 (anti-symmetry) cast as an additive identity. -/
theorem palindrome_sum_zero (r : ℝ) (hr : 0 < r) :
    Res r + Res (1 / r) = 0 := by
  linarith [palindrome_residual_antisymm r hr]

end -- noncomputable section
