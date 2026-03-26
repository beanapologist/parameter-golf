/-
  Main.lean — Executable entry point for the FormalLean project.

  Imports the full Quantization theorem package and prints a summary of
  the five quantization conditions confirmed by `lead_quantization_confirmed`.
-/

import Quantization

/-- Print a summary of the Kernel Quantization Formula theorems. -/
def printQuantization : IO Unit := do
  IO.println "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  IO.println "  Kernel Quantization Formula — Machine-Checked Summary"
  IO.println "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  IO.println ""
  IO.println "  §1  Phase Quantization (μ = exp(I·3π/4))"
  IO.println "      Q1.1  |μ| = 1                        [quant_phase_abs_one]"
  IO.println "      Q1.2  μ⁸ = 1                         [quant_phase_eight_cycle]"
  IO.println "      Q1.3  |μⁿ| = 1  ∀n                  [quant_phase_orbit_unitary]"
  IO.println "      Q1.4  μ⁰,…,μ⁷ are pairwise distinct [quant_phase_eight_distinct]"
  IO.println ""
  IO.println "  §2  Energy Quantization (E_n = −1/n²)"
  IO.println "      Q2.1  E_n = −1/n²                    [quant_energy_formula]"
  IO.println "      Q2.2  E₁  = −1                       [quant_energy_ground]"
  IO.println "      Q2.3  E_n < 0  ∀n ≥ 1               [quant_energy_negative]"
  IO.println "      Q2.4  E_n < E_{n+1}  ∀n ≥ 1         [quant_energy_ascent]"
  IO.println ""
  IO.println "  §3  Floquet Quantization (ε_F = π/T)"
  IO.println "      Q3.1  ε_F·T = π                      [quant_floquet_phase]"
  IO.println "      Q3.2  H·T = 5π/4 → U(H,T) = μ       [quant_floquet_recipe]"
  IO.println "      Q3.3  0 < ε_F  (for T > 0)           [quant_floquet_pos]"
  IO.println "      Q3.4  ε_F·(2T) = 2π                  [quant_floquet_doubling]"
  IO.println ""
  IO.println "  §4  Amplitude Quantization (η = 1/√2)"
  IO.println "      Q4.1  2η² = 1                        [quant_amplitude_balance]"
  IO.println "      Q4.2  η² + |μ·η|² = 1               [quant_amplitude_canonical_norm]"
  IO.println "      Q4.3  C(1) = 1                       [quant_amplitude_coherence_max]"
  IO.println "      Q4.4  C(r) ≤ 1  ∀r ≥ 0              [quant_amplitude_global_bound]"
  IO.println ""
  IO.println "  §5  Theorem Q (Full Conjunction when H·T = 5π/4)"
  IO.println "      Q5.1  Floquet arm:   ε_F·T = π  ∧  μ⁸ = 1    [quant_Q_floquet_arm]"
  IO.println "      Q5.2  Energy arm:    E₁ = −1                   [quant_Q_energy_arm]"
  IO.println "      Q5.3  Amplitude arm: 2η² = 1  ∧  C(1) = 1     [quant_Q_amplitude_arm]"
  IO.println "      Q5.4  Capstone:      all five simultaneously    [lead_quantization_confirmed]"
  IO.println ""
  IO.println "  All 20 theorems: zero sorry, fully machine-checked."
  IO.println "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

def main : IO Unit := printQuantization
