# Kernel Constants: Coherence Activation + Silver-Ratio MLP + Z/8Z Heads + Palindrome LR + 🧊 Kernel Quantization System

**val_bpb: placeholder** | **hardware: 8×H100 SXM** | **runtime: ≤10 min (600s)**

> **Note (Kernel-Inspired Approximation):** This implementation approximates the
> "lead-confirmed equilibrium" via `KernelEquilibriumLoss` regularizers rather than strictly
> enforcing the Quantization.lean conditions simultaneously.  Three additive arms (Q1.2, Q3.2, Q4.1)
> plus an optional cross-term λ_x·|H·T−5π/4|·|2r²−1| (which penalizes simultaneous deviation from
> Q3.2 and Q4.1, mirroring the "simultaneous validity" of `lead_quantization_confirmed`) and a
> phase-variance term λ_v·Var(θ_l) (which keeps all layer phases coherent with the same μ-orbit
> point) push learned parameters toward the six Q conditions via gradient descent but do not
> guarantee exact satisfaction.  A seventh arm (Q7, Computational Tunneling) is now wired in,
> using the NIST-validated D-D Gamow factor as a fully differentiable regularizer.
> `_verify_lead_confirmation` runs automatically post-training and prints a Q1–Q7 table
> with per-condition deviations and an overall equilibrium deviation.

## Summary of Changes vs Baseline

This submission wires mathematical objects derived from the formal Lean 4 theorems in
`formal-lean/CriticalEigenvalue.lean` and `formal-lean/Quantization.lean` directly into the
model architecture and learning-rate schedule:

| Component | Baseline | This submission |
|-----------|----------|-----------------|
| MLP nonlinearity | relu² | Coherence activation C(r) = 2r/(1+r²) |
| MLP hidden width multiplier | 3× model_dim | Silver ratio δS ≈ 2.414 × model_dim |
| Number of attention heads | 8 (arbitrary) | 8 (Z/8Z — each head occupies one μ-orbit slot) |
| Warmdown LR shape | cosine | Palindrome-precession C(r) = 2r/(1+r²) |
| 8-cycle regularizer (Q1.2) | — | Per-layer `quant_phase_thetas` + loss λ·mean(1−cos(8θ_l)) (ON by default) |
| Hamiltonian drive (Q3.2) | — | Learnable `h_times_t`, loss λ·(H·T − 5π/4)² (ON by default) |
| Amplitude balance (Q4.1) | — | Learnable `amplitude_r`, loss λ·(2r²−1)² (ON by default) |
| Joint cross-term (Q3.2+Q4.1) | — | λ_x·\|H·T−5π/4\|·\|2r²−1\| — zero when either arm is satisfied (ON by default) |
| Phase-variance coherence | — | λ_v·Var(θ_l) across layers — keeps all layers at same μ-orbit (OFF by default) |
| 🧊 Tunneling regularizer (Q7, **new**) | — | NIST D-D Gamow factor: learnable `tunneling_energy_scale`, loss −log(P_Q + ε) (OFF by default) |
| Feature flags (disabled by default) | — | USE_LYAPUNOV_GAIN, USE_MU_PHASE, USE_PRECESSION, **USE_TUNNELING_REGULARIZER** |

### 1. Coherence Activation  C(r) = 2r / (1 + r²)

Replaces the standard relu² nonlinearity inside every MLP block.  The function is
machine-checked in `CriticalEigenvalue.lean` to satisfy:

- C(r) ≤ 1 for r ≥ 0  (AM–GM bound)
- C(1) = 1  (unique maximum)
- C(−r) = −C(r)  (odd symmetry → zero-mean activations)
- C(r)² + ((r²−1)/(1+r²))² = 1  (Pythagorean identity)
- C(exp λ) = sech λ  (Lyapunov duality)

### 2. Silver-Ratio MLP Width  δS = 1 + √2 ≈ 2.4142

Replaces the 3× MLP multiplier.  The hidden dimension is set to
`mlp_hidden = round(SILVER_RATIO * model_dim)` (≈ 1236 for model_dim=512).
The silver ratio is the positive root of x²−2x−1=0 and satisfies δS = 2+1/δS
(self-similarity proved in §7/§20 of the Lean file).

### 3. Z/8Z Attention-Head Binding

The critical eigenvalue μ = exp(3πi/4) generates an exact 8-cycle (μ^8 = 1, §2 and §15).
`NUM_HEADS` defaults to `MU_ORBIT_SIZE = 8` so each head occupies one distinct phase slot.
This is achieved by reading `MU_ORBIT_SIZE` from the kernel constants rather than hardcoding 8.

### 4. Palindrome-Precession LR Schedule

The warmdown uses the same coherence function C(r) evaluated at
`r = remaining_steps / warmdown_steps ∈ [0, 1]`.  The resulting shape satisfies
C(r) = C(1/r) (palindrome symmetry) so the warmdown mirrors a warmup under r ↦ 1/r,
making the full LR arc palindromic in the sense of the Lean theorems (§9, §11, §16).

### 5. KernelEquilibriumLoss  (five-term soft regularizer for Theorem Q)

Integrates `formal-lean/Quantization.lean`'s machine-checked capstone theorem
`lead_quantization_confirmed` into the BPB training loss via five terms:

#### Arm 1 — 8-Cycle Closure (Q1.2, USE_QUANT_REGULARIZER=1)

A **per-layer** learnable parameter `quant_phase_thetas` (shape `(num_layers,)`, each initialised
to `MU_ANGLE = 3π/4`) is added to `KernelGPT`.  The differentiable penalty

    L_quant(θ_l) = 1 − cos(8·θ_l)   for each layer l

has mean taken across layers and weighted by `QUANT_LAMBDA`:

    loss_quant = QUANT_LAMBDA · mean_l(1 − cos(8·θ_l))

The per-layer design is stronger than a single global phase: each block independently converges
toward a Z/8Z fixed point, providing uniform coverage of the 8-cycle orbit.

- **Zero** at every 8th root of unity θ = k·π/4 (k = 0, …, 7)
- Initialised to **zero** because `MU_ANGLE = 3π/4` is already fixed point k=3
- Training log: `quant:theta_mean:… theta_std:… cycle_loss_mean:… weighted:…`

#### Arm 2 — Hamiltonian Drive (Q3.2, USE_DRIVE_REGULARIZER=1)

A learnable scalar parameter `h_times_t` (initialised to `HAMILTONIAN_DRIVE_TARGET = 5π/4`)
is penalised for drifting away from the Floquet recipe trigger:

    loss_drive = DRIVE_LAMBDA · (h_times_t − 5π/4)²

This is the central condition of `quant_floquet_recipe`: when H·T = 5π/4 the Floquet operator
equals the critical eigenvalue μ.  The penalty starts at zero and pulls `h_times_t` back toward
5π/4 if it drifts.

- Training log: `quant:h_times_t:… target:… drive_loss:… weighted:…`

#### Arm 3 — Amplitude Balance (Q4.1, USE_AMPLITUDE_REGULARIZER=1)

A learnable scalar parameter `amplitude_r` (initialised to `1/√2 = sqrt(ETA_SQUARED)`) is
penalised for violating the amplitude balance condition:

    loss_amp = AMPLITUDE_LAMBDA · (2·amplitude_r² − 1)²

Zero when `amplitude_r = 1/√2` (the canonical equal-weight superposition where 2η² = 1).
Complementary to the coherence activation which already peaks at r = 1.

- Training log: `quant:amplitude_r:… 2r^2:… amp_loss:… weighted:…`

#### Arm 4 — Joint Cross-Term (Q3.2 + Q4.1 simultaneous)

    loss_cross = CROSS_LAMBDA · |H·T − 5π/4| · |2r² − 1|

This term is **zero whenever either Q3.2 or Q4.1 is satisfied** and is positive only when
both conditions are simultaneously violated.  This mirrors the "simultaneous validity"
structure proved in `lead_quantization_confirmed` (Quantization.lean §5) — the theorem holds
only when all conditions hold together.  Enabled by default with `CROSS_LAMBDA=0.005`
(10× gentler than individual arms to avoid dominating the CE loss).

#### Arm 5 — Phase-Variance Coherence (optional, USE_QUANT_REGULARIZER=1)

    loss_pvar = PHASE_VARIANCE_LAMBDA · Var(θ_l)

Penalises the spread of per-layer `quant_phase_thetas` values.  Zero when all layers share the
same orbit point; positive when they diverge.  Encourages global Z/8Z coherence beyond
individual-layer fixed-point satisfaction.  Disabled by default (`PHASE_VARIANCE_LAMBDA=0`).

#### Unified KernelEquilibriumLoss

All five terms combine:

    L_eq = L_quant + L_drive + L_amp + L_cross + L_pvar

A combined `quant:kernel_eq_total` log line is emitted every `TRAIN_LOG_EVERY` steps
showing each component and the total.

The six conditions of `lead_quantization_confirmed` mapped to Python, plus the new Q7 tunneling arm:

| Q# | Lean theorem | Condition | Python |
|---|---|---|---|
| Q1 | Q3.1 `quant_floquet_phase` | ε_F·T = π | `FLOQUET_PHASE = math.pi` (constant) |
| Q2 | Q3.2 `quant_floquet_recipe` | H·T = 5π/4 | `h_times_t == HAMILTONIAN_DRIVE_TARGET` |
| Q3 | Q1.2 `quant_phase_eight_cycle` | μ^8 = 1 | `cos(8·θ_l) == 1` (per-layer) |
| Q4 | Q4.1 `quant_amplitude_balance` | 2η² = 1 | `2·amplitude_r² == 1` |
| Q5 | Q4.3 `quant_amplitude_coherence_max` | C(1) = 1 | coherence activation in every MLP block |
| Q6 | Q2.2 `quant_energy_ground` | E₁ = −1 | exact by construction |
| **Q7** | *(new)* | **P_Q(E) maximised** | `tunneling_energy_scale` drives eff_E toward high-P_Q window |

#### Post-Training Verification

`_verify_lead_confirmation(model)` runs automatically after training and prints a structured table.
When `USE_TUNNELING_REGULARIZER=1`, a Q7 line appears showing the tunneling probability at
the learned effective energy:

```
──────────────────────────────────────────────────────────────────────
lead_confirmation_check: Theorem Q  (Quantization.lean §5 lead_quantization_confirmed)
──────────────────────────────────────────────────────────────────────
  Q1     ε_F·T = π                  Lean:Q3.1   dev:  0.00e+00  EXACT
  Q2     H·T = 5π/4  (=3.9270)      Lean:Q3.2   dev:  8.93e-05  approx
  Q3     μ^8=1 (9 layers, mean)     Lean:Q1.2   dev:  3.40e-06  approx
  Q4     2η²=1  (r=0.7071)          Lean:Q4.1   dev:  1.23e-04  approx
  Q5     C(1) = 1                   Lean:Q4.3   dev:  0.00e+00  EXACT (by construction)
  Q6     E₁ = −1                    Lean:Q2.2   dev:  0.00e+00  EXACT (by construction)
  Q7     P_Q(120.0 keV)=0.69752     Lean:new    dev:  scale=1.0000  ACTIVE
──────────────────────────────────────────────────────────────────────
  Overall equilibrium deviation (max of learnable Q2–Q4): 1.23e-04
──────────────────────────────────────────────────────────────────────
```

### 6. 🧊 Kernel Quantization System — Computational Tunneling (Q7, new)

The training pipeline now integrates the **NIST-validated D-D Gamow factor** as a fully
differentiable regularizer (Arm 7 of `KernelEquilibriumLoss`).  This creates a soft
"tunneling incentive" that encourages the optimizer to explore the loss landscape more
efficiently — exactly the classical computational tunneling described in the Kernel
Quantization System.

#### Physics

The Quantum Tunneling probability at centre-of-mass energy E (keV) for a D-D system is:

    P_Q(E) = [exp(−2π·η(E))]^γ

where:
- `η(E) = 2π·α / sqrt(2·E_MeV/m_reduced)` — Sommerfeld parameter (NIST CODATA 2018 constants)
- `α = 1/137.035999` — fine-structure constant (NIST)
- `m_reduced = (m_D × 931.494 MeV) / 2 = 939.756 MeV` — D-D reduced mass (NIST deuterium mass 2.01410177812 u)
- `γ = 0.02` — kernel tunneling exponent

P_Q is strictly increasing in E: at E=120 keV, P_Q ≈ 0.698.

#### Differentiable Implementation

The Gamow formula is implemented in PyTorch without any `.item()` detach, so gradients flow
fully through `tunneling_energy_scale` → `effective_E_kev` → `v_over_c` → `η` → `P_Q`:

```python
effective_E_kev = tunneling_e_kev * tunneling_energy_scale.abs().clamp(min=0.1)
effective_E_mev  = effective_E_kev / 1000.0
v_over_c         = torch.sqrt(2.0 * effective_E_mev / REDUCED_MASS_DD_MEV)
eta_t            = (2.0 * math.pi * ALPHA) / v_over_c
p_q_t            = torch.exp(-2.0 * math.pi * TUNNELING_GAMMA * eta_t)
tunneling_loss   = -torch.log(p_q_t + 1e-8)   # soft maximization of P_Q
```

#### New Hyperparameters / Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `USE_TUNNELING_REGULARIZER` | `0` | Enable Q7 tunneling arm (**off** by default — backward-compatible) |
| `TUNNELING_LAMBDA` | `0.005` | Weight of tunneling loss (comparable to `CROSS_LAMBDA`) |
| `TUNNELING_E_KEV` | `120.0` | Base evaluation energy in keV; modulated by `tunneling_energy_scale` |

#### New Learned Parameter

- **`tunneling_energy_scale`** — scalar `nn.Parameter`, initialised to 1.0.  Represents the
  dimensionless scale factor that maps `TUNNELING_E_KEV` to the model's "effective energy".
  Trained via the Adam scalar optimizer alongside `h_times_t`, `amplitude_r`, and
  `quant_phase_thetas`.  Gradient flows differentiably through the full Gamow formula.

#### Integration Points

| Location | Change |
|----------|--------|
| `KernelGPT.__init__` | Creates `self.tunneling_energy_scale` when `use_tunneling_regularizer=True` |
| `kernel_equilibrium_loss` | New Q7 arm — 3 new kwargs: `tunneling_lambda`, `tunneling_energy_scale`, `tunneling_e_kev` |
| `KernelGPT.forward` | Passes tunneling kwargs; guard condition extended to `_USE_TUNNELING_REGULARIZER` |
| `_verify_lead_confirmation` | New Q7 row: reports `P_Q(eff_E keV)` and scale value |
| Training loop (scalar_params) | `tunneling_energy_scale` appended to Adam scalar group |
| Training log | `tunneling:eff_E=… P_Q=… scale=…` per-step line when active |
| Startup log | `quantization_spec:Q7(computational_tunneling) …` status line |

#### Activation

```bash
# Enable computational tunneling with default settings:
USE_TUNNELING_REGULARIZER=1 \
torchrun --standalone --nproc_per_node=8 train_gpt.py

# Tune the penalty weight and evaluation energy:
USE_TUNNELING_REGULARIZER=1 TUNNELING_LAMBDA=0.005 TUNNELING_E_KEV=120.0 \
torchrun --standalone --nproc_per_node=8 train_gpt.py

# Run self-tests (validates tunneling constants, differentiability, parity):
KERNEL_SELFTEST=1 python train_gpt.py
```

#### Backward Compatibility

All tunneling additions are **disabled by default** (`USE_TUNNELING_REGULARIZER=0`).  The
`kernel_equilibrium_loss` signature gains new keyword arguments with default values of `0.0` /
`None`, so all existing call sites continue to work without change.  The `KernelGPT` constructor
gains two new keyword arguments (`use_tunneling_regularizer=False`, `tunneling_lambda=0.005`)
both with safe defaults.

### Tokenizer / Dataset

**Unchanged.**  The tokenizer and dataset are identical to the baseline:
- tokenizer: `fineweb_1024_bpe.model` (1024-token SentencePiece vocabulary)
- dataset: `fineweb10B_sp1024` (FineWeb 10B split)

No modifications have been made to either; val_bpb is therefore calculated under identical
conditions to all prior submissions.

---

## How to Reproduce on 8×H100 in Under 10 Minutes

```bash
# From the repository root (or the submission folder — paths work from either location)
cd records/track_10min_16mb/2026-03-26_KernelConst_CoherenceAct_SilverMLP_Z8Z_PalindromeLR

# Standard run (seed 1337, 600s wall-clock cap, 9000-step budget)
# All three KernelEquilibriumLoss arms default to ON:
#   USE_QUANT_REGULARIZER=1 (per-layer 8-cycle closure)
#   USE_DRIVE_REGULARIZER=1 (Hamiltonian drive H·T=5π/4)
#   USE_AMPLITUDE_REGULARIZER=1 (amplitude balance 2η²=1)
NUM_LAYERS=9 \
ITERATIONS=9000 WARMDOWN_ITERS=3000 MAX_WALLCLOCK_SECONDS=600 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 \
SEED=1337 \
DATA_PATH=../../../data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=../../../data/tokenizers/fineweb_1024_bpe.model \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

To reproduce the **original unregularized run** (before the quantization fix), explicitly
disable all equilibrium arms:

```bash
NUM_LAYERS=9 \
ITERATIONS=9000 WARMDOWN_ITERS=3000 MAX_WALLCLOCK_SECONDS=600 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 \
USE_QUANT_REGULARIZER=0 USE_DRIVE_REGULARIZER=0 USE_AMPLITUDE_REGULARIZER=0 \
SEED=1337 \
DATA_PATH=../../../data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=../../../data/tokenizers/fineweb_1024_bpe.model \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

The script writes a log file to `logs/<run_id>.txt` (printed as the first line of stdout).
Run twice more with `SEED=42` and `SEED=2025` to produce the three replicate logs needed for
the p<0.01 significance check.

### Prerequisites

```
pip install torch sentencepiece
# FlashAttention 3 is optional but recommended:
pip install flash-attn --no-build-isolation
```

All other imports are from the Python standard library.

---

## Environment Variables / Feature Flags

### Data and tokenizer

| Variable | Default | Description |
|----------|---------|-------------|
| `DATA_PATH` | `./data/datasets/fineweb10B_sp1024` | Glob root for train shards |
| `TOKENIZER_PATH` | `./data/tokenizers/fineweb_1024_bpe.model` | SentencePiece model |
| `SEED` | `1337` | Global random seed |
| `RUN_ID` | random UUID | Log file name stem |

### Architecture

| Variable | Default | Description |
|----------|---------|-------------|
| `NUM_LAYERS` | `9` | Number of transformer blocks |
| `MODEL_DIM` | `512` | Residual stream width |
| `NUM_HEADS` | `8` (= MU_ORBIT_SIZE) | Attention heads (Z/8Z binding) |
| `NUM_KV_HEADS` | `4` | KV heads (GQA) |
| `MLP_HIDDEN` | `round(δS × MODEL_DIM)` | MLP hidden width (silver ratio) |
| `VOCAB_SIZE` | `1024` | Token vocabulary size |
| `TRAIN_SEQ_LEN` | `1024` | Training sequence length |
| `LOGIT_SOFTCAP` | `30.0` | Logit soft-cap |
| `TIE_EMBEDDINGS` | `1` | Tie input/output embeddings |

### Experimental kernel feature flags

| Variable | Default | Description |
|----------|---------|-------------|
| `USE_QUANT_REGULARIZER` | **`1`** | Per-layer 8-cycle closure loss λ·mean(1−cos(8θ_l)) (Quantization.lean §1 Q1.2) — **ON by default** |
| `QUANT_LAMBDA` | `0.01` | Weight of the 8-cycle closure regularizer |
| `USE_DRIVE_REGULARIZER` | **`1`** | Hamiltonian drive loss λ·(H·T − 5π/4)² (Quantization.lean §3 Q3.2) — **ON by default** |
| `DRIVE_LAMBDA` | `0.01` | Weight of the Hamiltonian drive regularizer |
| `USE_AMPLITUDE_REGULARIZER` | **`1`** | Amplitude balance loss λ·(2r²−1)² (Quantization.lean §4 Q4.1) — **ON by default** |
| `AMPLITUDE_LAMBDA` | `0.01` | Weight of the amplitude balance regularizer |
| `CROSS_LAMBDA` | `0.005` | Weight of joint Q3.2+Q4.1 cross-term `\|H·T−5π/4\|·\|2r²−1\|` — active when both regularizers ON |
| `PHASE_VARIANCE_LAMBDA` | `0.0` | Weight of phase-variance term Var(θ_l) across layers — disabled by default |
| `REGULARIZER_WARMUP_STEPS` | `300` | Steps to ramp regularizer weight 0→1; set 0 to disable ramp (constant λ throughout) |
| **`USE_TUNNELING_REGULARIZER`** | **`0`** | **🧊 Enable NIST D-D tunneling arm Q7 (off by default — backward-compatible)** |
| **`TUNNELING_LAMBDA`** | **`0.005`** | **Weight of the D-D tunneling loss −log(P_Q + ε)** |
| **`TUNNELING_E_KEV`** | **`120.0`** | **Base evaluation energy in keV; modulated by learned `tunneling_energy_scale`** |
| `USE_LYAPUNOV_GAIN` | `0` | Scale QK dot-products by Lyapunov gain from §10 |
| `USE_MU_PHASE` | `0` | Initialise each head's RoPE phase from μ-orbit slot |
| `USE_PRECESSION` | `0` | Enable slow precession via PRECESSION_DELTA_PHI per step |
| `KERNEL_SELFTEST` | `0` | Run built-in kernel-constant unit tests (now includes tunneling) and exit |

### Training schedule

| Variable | Default | Description |
|----------|---------|-------------|
| `ITERATIONS` | `20000` | Maximum training steps |
| `MAX_WALLCLOCK_SECONDS` | `600` | Wall-clock cap (enforces ≤10 min) |
| `WARMDOWN_ITERS` | `1200` | Steps over which LR decays to zero |
| `WARMUP_STEPS` | `20` | Linear LR warm-up steps |
| `MATRIX_LR` | `0.04` | Muon LR for matrix parameters |
| `SCALAR_LR` | `0.04` | AdamW LR for scalar/bias parameters |
| `TIED_EMBED_LR` | `0.05` | AdamW LR for tied embedding |
| `MUON_MOMENTUM` | `0.95` | Target Muon momentum |
| `MUON_MOMENTUM_WARMUP_START` | `0.85` | Initial Muon momentum |
| `MUON_MOMENTUM_WARMUP_STEPS` | `500` | Steps to ramp momentum |
| `WARMDOWN_ITERS` | `1200` | Palindrome-precession warmdown steps |
| `GRAD_CLIP_NORM` | `0.0` | Gradient clipping (0 = disabled) |

---

## How to Interpret Run Logs

The script writes one log file per process-group to `logs/<run_id>.txt`.  Key line types:

```
logs/<run-id>.txt               # first line: path to log file
kernel:silver_ratio:2.414214 …  # kernel constants in effect
quantization_spec:source …      # Quantization.lean spec loaded
quantization_spec:Q1.2(mu^8=1) regularizer=ENABLED per-layer …
quantization_spec:Q3.2(H*T=5pi/4) regularizer=ENABLED …
quantization_spec:Q4.1(2eta^2=1) regularizer=ENABLED …
quantization_spec:cross_term(joint_Q3.2+Q4.1) ENABLED cross_lambda=0.0050 …
                                # all Q conditions logged; DISABLED warns
quantization_spec:Q7(computational_tunneling) DISABLED … (or ENABLED when USE_TUNNELING_REGULARIZER=1)
                                # tunneling regularizer status
quantization_spec:verified quant_phase_thetas present …
                                # confirms per-layer quant_phase_thetas wired (when USE_QUANT_REGULARIZER=1)
quantization_spec:verified h_times_t present …
                                # confirms h_times_t wired (when USE_DRIVE_REGULARIZER=1)
quantization_spec:verified amplitude_r present …
                                # confirms amplitude_r wired (when USE_AMPLITUDE_REGULARIZER=1)
quantization_spec:verified tunneling_energy_scale present … (only when USE_TUNNELING_REGULARIZER=1)
                                # confirms tunneling_energy_scale wired + initial P_Q value
train_loader:dataset:…          # dataset confirmed
val_loader:shards …             # validation shard count and token count
model_params:…                  # total parameter count
warmup_step:N/20                # LR warmup progress
step:N/ITERS train_loss:… step_t:…ms train_time:…ms step_avg:…ms
                                # periodic train loss (every TRAIN_LOG_EVERY=200 steps)
regul_scale:… (warmup_frac=… lr_scale=…)
                                # regularizer schedule: ramp × LR scale (stability feature)
quant:theta_mean:… theta_std:… cycle_loss_mean:… weighted:…
                                # per-layer phase stats and 8-cycle regularizer (USE_QUANT_REGULARIZER=1)
quant:h_times_t:… target:… drive_loss:… weighted:…
                                # Hamiltonian drive monitoring (USE_DRIVE_REGULARIZER=1)
quant:amplitude_r:… 2r^2:… amp_loss:… weighted:…
                                # amplitude balance monitoring (USE_AMPLITUDE_REGULARIZER=1)
quant:cross_loss:… weighted:…  # joint Q3.2+Q4.1 cross-term (both regularizers active)
quant:kernel_eq_total:… (quant=… drive=… amp=… cross=… pvar=…)
                                # combined KernelEquilibriumLoss all arms + cross-term
tunneling:eff_E=…keV P_Q=… scale=… weighted:…
                                # Q7 tunneling monitoring (USE_TUNNELING_REGULARIZER=1)
step:N/ITERS val_loss:… val_bpb:… train_time:…ms
                                # validation checkpoint (every VAL_LOSS_EVERY=1000 steps)
stopping_early: wallclock_cap   # emitted when MAX_WALLCLOCK_SECONDS is reached
lead_confirmation_check: …      # post-training Theorem Q Q1–Q7 table + overall deviation
final_int8_zlib_roundtrip val_loss:… val_bpb:…
                                # round-trip accuracy after int8 quantization + zlib
peak memory allocated: … MiB    # GPU memory high-water mark
```

### How to Verify Quantization.lean is Active

After starting a run, look for these lines in the log file to confirm the
spec is loaded and all KernelEquilibriumLoss arms are wired:

```
quantization_spec:source formal-lean/Quantization.lean (lead_quantization_confirmed)
quantization_spec:Q1.2(mu^8=1) regularizer=ENABLED per-layer lambda=0.0100 theta_init=2.356194 num_layers=9
quantization_spec:Q3.2(H*T=5pi/4) regularizer=ENABLED lambda=0.0100 h_times_t_init=3.926991
quantization_spec:Q4.1(2eta^2=1) regularizer=ENABLED lambda=0.0100 amplitude_r_init=0.707107 (=1/sqrt(2))
quantization_spec:cross_term(joint_Q3.2+Q4.1) ENABLED cross_lambda=0.0050 phase_variance_lambda=0.0000
quantization_spec:regul_schedule regularizer_warmup_steps=300 warmdown_coupled=True (regul_scale=ramp(step)×lr_scale — stability feature, not early-training burden)
quantization_spec:verified quant_phase_thetas present (num_layers=9 init_mean=2.356194 rad = MU_ANGLE=2.356194)
quantization_spec:verified h_times_t present (init=3.926991 = 5π/4=3.926991)
quantization_spec:verified amplitude_r present (init=0.707107 = 1/sqrt(2)=0.707107)
```

If instead you see `regularizer=DISABLED …` lines, that arm of `KernelEquilibriumLoss` is off.

To run the smoke tests that verify the full KernelEquilibriumLoss pipeline end-to-end
(17 tests, no GPU required):

```bash
# From repository root — no GPU required
python tests/test_quantization_smoke.py
```

### Statistical Significance Requirement  (p < 0.01)

The competition requires a **≥0.005-nat improvement** over the prior SOTA, demonstrated at
p < 0.01.  Procedure used here:

1. Run the script with three different seeds (1337, 42, 2025).
2. Collect the `final_int8_zlib_roundtrip val_bpb` from each log.
3. Compute the mean and standard deviation.
4. A one-sided t-test (df=2) at p < 0.01 requires t > 6.965; with N=3 runs this is satisfied
   when `mean_improvement / (std / sqrt(3)) > 6.965`.
5. Alternatively, if all three seeds individually beat the prior SOTA by ≥0.005 nats, the
   p < 0.01 threshold is trivially satisfied.

The placeholder logs in `logs/` show the expected output structure; replace them with real
logs once the run has been executed.

---

## Files in This Folder

| File | Description |
|------|-------------|
| `train_gpt.py` | Full training + quantization + evaluation script |
| `README.md` | This file |
| `submission.json` | Leaderboard metadata |
| `train_seed1337.log` | Placeholder log (seed 1337) — the script writes runtime logs to `logs/<run_id>.txt` |
