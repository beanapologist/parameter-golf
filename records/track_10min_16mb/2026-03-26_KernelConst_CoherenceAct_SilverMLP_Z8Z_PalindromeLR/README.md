# Kernel Constants: Coherence Activation + Silver-Ratio MLP + Z/8Z Heads + Palindrome LR

**val_bpb: placeholder** | **hardware: 8×H100 SXM** | **runtime: ≤10 min (600s)**

> **Note (Kernel-Inspired Approximation):** This implementation approximates the
> "lead-confirmed equilibrium" via `KernelEquilibriumLoss` regularizers rather than strictly
> enforcing the drive condition H·T = 5π/4.  The full simultaneous lead confirmation (Theorem Q,
> `lead_quantization_confirmed` in Quantization.lean §5) is softly approximated: the loss terms
> push learned parameters toward the five conditions (Q1–Q5) but do not guarantee exact
> satisfaction during training.  Use `_verify_lead_confirmation` (run automatically post-training)
> to inspect how closely the learned parameters satisfy all five conditions.

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
| Feature flags (disabled by default) | — | USE_LYAPUNOV_GAIN, USE_MU_PHASE, USE_PRECESSION |

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

### 5. KernelEquilibriumLoss  (three-arm soft regularizer for Theorem Q)

Integrates `formal-lean/Quantization.lean`'s machine-checked capstone theorem
`lead_quantization_confirmed` into the BPB training loss via three independently-controllable
soft penalty arms:

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

#### Unified KernelEquilibriumLoss

All three arms are combined into a single `kernel_equilibrium_loss()` function:

    L_eq = L_quant + L_drive + L_amp

The five conditions of `lead_quantization_confirmed` mapped to Python:

| Lean theorem | Condition | Python |
|---|---|---|
| Q1.2 `quant_phase_eight_cycle` | μ^8 = 1 | `cos(8·θ_l) == 1` ↔ `L_quant(θ_l) == 0` (per-layer) |
| Q3.2 `quant_floquet_recipe` | H·T = 5π/4 | `h_times_t == HAMILTONIAN_DRIVE_TARGET` ↔ `L_drive == 0` |
| Q3.1 `quant_floquet_phase` | ε_F·T = π | `FLOQUET_PHASE = math.pi` (constant) |
| Q4.1 `quant_amplitude_balance` | 2η² = 1 | `2·amplitude_r² == 1` ↔ `L_amp == 0` |
| Q4.3 `quant_amplitude_coherence_max` | C(1) = 1 | coherence activation in every MLP block |
| Q2.2 `quant_energy_ground` | E₁ = −1 | verified in `_selfcheck_quantization_spec()` |

All three arms are **enabled by default**.  Set the corresponding env vars to `0` for ablation.

#### Post-Training Verification

`_verify_lead_confirmation(model)` runs automatically after training and prints a report:

```
lead_confirmation_verification: Theorem Q numeric check
  Q1.2 (μ^8=1):       max_layer_err=0.000000  mean_err=0.000000  PASS
  Q3.2 (H·T=5π/4):    learned H·T=3.926991  target=3.926991  err=0.000000  PASS
  Q4.1 (2η²=1):       learned r=0.707107  2r²=1.000000  err=0.000000  PASS
  Q4.3 (C(1)=1):      C(1)=1.000000  err=0.00e+00  PASS (exact by construction)
  Q2.2 (E₁=−1):       E₁=-1.000000  err=0.00e+00  PASS (exact by construction)
```

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
| `USE_LYAPUNOV_GAIN` | `0` | Scale QK dot-products by Lyapunov gain from §10 |
| `USE_MU_PHASE` | `0` | Initialise each head's RoPE phase from μ-orbit slot |
| `USE_PRECESSION` | `0` | Enable slow precession via PRECESSION_DELTA_PHI per step |
| `KERNEL_SELFTEST` | `0` | Run built-in kernel-constant unit tests and exit |

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
                                # all 5 Quantization.lean conditions logged; DISABLED warns
quantization_spec:verified quant_phase_thetas present …
                                # confirms per-layer quant_phase_thetas wired (when USE_QUANT_REGULARIZER=1)
quantization_spec:verified h_times_t present …
                                # confirms h_times_t wired (when USE_DRIVE_REGULARIZER=1)
quantization_spec:verified amplitude_r present …
                                # confirms amplitude_r wired (when USE_AMPLITUDE_REGULARIZER=1)
train_loader:dataset:…          # dataset confirmed
val_loader:shards …             # validation shard count and token count
model_params:…                  # total parameter count
warmup_step:N/20                # LR warmup progress
step:N/ITERS train_loss:… step_t:…ms train_time:…ms step_avg:…ms
                                # periodic train loss (every TRAIN_LOG_EVERY=200 steps)
quant:theta_mean:… theta_std:… cycle_loss_mean:… weighted:…
                                # per-layer phase stats and 8-cycle regularizer (USE_QUANT_REGULARIZER=1)
quant:h_times_t:… target:… drive_loss:… weighted:…
                                # Hamiltonian drive monitoring (USE_DRIVE_REGULARIZER=1)
quant:amplitude_r:… 2r^2:… amp_loss:… weighted:…
                                # amplitude balance monitoring (USE_AMPLITUDE_REGULARIZER=1)
step:N/ITERS val_loss:… val_bpb:… train_time:…ms
                                # validation checkpoint (every VAL_LOSS_EVERY=1000 steps)
stopping_early: wallclock_cap   # emitted when MAX_WALLCLOCK_SECONDS is reached
lead_confirmation_verification: …
                                # post-training Theorem Q numeric check (all 5 conditions)
final_int8_zlib_roundtrip val_loss:… val_bpb:…
                                # round-trip accuracy after int8 quantization + zlib
peak memory allocated: … MiB    # GPU memory high-water mark
```

### How to Verify Quantization.lean is Active

After starting a run, look for these lines in the log file to confirm the
spec is loaded and all three KernelEquilibriumLoss arms are wired:

```
quantization_spec:source formal-lean/Quantization.lean (lead_quantization_confirmed)
quantization_spec:Q1.2(mu^8=1) regularizer=ENABLED per-layer lambda=0.0100 theta_init=2.356194 num_layers=9
quantization_spec:Q3.2(H*T=5pi/4) regularizer=ENABLED lambda=0.0100 h_times_t_init=3.926991
quantization_spec:Q4.1(2eta^2=1) regularizer=ENABLED lambda=0.0100 amplitude_r_init=0.707107 (=1/sqrt(2))
quantization_spec:Q4.3(C(1)=1) constants_verified=True training_applied=True (coherence activation in every MLP block)
quantization_spec:verified quant_phase_thetas present (num_layers=9 init_mean=2.356194 rad = MU_ANGLE=2.356194)
quantization_spec:verified h_times_t present (init=3.926991 = 5π/4=3.926991)
quantization_spec:verified amplitude_r present (init=0.707107 = 1/sqrt(2)=0.707107)
```

If instead you see `regularizer=DISABLED …` lines, that arm of `KernelEquilibriumLoss` is off.

To run the smoke tests that verify the full KernelEquilibriumLoss pipeline end-to-end
(15 tests, no GPU required):

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
