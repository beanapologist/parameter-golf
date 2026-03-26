# Kernel Constants: Coherence Activation + Silver-Ratio MLP + Z/8Z Heads + Palindrome LR

**val_bpb: placeholder** | **hardware: 8×H100 SXM** | **runtime: ≤10 min (600s)**

## Summary of Changes vs Baseline

This submission wires three mathematical objects derived from the formal Lean 4 theorems in
`formal-lean/CriticalEigenvalue.lean` directly into the model architecture and learning-rate
schedule:

| Component | Baseline | This submission |
|-----------|----------|-----------------|
| MLP nonlinearity | relu² | Coherence activation C(r) = 2r/(1+r²) |
| MLP hidden width multiplier | 3× model_dim | Silver ratio δS ≈ 2.414 × model_dim |
| Number of attention heads | 8 (arbitrary) | 8 (Z/8Z — each head occupies one μ-orbit slot) |
| Warmdown LR shape | cosine | Palindrome-precession C(r) = 2r/(1+r²) |
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

### Experimental kernel feature flags (all OFF by default)

| Variable | Default | Description |
|----------|---------|-------------|
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
train_loader:dataset:…          # dataset confirmed
val_loader:shards …             # validation shard count and token count
model_params:…                  # total parameter count
warmup_step:N/20                # LR warmup progress
step:N/ITERS train_loss:… step_t:…ms train_time:…ms step_avg:…ms
                                # periodic train loss (every TRAIN_LOG_EVERY=200 steps)
step:N/ITERS val_loss:… val_bpb:… train_time:…ms
                                # validation checkpoint (every VAL_LOSS_EVERY=1000 steps)
stopping_early: wallclock_cap   # emitted when MAX_WALLCLOCK_SECONDS is reached
final_int8_zlib_roundtrip val_loss:… val_bpb:…
                                # round-trip accuracy after int8 quantization + zlib
peak memory allocated: … MiB    # GPU memory high-water mark
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
