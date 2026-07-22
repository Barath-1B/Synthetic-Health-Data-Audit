## VAE Decoder Architecture Fix

### Problem (old architecture)
The original VAE decoder applied `sigmoid` to all output columns uniformly.
Sigmoid saturates at x → ±∞, compressing values near 0 and 1.
For wide-range continuous features (Age, Income_Ratio, Total_Calories, etc.),
sigmoid saturation produces compressed synthetic distributions that fail KS tests.

### Fix (new architecture)
Column-type-aware output activation:
- **Binary columns** (9 of 23): sigmoid — constrain to [0,1] naturally
- **Continuous columns** (14 of 23): clamp(x, 0.0, 1.0) — linear in the valid range,
  no saturation, preserves distributional spread

Binary columns detected by get_binary_col_indices() in train.py:
  Gender, Arthritis, Congestive_Heart_Failure, Coronary_Heart_Disease,
  Heart_Attack, Stroke, Cancer, Trouble_Sleeping_Doc, Ever_Had_Drink

Additional encoder improvement: added BatchNorm1d + LeakyReLU(0.2)
replacing plain ReLU, for more stable training.

### KS pass rate
5-epoch smoke test (pre-convergence, for architectural validation only):
  Old arch (sigmoid, all cols):    0/23 pass  rate=0.000
  New arch (clamp, cont cols):     0/23 pass  rate=0.000

Note: 5 epochs is far below convergence for both architectures — this comparison
does not reflect final quality. Both produce 0/23 at 5 epochs because the model
has not yet learned to reproduce the real data distribution.

Final KS pass rate before/after will be recorded here after full 200-epoch training:
  Pre-fix (sigmoid all cols):   [PENDING — run seed 42 with old arch 200 epochs]
  Post-fix (clamp cont cols):   [PENDING — run seed 42 with new arch 200 epochs]

Key validation: new arch produces no out-of-range values: CONFIRMED
(syn.min() >= 0 and syn.max() <= 1 verified on 1000 generated rows)
