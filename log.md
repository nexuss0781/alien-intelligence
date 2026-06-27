=== Alien Intelligence (AI²) Training Log ===

## Run 1 (old, broken)
- **Loss stuck at ln(132)=4.8828**: no gradient computation in trainer
- **Conflict inconsistent**: STRE restriction maps re-initialized every forward
- **Stopped at step 16**: Colab process killed early

## Fixes applied (current)
- Added gradient computation for output layer (W_out, b_out) in Model
- Trainer now calls zero_grad() → compute_gradients() → optimizer.step() → sync_params
- STRE: identity restriction maps, stored per-layer weights
- Comprehensive math tests (tests/test_math.cpp) covering all 6 components
- Fixed link order in CMakeLists.txt

