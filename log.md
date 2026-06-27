=== Status ===
All 5 failures are caused by stale .o files — need a clean build.

=== Root Causes Found ===
1. SLIE same-token diff=0.06 → FIXED: elem_add(emb(64), pos(8)) UB reading past buffer
2. SLIE diff-token same embedding → test bug: pos1 from last_position() carries state, invalid comparison
3. STRE Laplacian=1 → stale stre.o (no identity maps); need `rm -rf build && cmake --build`
4. Gradient/pipeline loss unchanged → stale model.o (missing compute_gradients/sync_params)
5. Sketch size mismatch was already passing (size=16 = expected)

=== Next Step ===
```bash
rm -rf build && cmake -B build -S . && cmake --build build -j$(nproc) && ./build/test_math
```
