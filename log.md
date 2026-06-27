=== Alien Intelligence (AI²) Build & Test ===
-- Configuring done (0.0s)
-- Generating done (0.0s)
-- Build files have been written to: /content/alien-intelligence/build
[ 25%] Built target ai2_core
[ 33%] Built target test_slie
[ 40%] Built target test_lssc
[ 62%] Built target ai2_train_lib
[ 70%] Built target test_stre
[ 77%] Built target test_uq
[ 85%] Built target test_ataa
[ 92%] Built target test_ssog
[ 96%] Linking CXX executable ai2_train
[100%] Built target ai2_train

=== Running Tests ===
=== Component 5: ATAA Tests ===
  Task encoding produces fixed-size embedding... PASS
  Task encoding is deterministic for same input... PASS
  Hypernetwork produces correct LoRA adapter shapes... PASS
  LoRA application produces correct output dimension... PASS
  LoRA produces different output from base... PASS
  Gradient subspace update works... PASS
  Gradient projection does not change shape... PASS
  Projected gradient has smaller or equal norm... PASS
  Full adaptation returns adapters... PASS
  Current task is set after adaptation... PASS
  No NaN/Inf in hypernetwork outputs... PASS
  OGD accumulates multiple gradient updates... PASS
All ATAA tests PASSED.

=== Component 2: LSSC Tests ===
  SSM step produces consistent output dimensions... PASS
  SSM state evolves over time... PASS
  SSM scan processes sequence in O(n)... PASS
  RFA random features have correct dimension... PASS
  RFA features are bounded and positive... PASS
  RFA causal produces correct output shape... PASS
  RFA causal first token depends only on itself... PASS
  Gated fusion produces correct output... PASS
  Gated fusion produces convex combination of SSM and RFA... PASS
  Full forward processes sequence in O(n)... PASS
  No NaN/Inf during long sequence processing... PASS
  SSM remains stable for many steps... PASS
All LSSC tests PASSED.

=== Component 1: SLIE Tests ===
  CHE produces valid embeddings... PASS
  CHE is deterministic (same token -> same embedding)... PASS
  CHE gives different embeddings for different tokens... PASS
  SPE updates positional state... PASS
  SPE gives different states for different positions... PASS
  Sketch updates counters... PASS
  reset_position resets state... PASS
  O(1) per token — embedding dimension independent... PASS
  Handles OOB vocabulary indices gracefully... PASS
  No NaN or Inf in outputs... PASS
All SLIE tests PASSED.

=== Component 6: SSOG Tests ===
  LSH routing selects k experts... PASS
  LSH routing is deterministic for same input... PASS
  LSH routing can give different experts for different inputs... PASS
  Gating weights sum to 1... PASS
  Gating weights are non-negative... PASS
  Expert forward produces correct output shape... PASS
  Sparse mixture produces correct shape... PASS
  Base distribution sums to 1... PASS
  Base distribution probabilities are non-negative... PASS
  Calibrated distribution sums to 1... PASS
  Calibrated distribution spreads mass when uncertain... PASS
  Full forward produces valid output... PASS
  No NaN/Inf in any outputs... PASS
  Handles many experts efficiently (O(1) per token)... PASS
All SSOG tests PASSED.

=== Component 3: STRE Tests ===
  Graph construction produces bounded-degree graph... PASS
  No duplicate edges in graph... PASS
  Restriction maps initialized correctly... PASS
  Propagation produces correct output dimensions... PASS
  Forward pass produces no NaN/Inf... PASS
  Conflict score is non-negative and bounded... PASS
  Pathology detection runs without error... PASS
  Graph has reasonable connectivity... PASS
  Graph respects bounded degree (treewidth proxy)... PASS
  Different seeds/states produce different graphs... PASS
  Handles many nodes efficiently (O(n) check)... PASS
All STRE tests PASSED.

=== Component 4: UQ Tests ===
  Conformal score is in [0,1]... PASS
  Conformal score = 1 - p... PASS
  Reservoir updates correctly... PASS
  Reservoir respects maximum size... PASS
  Prediction set is non-empty for confident predictions... PASS
  Prediction set can be large for uncertain predictions... PASS
  Ensemble accepts member predictions... PASS
  Ensemble clears correctly... PASS
  Epistemic uncertainty is in [0,1]... PASS
  Aleatoric uncertainty is in [0,1]... PASS
  Fused uncertainty is in [0,1]... PASS
  Full UQ step produces valid result... PASS
  No NaN/Inf in any UQ outputs... PASS
All UQ tests PASSED.

=== All components built and tested successfully ===

