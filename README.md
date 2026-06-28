# Alien Intelligence (AI²)

**A Sub-Quadratic Neural Architecture for Language Modeling**  
*Beyond the Attention Bottleneck: Sparse, Streaming, and Spectral Computation*

---

## Executive Summary

AI² (Alien Intelligence) is a fully-specified neural architecture for language modeling that replaces the quadratic-complexity attention mechanism with a pipeline of six sub-quadratic components, each operating in O(n) or O(1) time per token. The system was built from scratch in C++17 with a CUDA GPU backend, trained on a character-level language modeling task over 1.3M tokens, and validated across 277 unit tests covering every component's correctness, numerical stability, and complexity bounds.

The architecture explores an alternative to the dominant attention paradigm: instead of pairwise comparisons across all positions, AI² uses random projections, streaming state-space models, spectral graph propagation, sparse expert mixtures, and conformal prediction to process sequences. At its core, the hypothesis is that the information flow necessary for language understanding can be sustained through structures other than the full attention matrix — namely, through compressed latent states, locality-sensitive hash routing, and graph Laplacian diffusion.

AI² is not a production system. It is a research artifact — a concrete, testable instantiation of ideas at the intersection of state-space models, sheaf theory, sparse mixture-of-experts, and conformal uncertainty quantification. The codebase is designed for reproducibility, extension, and empirical investigation.

---

## Architecture

AI² processes a sequence of tokens through six chained components, each described below with its mathematical formulation, complexity analysis, and role in the pipeline.

```
Token → SLIE → LSSC → STRE → [UQ] → SSOG → Distribution
                ↑        ↑
           (embed)   (graph)
               \      /
              ATAA (adaptive)
```

### 1. SLIE — Sparse Learned Indexing of Embeddings

SLIE replaces the standard embedding table (size |V| × `d_model`) with a hash-based embedding scheme inspired by the hashing trick and continuous bag-of-hashes architectures.

#### Consistent Hashing Embedding (CHE)

Given a token with integer ID, the embedding is:

$$e(t) = \bigoplus_{j=1}^{k} W_j[h_j(t)]$$

where $h_j: \mathbb{N} \to [0, m)$ are pairwise-independent universal hash functions (Carter-Wegman construction), $W_j \in \mathbb{R}^{m \times (d/k)}$ are learnable weight shards, and $\bigoplus$ denotes concatenation. Each hash lookup is O(1), and the total embedding computation is O(k) — independent of vocabulary size.

This eliminates the memory bottleneck of standard embedding tables: instead of $O(|V| \cdot d)$, the CHE uses $O(k \cdot m \cdot (d/k)) = O(m \cdot d)$ parameters, with $m \ll |V|$ in practice.

#### Streaming Positional Encoding (SPE)

Position information is injected through a tiny recurrent state:

$$p_t = \tanh(W_h \cdot p_{t-1} + W_i \cdot e_t + b)$$

where $p_t \in \mathbb{R}^{d_{pos}}$ is the positional state ($d_{pos} = 8$), and $W_h, W_i, b$ are fixed random projections. The final embedding is $e'_t = e_t + \text{MLP}(p_t)$.

Unlike sinusoidal or learned absolute positional encodings, the SPE is a streaming O(1) update that naturally captures position-relative and distance-dependent information.

#### Count-Min Sketch Frequency Tracking

A Count-Min Sketch with `sketch_depth` parallel hash tables tracks token frequencies:

$$C_j[h_j(t)] \gets C_j[h_j(t)] + 1, \quad \hat{f}(t) = \min_j C_j[h_j(t)]$$

These frequencies provide a simple O(1) estimate of token rarity, exposed as auxiliary features.

### 2. LSSC — Latent Subspace State Compression

LSSC compresses the token sequence through three parallel pathways fused by a learned gate. It operates as a streaming encoder, producing a hidden state $h_t \in \mathbb{R}^{d_{model}}$ for each position.

#### Diagonal State-Space Model (SSM)

The backbone is a complex-valued diagonal linear recurrence:

$$x_{t+1} = \Lambda x_t + B u_t, \quad y_t = \Re(C x_t) + D u_t$$

where $\Lambda = \mathrm{diag}(\lambda_1, \dots, \lambda_{d_{state}})$ with $\lambda_j = e^{\omega_j + i\phi_j}$, and $\omega_j \sim \mathcal{N}(-0.5, 0.1)$, $\phi_j \sim \mathcal{U}(0, 2\pi)$. The eigenvalues are strictly inside the unit circle ($|\lambda_j| < 1$), ensuring bounded dynamics. All parameters are fixed random (not trained), serving as a rich temporal filter bank.

The recurrence can be unrolled in O(n) via sequential scan, or parallelized using the associative scan property of linear recurrences.

#### Random Feature Attention (RFA)

A linear attention mechanism using random Fourier features:

$$\phi(x, w) = \exp(w^\top x - \|x\|^2 / 2)$$

$$\text{Attention}(Q, K, V)_i = \frac{\phi(Q_i) \cdot \sum_{j \leq i} \phi(K_j)^\top V_j}{\phi(Q_i) \cdot \sum_{j \leq i} \phi(K_j)^\top}$$

With $W_{rf} \in \mathbb{R}^{n_{rf} \times d_{model}}$ fixed (rows drawn i.i.d. from $\mathcal{N}(0, I)$), the numerator and denominator are computed in O(n · n_{rf} · d_{model}) = O(n) via prefix sums, avoiding the O(n²) of standard attention.

#### Gated Fusion

The SSM and RFA outputs are combined via a data-dependent gate:

$$\alpha_t = \sigma(W_g z_t + b_g), \quad h_t = \alpha_t \odot y_t^{SSM} + (1 - \alpha_t) \odot y_t^{RFA}$$

where $z_t$ is a learned context vector and $\sigma$ is the logistic sigmoid.

### 3. STRE — Spectral-Topological Reasoning Engine

STRE models the sequence as a graph and performs message passing on its sheaf Laplacian, capturing higher-order token interactions beyond the Markovian horizon.

#### Graph Construction

Given hidden states $H \in \mathbb{R}^{n \times d_{model}}$ from LSSC, STRE first partitions positions into nodes using LSH clustering. Each position is assigned to a node via:

$$\text{node}(i) = \bigcup_{l=1}^{L} \text{LSH}_l(H_i) \mod N_{nodes}$$

Edge set $\mathcal{E}$ connects nodes whose positions are adjacent in the sequence, respecting bounded degree $\Delta = O(1)$. The construction is O(n).

#### Sheaf Laplacian Propagation

With restriction maps $F_{v,u} \in \mathbb{R}^{d_{node} \times d_{node}}$ initialized as identity (constant, deterministic), the sheaf Laplacian is:

$$(\Delta_F x)_v = \sum_{u \sim v} F_{v,u}^\top (F_{v,u} x_v - F_{u,v} x_u)$$

For identity restriction maps, this simplifies to the standard graph Laplacian:

$$(\Delta x)_v = \sum_{u \sim v} (x_v - x_u)$$

Propagation through $L$ layers follows:

$$x_v^{(l+1)} = \tanh\left(W^{(l)} x_v^{(l)} + \sum_{u \sim v} F_{v \to u}^{(l)} x_u^{(l)}\right)$$

where $W^{(l)} \in \mathbb{R}^{d_{node} \times d_{node}}$ are fixed random weights. Each layer is O(n · Δ · d_{node}) = O(n).

#### Conflict Detection

A pathology score quantifies topological irregularity:

$$\text{conflict}(G) = \frac{1}{n} \sum_v \|(\Delta_F x)_v\|^2$$

Nodes with anomalously low feature norm or saturated degree are flagged for downstream uncertainty quantification.

### 4. UQ — Uncertainty Quantification

UQ provides per-token uncertainty estimates through three complementary tracks fused into a single calibrated measure.

#### Conformal Prediction (Track A)

A reservoir of $R$ recent non-conformity scores $s_i = 1 - p(y_i | x_i)$ is maintained. At inference, the $(1-\alpha)$-quantile $\hat{q}$ of the reservoir gives:

$$\mathcal{C}(x) = \{y : 1 - p(y | x) \leq \hat{q}\}$$

The prediction set $\mathcal{C}(x)$ is guaranteed to contain the true label with probability at least $1-\alpha$ under exchangeability. The conformal uncertainty is $|\mathcal{C}| / |\mathcal{V}|$, ranging from 0 (fully certain) to 1 (fully uncertain).

#### Bayesian Ensemble (Track B)

An ensemble of $K$ probabilistic predictors provides:

$$p_{ens}(y) = \frac{1}{K} \sum_k p_{\theta_k}(y)$$

$$\text{Var}_{epistemic} = \text{Var}_k[\mathbb{E}_{y \sim p_{\theta_k}}[y]], \quad
\text{Var}_{aleatoric} = \mathbb{E}_k[\text{Var}_{y \sim p_{\theta_k}}[y]]$$

Epistemic uncertainty captures model disagreement (reducible with more data), while aleatoric captures irreducible noise.

#### Fusion

The three uncertainty signals (conformal, epistemic, aleatoric) are fused with a conflict score from STRE via:

$$C_t = \sigma(W_u \cdot [\text{epistemic}, \text{aleatoric}, \text{conflict}] + b_u)$$

This single scalar $C_t \in [0, 1]$ modulates the output distribution during calibration.

### 5. ATAA — Adaptive Task-Aware Attention

ATAA provides in-context task adaptation through a hypernetwork that generates LoRA adapters, combined with orthogonal gradient descent for continual learning.

#### Task Encoding

A set of $k$ in-context examples is encoded into a task descriptor:

$$\tau = \frac{1}{k} \sum_{i=1}^k \tanh(W_2 \cdot \tanh(W_1 \cdot e_i + b_1) + b_2)$$

where $e_i \in \mathbb{R}^{d_{model}}$ are the LSSC hidden states of example tokens, and $W_1, W_2$ are fixed random projections.

#### Hypernetwork for LoRA Adapters

The task descriptor generates adapter weights:

$$[B, A] = \text{reshape}(W_{h2} \cdot \text{ReLU}(W_{h1} \cdot \tau + b_{h1}) + b_{h2})$$

where $B \in \mathbb{R}^{d_{model} \times r}$, $A \in \mathbb{R}^{r \times d_{model}}$. The task-adapted weight is:

$$W_{task} = W_0 + BA$$

LoRA adaptation injects only $2 \cdot r \cdot d_{model}$ parameters per task instead of full fine-tuning.

#### Orthogonal Gradient Descent (OGD)

To prevent catastrophic forgetting across tasks, gradients are projected onto the null space of previous tasks:

$$U \leftarrow \text{update\_basis}(U, \nabla\mathcal{L}), \quad P_t = I - UU^\top, \quad \nabla_{proj} = P_t \nabla\mathcal{L}$$

The basis $U \in \mathbb{R}^{d_{model} \times s}$ ($s$ = sketch_rank = 16) is maintained via streaming Gram-Schmidt, ensuring new tasks minimally interfere with previously learned representations.

### 6. SSOG — Sparse Set of Generators (Output Head)

SSOG replaces the standard softmax output layer with a sparse mixture of expert projectors, reducing the per-token output computation from $O(|\mathcal{V}| \cdot d_{model})$ to $O(k \cdot d_{model} + |\mathcal{V}|)$ where $k \ll n_{experts}$.

#### LSH Expert Routing

For a hidden state $h \in \mathbb{R}^{d_{model}}$, a set of $k$ experts is selected via:

$$\text{gate}(h) = \{i_1, \dots, i_k : i_t = \text{LSH}_t(h) \bmod n_{experts}\}$$

Each of the $n_{lsh\_tables}$ LSH functions independently routes $h$ to a specific expert; the set is deduplicated and truncated to exactly $k$ experts.

#### Gating and Mixture

Gating weights ensure the selected experts form a convex combination:

$$g_i(h) = \text{softmax}_i(\{W_{gate}[h] \cdot W_j[0]\}_{j \in \text{gate}(h)})$$

Each expert computes a linear projection:

$$E_i(h) = W_i \cdot h + b_i, \quad \text{mixture} = \sum_{i \in \text{gate}(h)} g_i(h) \cdot E_i(h)$$

The mixture aggregates $k$ expert opinions in $O(k \cdot d_{model})$ time.

#### Output Distribution

The final distribution is:

$$p(y) = \text{softmax}(W_{out} \cdot \text{mixture} + b_{out})$$

$$\tilde{p}(y) = (1 - C) \cdot p(y) + C \cdot \frac{\mathbb{1}_{\mathcal{C}}(y)}{|\mathcal{C}|}$$

where $C$ is the conformal uncertainty and $\mathcal{C}$ is the conformal prediction set. When uncertainty is high, probability mass is spread across the prediction set rather than concentrated on a single mode.

---

## Training Pipeline

The entire system is trained end-to-end on a character-level language modeling objective. Only the output projection parameters $W_{out}$ and $b_{out}$ are trained; all other weights (SLIE embedding shards, SSM parameters, random features, STRE projections, expert weights, etc.) are fixed random initializations. This design choice isolates the learning dynamics to the output layer, enabling controlled investigation of whether the fixed random pipeline provides useful representations.

### GPU Acceleration

The training loop exploits a pre-computation stratagem: since the SSOG expert weights are static, their outputs depend only on the hidden state $h$ and can be pre-computed. A HiddenCache stores the SSOG mixture (weighted expert sum), raw hidden state, and routing information for every token in the training set. During each training step, the GPU runs:

```cuda
// Forward: mixture → output projection → softmax → cross-entropy
__global__ void output_proj_kernel(mixture, W_out, b_out, logits, n);

// Backward: gradient of cross-entropy w.r.t. W_out, b_out
// (computed via the same kernel or separate reduction)
```

This reduces the per-step GPU workload from ~3GB of forward computation to ~20MB of output projection, achieving throughput of ~10,000+ tok/s on a Tesla T4.

### Checkpointing

| Frequency | File Pattern | Description |
|-----------|-------------|-------------|
| Every 100 steps | `checkpoints/*_step_*.bin` | Fine-grained resume points |
| Every 3 epochs | `checkpoints/*_epoch_*.bin` | Epoch-level snapshots |
| Final | `checkpoints/*_final.bin` | End-of-training archive |
| Every save | `model/model.bin` | Always-latest model (overwritten) |

The `--resume` flag uses prefix-matching to find the latest checkpoint within the current training stage (pretrain or finetune). If finetune checkpoints exist, pretrain is skipped entirely.

---

## Verification & Testing

The codebase includes 277 test assertions across 8 test executables, covering mathematical correctness, numerical stability, complexity bounds, and integration.

### Component Tests

| Target | Tests | Verification Focus |
|--------|-------|--------------------|
| `test_math` | 10 sections | Types, linear algebra, Optimizer (SGD, Adam, weight decay), gradient correctness (numerical), pipeline integration (10-step training loop) |
| `test_slie` | 9 | CHE determinism, SPE streaming state, Count-Min sketch update, O(1) complexity, OOB handling, numerical stability |
| `test_lssc` | 11 | SSM step/scan linearity, RFA prefix-sum correctness, gated fusion convexity, long-sequence (500-step) numerical stability |
| `test_stre` | 11 | Graph construction bounded degree, Laplacian propagation, conflict score range, pathology detection, connectivity, O(n) scaling |
| `test_uq` | 11 | Conformal score bounds, reservoir enforcement, prediction set size, ensemble statistics, fusion range, numerical stability |
| `test_ataa` | 11 | Task encoding determinism, LoRA adapter shapes, OGD projection norm reduction, continual accumulation |
| `test_ssog` | 13 | Expert routing (k selection, determinism), gating (sum-to-1), mixture shape, base/calibrated distribution properties, many-expert O(1) |

All tests pass on the target platform (GCC 11.4, CUDA 12.8, Tesla T4). Tests are self-contained with no external dependencies.

---

## Open Problems & Areas Requiring Investigation

The architecture was empirically evaluated on a character-level language modeling task. The full inference log is reproduced below for reproducibility:

```
$ bash inf.txt
=== Alien Intelligence (AI²) Inference ===
[1/2] Checking build...
  Binary exists, skipping build
[2/2] Starting inference...
  Usage: pass prompt as arg or pipe input

Using checkpoint: checkpoints/ai2_finetune_final.bin

  GPU: Tesla T4  14912MB  SMs: 40
  GPU acceleration enabled (full forward+backward)

AI² Inference ready. Type input (Ctrl+D to exit).

> hi
=*CPI7$Sxk8+,Yp>BS/,"I#P|tkvNq7
> hello
K8"=j.\v$AV}-MESkZWnoGJ{|&ho&*s\xzd5LF.F-R&u9"fx+b0A/@wI#d&P/OOOKpTCK5YT%rIf!Ya#yyvJu{.Rh6VvX@BBnw,V41]1atsZ<UNK>pX/;Cb3i&qsK/T'YkC<VRK:/<g=ULJ9aAdtp+!y!Wyk&ed8x?zN\v{>$W7wSsUah!A9CO6
```

The model produces tokens from the correct vocabulary but without sequential coherence — it has learned the marginal character distribution but not the conditional dependencies between positions. This section analyzes the likely causes and maps the landscape for future research.

### 1. Learning Dynamics in Fixed-Projection Pipelines

**Observation:** With only $W_{out} \in \mathbb{R}^{|\mathcal{V}| \times d_{model}}$ and $b_{out} \in \mathbb{R}^{|\mathcal{V}|}$ trained (~33K parameters for $|\mathcal{V}|=132$, $d_{model}=256$), the model is effectively a linear probe on top of a fixed random feature map. The random projections in SLIE, LSSC, STRE, and SSOG experts are not optimized for the language modeling objective, meaning the hidden representations are not task-adapted.

**Investigations needed:**
- Train SLIE embedding shards $W_j$ via backpropagation through the hash lookups (straight-through estimator or soft hash relaxation)
- Learn LSSC diagonal SSM eigenvalues $\{\lambda_j\}$ and input projections $B$ rather than fixing them
- Adapt STRE propagation weights $W^{(l)}$ and restriction maps $F_{v,u}$ through the graph Laplacian
- Backpropagate through SSOG expert weights $W_i$ and gating mechanism

A key research question is whether sub-quadratic architectures require **some** trained components at each layer, or whether a purely random projection hierarchy can be sufficient given sufficient width.

### 2. Information Flow Through Random Recurrences

**Observation:** The LSSC state-space model with fixed random eigenvalues is not guaranteed to preserve task-relevant information across long distances. While the eigenvalue spectrum ensures $|\lambda_j| < 1$, it does not ensure that the learned output layer can extract useful information from the high-dimensional state trajectory.

**Investigations needed:**
- Learnable eigenvalue parameterization with stability constraints ($\lambda_j = \sigma(a_j) \cdot e^{i\theta_j}$ where $\sigma$ is a squashing function)
- Time-scale parameterization: initialize eigenvalues to cover a range of decay rates from $\tau \approx 1$ to $\tau \approx 10^4$ tokens
- Diagonal state-space initialization strategies that capture hierarchical temporal structure

### 3. Topological Graph Construction Quality

**Observation:** The LSH-based node clustering in STRE depends on the quality of random projections onto the LSH hyperplanes. For character-level inputs with small $d_{model}$, the clustering may produce either too few or degenerate clusters, limiting the graph's ability to capture meaningful token relationships.

**Investigations needed:**
- Adaptive LSH: learn LSH projections that optimize graph connectivity for the downstream task
- Multi-resolution graph construction with coarsening and refinement
- Sheaf restriction maps that go beyond identity, learned to modulate information flow along edges
- Alternative graph constructions based on syntactic or semantic proximity rather than LSH

### 4. Expert Capacity in Sparse Mixtures

**Observation:** The SSOG uses $n_{experts}=32$ with $k_{experts}=4$ active per token, and expert weights $W_i \in \mathbb{R}^{d_{model} \times d_{model}}$ are fixed random. The fixed random experts do not specialize; each expert computes a random linear projection rather than a learned function.

**Investigations needed:**
- Train expert weights $W_i$ and biases $b_i$ via backpropagation with load-balancing loss
- Expert specialization through auxiliary losses (e.g., $\ell_2$ on expert gradient norms, importance weighting)
- Conditional computation that learns which experts are relevant for which tokens
- Routing mechanism that considers both the input token and its context

### 5. Gradient Signal Through the Pipeline

**Observation:** Gradients do not propagate back through SLIE, LSSC, STRE, or SSOG experts — only $W_{out}$ and $b_{out}$ receive gradient updates. This means the pipeline's representations are effectively frozen at random initialization, regardless of training duration.

**Investigations needed:**
- Can the output layer gradient signal ($\nabla_{W_{out}} \mathcal{L}$) be used to approximate directions for upstream improvement without full backpropagation?
- Is there a theoretical justification for frozen random projections being sufficient (e.g., neural tangent kernel-style analysis)?
- How does the required width scale with frozen vs. trained parameters?

### 6. Evaluation on Structured Tasks

**Observation:** The current evaluation is limited to character-level language modeling. The architectural components — particularly STRE's graph propagation and LSSC's state-space model — may be better suited to different modalities (e.g., graph-structured data, genomic sequences, time series) where their inductive biases align with the data structure.

**Investigations needed:**
- Benchmark on tasks with explicit graph structure (molecular property prediction, code ASTs)
- Evaluate on long-range arenas (ListOps, Pathfinder, Retrieval) to test the sub-quadratic claim
- Probe the SSM-RFA gate: does it learn to prioritize SSM for local patterns and RFA for long-range dependencies?
- Measure the correlation between STRE conflict score and actual prediction error

---

## Open Research Directions

### Architectural
- **Learned end-to-end sub-quadratic layer**: Can SLIE → LSSC → STRE be composed into a single recurrent cell with learnable parameters, maintaining O(n) complexity?
- **Sheaf Laplacian with learned restriction maps**: Moving beyond identity restriction maps $F_{v,u}$ to learned linear maps that conditionally route information based on token content
- **Hierarchical SSOG**: Multi-level expert mixture where coarse experts specialize in broad token classes and fine experts in specific tokens
- **Cross-attention without attention**: Using STRE's graph propagation to implement cross-document reasoning on a bipartite token-document graph

### Theoretical
- **Expressive power of random projection hierarchies**: Under what conditions can a fixed random feature hierarchy (LSSC random features + STRE random projections) approximate any sequence-to-sequence function?
- **Stability-radius tradeoff for diagonal SSMs**: Given a learnable eigenvalue budget $d_{state}$, what initial distribution maximizes the space of realizable temporal dynamics?
- **Sheaf Laplacian spectral theory for sequences**: How does the spectral gap of the constructed graph change with sequence properties (repetition, novelty, syntactic structure)?
- **Conformal prediction with dependent data**: How does the exchangeability assumption degrade for language with long-range dependencies, and can the conformal set size be used as a detection statistic?

### Empirical
- **Ablation study**: Which components contribute most to predictive performance when trained? Isolate SLIE vs. LSSC vs. STRE vs. SSOG contributions
- **Scaling laws**: How does test loss scale with $d_{model}$, $n_{experts}$, $d_{state}$, and $n_{rf}$ for sub-quadratic architectures?
- **Comparison at equivalent compute budget**: Compare a sub-quadratic pipeline to a Transformer of matched parameter count and training FLOPs, not just architecture class
- **In-context learning capability**: Does ATAA's task adaptation via hypernetwork actually enable few-shot learning? Measure on standard few-shot benchmarks

### Engineering
- **Parallel associative scan for SSM training**: Replace sequential scan with Blelloch-style prefix sum for O(log n) parallel training of the state-space model
- **GPU kernel fusion for the full pipeline**: Combine SLIE lookup, LSSC scan, STRE propagation, and SSOG mixture into a single fused kernel to reduce memory bandwidth
- **Reformal inference for STRE**: Use incremental graph updates to avoid O(n) recomputation per token during generation

---

## Conclusion

AI² represents a concrete exploration of the design space beyond the attention bottleneck. The architecture replaces every quadratic component with a sub-quadratic alternative: hash-based embeddings instead of lookup tables, random feature attention and state-space models instead of full attention, graph propagation instead of pairwise interaction, sparse expert mixtures instead of dense projections, and conformal prediction for calibrated uncertainty.

The empirical observation — that a fully random projection pipeline with only output layer training produces tokens from the correct distribution but without sequential coherence — is informative. It suggests that the frozen random feature hierarchy captures marginal statistics but not the conditional dependencies that define language. This is consistent with the theory: random projections preserve distances (Johnson-Lindenstrauss) but do not discover task-specific structure without task-specific gradients.

The path forward involves strategic relaxation of the frozen-parameter constraint: backpropagating through the SSOG experts, LSSC state parameters, or SLIE hash embeddings. The sub-quadratic complexity guarantees are preserved as long as the component structure (hash-based lookup, prefix-sum attention, sparse graph, fixed-degree routing) remains intact — only the numerical values within that structure become learned.

This codebase is released as a research artifact for the community to build upon, ablate, and extend. The complete source, including all 277 tests, GPU kernels, and training infrastructure, is available for reproduction and modification.

---

**Repository**: [github.com/nexuss0781/alien-intelligence](https://github.com/nexuss0781/alien-intelligence)  
**Build**: C++17, CUDA 12+, CMake  
**License**: MIT
