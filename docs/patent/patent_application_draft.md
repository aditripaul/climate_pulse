# PATENT APPLICATION DRAFT
**Status:** Provisional Draft — Not for Filing Without Legal Review  
**Prepared by:** ClimatePulse Research Team  
**Date:** July 2026  
**Subject System:** ClimatePulse — AI-Powered Global Climate Forecasting Framework  
**Code Reference:** git:81edd882785b7b3bfb8119193af6ddac2c5355ad  

> **IMPORTANT**: This document is a **preliminary technical draft** for attorney review. It has **not** been reviewed by a registered patent attorney or patent agent. It must not be filed as-is with any patent office. The claims herein have not been subjected to a professional prior-art search or freedom-to-operate (FTO) analysis. All numerical results cited reflect empirical measurements from the ClimatePulse evaluation pipeline (`make gen-eval-artifacts`).

---

## PRELIMINARY NOTES FOR COUNSEL

### Recommended Filing Strategy

1. **Immediately file a US Provisional Application** (35 U.S.C. § 111(b)) to establish a priority date before any public disclosure.
2. Convert to a **Non-Provisional Application** within 12 months.
3. File **PCT Application** (PCT/US) simultaneously with the Non-Provisional to preserve international rights (EP, CN, JP, IN, AU).

### Public Disclosure Risk Assessment

| Item | Status | Risk |
|:---|:---|:---|
| GitHub repository (claimed public) | Confirm if currently public or private | **CRITICAL**: If public, US grace period clock has started; EP/CN rights are lost |
| Research paper PDF (`climatepulse_research_paper.pdf`) | Present in repo | Verify if submitted to arXiv or journal |
| `research_paper.md` watermark `research.aditripaul.org` | Suggests a public website | Verify date of first web publication |

### Three Recommended Patent Applications

| Application | Core Invention | Priority |
|:---|:---|:---|
| **Patent 1** | ENSO-Conditioned Spherical Fibonacci Mesh Neural Weather Model | **Highest** |
| **Patent 2** | Unified Multi-Modal Climate Inference System with Integrated Cross-Domain Conditioning | High |
| **Patent 3** | Latitude-Cosine-Weighted Physical Loss Function for Spherical-Domain Graph Neural Networks | Medium |

---

---

# PATENT APPLICATION — DOCUMENT 1 (PRIMARY)

## TITLE

ENSO-CONDITIONED ENCODER-PROCESSOR-DECODER NEURAL WEATHER MODEL OPERATING ON A FIBONACCI SPHERICAL LATENT MESH WITH VECTORIZED BIPARTITE GRAPH-SAGE CONVOLUTION

---

## FIELD OF THE INVENTION

The present invention relates to machine learning systems for global atmospheric field prediction. More particularly, it relates to a neural weather model architecture that employs (i) a Fibonacci golden-angle lattice as a latent computational spherical mesh; (ii) tripartite bipartite Graph Sample and Aggregation (GraphSAGE) convolutions between an equirectangular observation grid, the Fibonacci latent mesh, and back to the observation grid; and (iii) a real-time conditioning mechanism that injects El Nino-Southern Oscillation (ENSO) scalar anomaly information into every latent mesh processor layer via a learned Multi-Layer Perceptron (MLP) projection.

---

## BACKGROUND

### The Global Weather Prediction Problem

Global numerical weather prediction (NWP) systems — such as the Integrated Forecasting System (IFS) operated by ECMWF — produce highly accurate atmospheric forecasts but require dedicated supercomputing clusters running for hours per forecast cycle. The consequent latency renders real-time decision support impractical in many civil protection, agricultural, and energy-dispatch applications.

### Prior Art: Data-Driven Weather Models and Their Limitations

Data-driven neural weather models have emerged as a computationally efficient alternative. Lam et al. (2022, "GraphCast") and Keisler (2022) established the Encoder-Processor-Decoder paradigm on icosahedral meshes. Pathak et al. (2022, "FourCastNet") demonstrated global forecasting using Adaptive Fourier Neural Operators. These systems share the following limitations:

1. **Polar Distortion**: Standard equirectangular grid CNNs apply spatially-invariant convolution kernels across the entire globe. Because equirectangular grids oversample polar regions (a 5.625 degree cell covers approximately 395 km^2 at 80N but approximately 3,400 km^2 at the equator), convolution kernels apply unequal computational effort per unit physical area, introducing a systematic spatial bias.

2. **Single-Task Architecture**: All prior neural weather models are single-task: they map an atmospheric state at time t to a predicted state at time t+1 (or t+k). None incorporate a separate, dedicated signal representing the **slowly-varying macro-climatic forcing** (e.g., the current ENSO phase) as a persistent conditioning signal injected into the latent message-passing computation. As a result, such models treat a La Nina year identically to an El Nino year at the initialization step, despite the documented global teleconnective influence of the Nino 3.4 index on regional precipitation and temperature anomalies.

3. **Icosahedral vs. Fibonacci Mesh**: Prior art (GraphCast, Keisler) uses icosahedral mesh refinement. Icosahedral meshes require hierarchical subdivision and yield a fixed set of mesh resolutions (20, 80, 320, 1280, 5120 nodes). The Fibonacci golden-angle spiral lattice provides a *continuously* parameterized, provably near-uniform distribution of points on the unit sphere for any target count N, enabling finer control over the resolution-computation tradeoff without the combinatorial constraints of icosahedral subdivision.

4. **Absence of Physical Area Weighting in the Training Objective**: Prior data-driven weather models either omit spatial area weighting entirely or implement it as a post-hoc metric. Equatorial cells represent physically larger surface areas than polar cells; a uniform MSE loss therefore disproportionately optimizes forecast skill at the poles, where forecast verification data are also sparser and less reliable.

### Need in the Art

There is an unmet need for a neural weather model that: (a) employs a spherical latent mesh that avoids equirectangular polar distortion and can be parameterized to any resolution; (b) explicitly conditions all latent message-passing layers on the large-scale ENSO forcing signal; and (c) applies a physically motivated area-weighted training objective.

---

## SUMMARY OF THE INVENTION

The present invention provides a neural weather forecasting system and method comprising:

**Claim Area 1.** A spherical latent mesh generated using the Fibonacci golden-angle spiral lattice, producing N near-uniformly distributed nodes on the unit sphere for any user-specified N, with no constraint on N being a power of 2 or an icosahedral subdivision level.

**Claim Area 2.** Three bipartite GraphSAGE convolution stages — Grid-to-Mesh (G2M), Mesh-to-Mesh (M2M, with residual connections), and Mesh-to-Grid (M2G) — operating on edge sets constructed via KD-Tree k-nearest-neighbor queries in 3D Cartesian unit-sphere space.

**Claim Area 3.** An ENSO conditioning mechanism in which the scalar Nino 3.4 sea surface temperature anomaly index is passed through a learned two-layer MLP and the resulting d-dimensional embedding is expanded across all N mesh nodes and **additively injected into each message-passing step** of the M2M processor, **before** the residual connection, so that the macro-climatic anomaly modulates the spatial message passing at every processor layer without being added twice through the residual path.

**Claim Area 4.** A vectorized batch-offset edge expansion algorithm that converts a single-graph [2, E] edge index tensor into a batched [2, B x E] edge index using PyTorch tensor primitives `repeat` and `repeat_interleave`, eliminating Python-level loops over the batch dimension and enabling GPU-efficient batched training of graph neural networks.

**Claim Area 5.** A cosine-latitude-weighted MSE loss function that assigns each grid node a loss weight proportional to the cosine of its geographic latitude, normalized to unit mean, so that the training objective reflects the physical surface area represented by each grid node.

---

## DETAILED DESCRIPTION OF THE PREFERRED EMBODIMENT

### 1. System Architecture Overview

The claimed Neural Weather Model (hereinafter "the NWM") operates on global atmospheric reanalysis data formatted as regular equirectangular grids. In the preferred embodiment, the input data comprise ERA5 reanalysis at 5.625 degree spatial resolution, yielding a grid of $N_{\text{grid}}$ = 32 x 64 = 2,048 nodes. Each node carries C=2 atmospheric channels (2-meter temperature T2m; total precipitation TP) over a sliding temporal window of W=2 time steps, plus 4 static positional features, giving grid_in_channels = C x W + 4 = 8 input features per node.

The NWM processes a single sample through five sequential stages:

```
Stage 1: Grid Embedding       (Linear + LayerNorm + GELU) -> [B * $N_{\text{grid}}$, d]
Stage 2: Encoder G2M          (BipartiteGNN: Grid -> Mesh) -> [B * $N_{\text{mesh}}$, d]
Stage 3: Processor M2M x L   (SAGEConv + ENSO cond. + residual) -> [B * $N_{\text{mesh}}$, d]
Stage 4: Decoder M2G          (BipartiteGNN: Mesh -> Grid + skip) -> [B * $N_{\text{grid}}$, d]
Stage 5: Output Projection    (Linear) -> [B * $N_{\text{grid}}$, grid_out_channels]
```

In the preferred embodiment: d = 128, L = 4, $N_{\text{mesh}}$ = 2,000, $N_{\text{grid}}$ = 2,048.

### 2. Fibonacci Spherical Mesh Generation

The latent mesh is generated using the Fibonacci golden-angle spiral lattice. For a target mesh size of $N_{\text{mesh}}$ points, the i-th mesh node (i = 0, ..., $N_{\text{mesh}}$ - 1) is placed at:

$$ \phi_{\text{golden}} = \pi (3 - \sqrt{5}) \quad [\text{golden angle, radians}] $$
$$ y_i = 1 - \frac{2i}{N_{\text{mesh}} - 1} \quad [\text{uniform in } [-1, +1]] $$
$$ r_i = \sqrt{\max(1 - y_i^2, 0)} \quad [\text{radius in } xy\text{-plane}] $$
$$ \theta_i = \phi_{\text{golden}} \cdot i \quad [\text{azimuthal angle}] $$
$$ x_i = \cos(\theta_i) \cdot r_i $$
$$ z_i = \sin(\theta_i) \cdot r_i $$

This produces a set of $N_{\text{mesh}}$ 3D Cartesian unit-sphere points with provably near-uniform angular coverage. The key property exploited by the invention is that phi_golden is maximally irrational, ensuring no two lattice points lie at the same azimuthal angle within any finite prefix of the sequence. This prevents the clustering artifacts that arise in other parameterizations.

The Fibonacci mesh construction is implemented in `src/models/nwm/graph_builder.py::generate_fibonacci_sphere()` (lines 23-32 of the reference implementation). The construction is fully vectorized using NumPy array operations with no Python-level per-point loops.

### 3. Bipartite Edge Construction via KD-Tree Nearest Neighbor

Three distinct bipartite edge sets are constructed using KD-Trees built on the 3D Cartesian coordinates of grid and mesh nodes:

| Edge Set | Source Set | Target Set | k (Neighbors) | Edges (Preferred Embodiment) |
|:---|:---|:---|:---:|:---:|
| Grid-to-Mesh (G2M) | 2,048 grid nodes | 2,000 mesh nodes | 4 | 8,192 |
| Mesh-to-Mesh (M2M) | 2,000 mesh nodes | 2,000 mesh nodes | 8 (excl. self) | 16,000 |
| Mesh-to-Grid (M2G) | 2,000 mesh nodes | 2,048 grid nodes | 4 | 8,000 |

For G2M edges: each grid node i is connected to its k=4 nearest mesh nodes, determined by Euclidean distance in 3D unit-sphere space (equivalent to great-circle distance). The choice of 3D Cartesian distance avoids the singularity at the poles inherent in longitude-based distance metrics, allows uniform spatial indexing via KD-Tree, and preserves rotational isotropy.

For M2M edges: self-loops are explicitly excluded by querying k+1 neighbors and discarding the first returned index (which is always the query node itself).

The edge construction is implemented in `src/models/nwm/graph_builder.py::SphericalGraphBuilder` (lines 35-94 of the reference implementation).

### 4. Static Grid Node Features

Each grid node is assigned four static positional features derived from its geographic latitude phi and longitude lambda:

$$ \mathbf{f}_{\text{static}} = [\sin(\phi), \cos(\phi), \sin(\lambda), \cos(\lambda)] \in \mathbb{R}^4 $$

These features are constant across all time steps and are concatenated to the dynamic atmospheric features at the input to Stage 1. The sine-cosine encoding provides: (a) continuity at the dateline; (b) rotational information at the poles; and (c) unambiguous separation of latitude and longitude.

### 5. Grid Embedding Stage

The Grid Embedding stage applies an MLP to every grid node independently:

$$ \mathbf{h}_g = \text{GELU}(\text{LayerNorm}(\text{Linear}(\mathbf{x}_g, d))) \in \mathbb{R}^d $$

where $\mathbf{x}_g \in \mathbb{R}^{\text{grid\_in\_channels}}$ and $d = 128$ (preferred embodiment). This embedding is retained as a skip connection for the decoder stage.

### 6. Encoder: Grid-to-Mesh (G2M) Bipartite GraphSAGE

The encoder propagates information from the 2,048 observation grid nodes into the 2,000 latent mesh nodes:

$$ \mathbf{h}_m = \text{GELU}(\text{LayerNorm}(\text{SAGEConv}((\mathbf{h}_g, \mathbf{0}), \text{G2M\_edges}))) \in \mathbb{R}^{N_{\text{mesh}} \times d} $$

The initial mesh node states are set to the zero vector ($\mathbf{h}_{m,\text{init}} = \mathbf{0} \in \mathbb{R}^d$). The SAGEConv bipartite formulation aggregates neighbor messages from grid nodes into each mesh node using mean aggregation, then applies a linear projection. The zero initialization ensures the encoder output is determined entirely by the aggregated grid information.

### 7. Processor: ENSO-Conditioned Mesh-to-Mesh Message Passing

The processor applies L=4 stacked GraphSAGE message-passing steps over the M2M edge set with residual connections. The novel ENSO conditioning mechanism operates as follows:

**Step 7.1 — Condition Embedding:**

The scalar Nino 3.4 index nino_34 in R is passed through a two-layer condition MLP:

$$ \mathbf{c} = \text{MLP}_{\text{cond}}(\text{nino\_34}) = \text{Linear}_2(\text{GELU}(\text{Linear}_1([\text{nino\_34}]))) \in \mathbb{R}^d $$

where $\text{Linear}_1: \mathbb{R}^1 \to \mathbb{R}^d$ and $\text{Linear}_2: \mathbb{R}^d \to \mathbb{R}^d$.

**Step 7.2 — Mesh-wide Injection:**

The d-dimensional condition embedding c is tiled across all $N_{\text{mesh}}$ mesh nodes to produce a conditioning tensor:

$$ \mathbf{C} = \mathbf{c}\text{.unsqueeze}(0)\text{.expand}(N_{\text{mesh}}, d) \in \mathbb{R}^{N_{\text{mesh}} \times d} $$

**Step 7.3 — Conditioned Message Passing (per processor layer l):**

$$ \mathbf{msg}^{(l)} = \text{SAGEConv}(\mathbf{h}_m^{(l)}, \text{M2M\_edges}) \quad [\text{spatial message passing}] $$
$$ \mathbf{msg}^{(l)} = \text{LayerNorm}(\mathbf{msg}^{(l)} + \mathbf{C}) \quad [\text{condition injected into msg only}] $$
$$ \mathbf{msg}^{(l)} = \text{GELU}(\mathbf{msg}^{(l)}) $$
$$ \mathbf{h}_m^{(l+1)} = \mathbf{h}_m^{(l)} + \mathbf{msg}^{(l)} \quad [\text{residual over unconditioned state}] $$

The critical design decision is that the condition tensor C is **added to the message only**, not to the residual connection. This prevents the Nino 3.4 signal from being added twice per layer (once explicitly in the message path, once again through the residual carrying the pre-conditioned state from the previous layer). The result is that each processor layer applies exactly one unit of ENSO conditioning, maintaining a consistent signal magnitude across all L layers regardless of depth.

This architecture is fundamentally distinct from prior ENSO conditioning approaches that either: (a) inject ENSO information only at the input encoding step (losing the signal after the first message-passing round); or (b) concatenate it to node features implicitly (which provides no guarantee that the macro-climatic signal is preserved across deep stacks).

### 8. Decoder: Mesh-to-Grid (M2G) Bipartite GraphSAGE with Skip Connection

The decoder propagates processed mesh representations back to the observation grid:

$$ \mathbf{h}_{g,\text{out}} = \text{SAGEConv}((\mathbf{h}_m^{(L)}, \mathbf{0}), \text{M2G\_edges}) + \mathbf{h}_g \in \mathbb{R}^{N_{\text{grid}} \times d} $$

The addition of h_g (the grid embedding from Stage 1) implements a skip connection from the encoder input to the decoder output, ensuring locally specific atmospheric information is not lost during the latent spherical mesh computation.

### 9. Output Projection

The output projection applies a single linear layer:

$$ \hat{\mathbf{y}} = \text{Linear}(\mathbf{h}_{g,\text{out}}) \in \mathbb{R}^{N_{\text{grid}} \times \text{grid\_out\_channels}} $$

where $\text{grid\_out\_channels} = C \times \text{prediction\_steps} = 2$ (T2m and TP for one step ahead, in the preferred embodiment).

### 10. Vectorized Batch-Offset Edge Expansion Algorithm

In standard PyTorch Geometric frameworks, applying a single graph's edge index to a batched graph of B samples requires either converting to a PyG Batch object or looping in Python over the batch dimension (O(B x E) Python overhead). The present invention employs a fully vectorized approach:

Given a single-graph edge index `edge_idx` $\in \mathbb{Z}^{2 \times E}$, the batched edge index for $B$ samples is computed as:

```python
src_offsets = arange(B) * num_src   # [B] -> repeat_interleave(E) -> [B*E]
tgt_offsets = arange(B) * num_tgt   # [B] -> repeat_interleave(E) -> [B*E]
batched_src = edge_idx[0].repeat(B) + src_offsets   # [B*E]
batched_tgt = edge_idx[1].repeat(B) + tgt_offsets   # [B*E]
# Result: [2, B*E]
batched_edge_idx = stack([batched_src, batched_tgt], dim=0)
```

This computation executes as a single GPU tensor kernel without any Python-level iteration. The algorithm is general-purpose and applicable to any bipartite GNN processing batches of identically-structured graphs.

The algorithm is implemented in `src/models/nwm/stgnn_model.py::_batch_edges()` (lines 11-45 of the reference implementation).

### 11. Latitude-Cosine-Weighted MSE Loss

The training objective weights each grid node's squared prediction error by the cosine of its geographic latitude:

$$ w_i = \frac{\cos(\phi_i)}{\frac{1}{N_{\text{grid}}} \sum_{j=1}^{N_{\text{grid}}} \cos(\phi_j)} \quad \text{for } i \in \{1, \dots, N_{\text{grid}}\} $$
$$ \mathcal{L} = \frac{1}{B} \sum_{b=1}^{B} \frac{1}{N_{\text{grid}}} \sum_{i=1}^{N_{\text{grid}}} w_i \sum_{c} (\hat{y}_{b,i,c} - y_{b,i,c})^2 $$

This weighting reflects the physical fact that a grid cell at latitude phi subtends a surface area proportional to cos(phi) relative to an equatorial cell. The normalization `w_i = cos(phi_i) / mean(cos(phi))` ensures the total weighted loss has the same scale as an unweighted MSE.

The implementation is in `src/models/nwm/trainer.py::NWMTrainer::_weighted_mse()` (lines 262-277) and `src/models/nwm/dataset.py` (lines 56-92).

Additionally, normalization statistics for T2m and TP are computed using latitude-cosine-weighted averaging across the full spatial-temporal domain:

$$ \mu_{\text{t2m}} = \frac{\sum_i w_i \cdot \text{t2m\_spatial\_mean}_i}{\sum_i w_i} $$
$$ \sigma_{\text{t2m}}  = \sqrt{\frac{\sum_t \sum_i w_i (\text{t2m}_{t,i} - \mu_{\text{t2m}})^2}{T \sum_i w_i}} $$

This prevents normalization statistics from being dominated by the numerically large count of high-latitude nodes, ensuring the global mean and variance reflect physical surface area rather than raw grid-cell count. The same latitude-cosine weight vector is used for both the normalization statistics and the training loss (Section 11), providing mathematical consistency between the input feature distribution and the optimization objective.

### 12. Training Configuration (Preferred Embodiment)

| Hyperparameter | Value |
|:---|:---|
| Hidden dimension d | 128 |
| Processor layers L | 4 |
| G2M neighbors k | 4 |
| M2M neighbors k | 8 (excl. self) |
| M2G neighbors k | 4 |
| Optimizer | AdamW |
| Learning rate | 1e-3 |
| Weight decay | 1e-5 |
| LR scheduler | ReduceLROnPlateau (factor=0.5, patience=3) |
| Gradient clip norm | 1.0 |
| Max epochs | 100 |
| Early stopping patience | 10 (val MSE) |

### 13. Empirical Performance

The NWM was trained on ERA5 WeatherBench2 data from 2020-2022 at 5.625 degree resolution.

| Metric | Epoch 0 | Best Checkpoint (Epoch 78) | Final (Epoch 88) |
|:---|:---:|:---:|:---:|
| Training MSE | 0.2938 | 0.1045 | 0.1035 |
| Validation MSE | 0.1955 | **0.1182** | 0.1185 |

Training MSE improved by 64.8% from initialization. Validation MSE improved by 39.5%.

*Source: `runtime/eval_artifacts/run_results/summary_metrics.json`*

## BRIEF DESCRIPTION OF THE DRAWINGS

To satisfy 37 C.F.R. § 1.81, the following formal drawings are incorporated by reference:
- **FIG. 1**: A block diagram of the unified multi-modal climate analysis system.
- **FIG. 2**: A visual comparison of the Fibonacci spiral lattice mesh versus standard icosahedral meshes.
- **FIG. 3**: A schematic of the ENSO macro-climatic conditioning injection mechanism.
- **FIG. 4**: A flowchart of the vectorized batch-offset edge expansion algorithm for GPU acceleration.
- **FIG. 5**: A graph showing training and validation loss curves utilizing the latitude-cosine-weighted objective.

---

## CLAIMS — PATENT APPLICATION DOCUMENT 1

**Claim 1** (Independent, System)

A neural weather forecasting system comprising:

(a) a spherical latent mesh comprising $N_{\text{mesh}}$ nodes placed at positions generated by a Fibonacci golden-angle spiral lattice on a unit sphere, wherein each node position $i$ is determined by a golden angle $\phi_{\text{golden}} = \pi(3 - \sqrt{5})$ radians and an index-parameterized y-coordinate $y_i = 1 - \frac{2i}{N_{\text{mesh}}-1}$, and wherein $N_{\text{mesh}}$ is a user-configurable integer not constrained to icosahedral subdivision levels;

(b) an encoder comprising a first bipartite GraphSAGE convolutional layer that propagates features from a set of $N_{\text{grid}}$ observation grid nodes to said $N_{\text{mesh}}$ spherical latent mesh nodes via Grid-to-Mesh edge connections determined by k-nearest-neighbor queries in three-dimensional Cartesian unit-sphere space;

(c) a processor comprising L stacked mesh-to-mesh GraphSAGE message-passing layers, each layer incorporating a conditioning mechanism that additively injects a learned projection of a macro-climatic scalar anomaly index into the computed message tensor before a residual connection, wherein said injection is applied to the message only and not to the residual state;

(d) a decoder comprising a second bipartite GraphSAGE convolutional layer that propagates processed mesh-node representations back to said $N_{\text{grid}}$ observation grid nodes via Mesh-to-Grid edge connections, with a skip connection from the output of (b); and

(e) an output projection layer that maps decoded grid-node representations to predicted atmospheric field values at one or more future time steps.

**Claim 2** (Dependent on Claim 1)

The system of Claim 1, wherein the macro-climatic scalar anomaly index is the Nino 3.4 sea surface temperature anomaly index representing the departure from climatological mean sea surface temperature in the region bounded by 5S-5N latitude and 170W-120W longitude.

**Claim 3** (Dependent on Claim 1)

The system of Claim 1, wherein the conditioning mechanism of (c) comprises: (i) a two-layer MLP with GELU activations mapping the scalar anomaly index to a $d$-dimensional embedding vector; (ii) broadcasting said embedding vector across all $N_{\text{mesh}}$ mesh nodes to produce a conditioning tensor $\mathbf{C} \in \mathbb{R}^{N_{\text{mesh}} \times d}$; and (iii) at each processor layer $l$, computing: message $\mathbf{m}_l = \text{SAGEConv}(\mathbf{h}_m^{(l)}, \text{M2M\_edges})$; conditioned message $\mathbf{m}_l' = \text{GELU}(\text{LayerNorm}(\mathbf{m}_l + \mathbf{C}))$; updated state $\mathbf{h}_m^{(l+1)} = \mathbf{h}_m^{(l)} + \mathbf{m}_l'$; such that the macro-climatic conditioning is applied exactly once per layer at the message level, and the residual $\mathbf{h}_m^{(l)}$ retains the unconditioned representation from the previous layer.

**Claim 4** (Dependent on Claim 1)

The system of Claim 1, wherein static positional features concatenated to each grid node's dynamic atmospheric features comprise $[\sin(\phi), \cos(\phi), \sin(\lambda), \cos(\lambda)]$, providing continuous encoding across the geographic dateline and unambiguous polar identification.

**Claim 5** (Dependent on Claim 1)

The system of Claim 1, wherein said system is trained using a latitude-cosine-weighted mean squared error loss function: $\mathcal{L} = \frac{1}{B} \sum_{b} \frac{1}{N_{\text{grid}}} \sum_{i} w_i \sum_{c} (\hat{y}_{b,i,c} - y_{b,i,c})^2$, wherein $w_i = \cos(\phi_i) / \text{mean}_j\{\cos(\phi_j)\}$, and $\phi_i$ is the geographic latitude of the $i$-th grid node.

**Claim 6** (Dependent on Claim 5)

The system of Claim 5, wherein z-score normalization statistics for each atmospheric channel are computed using the same latitude-cosine-weighted averaging applied across the full spatial-temporal domain of the dataset, such that the global mean and standard deviation reflect the physical surface area distribution of the equirectangular grid rather than the raw grid-cell count, and wherein the same latitude-cosine weight vector governs both the normalization statistics and the training loss, ensuring mathematical consistency between the input feature distribution and the optimization objective.

**Claim 7** (Dependent on Claim 1)

The system of Claim 1, wherein the Mesh-to-Mesh self-neighborhood edge set excludes self-loops by querying k+1 nearest neighbors and removing the first returned index.

**Claim 8** (Independent, Method — Vectorized Batch Expansion for Hardware Acceleration)

A computer-implemented method for accelerating graph neural network execution on a parallel computing device by vectorizing batch-offset expansion of a graph edge index tensor, comprising:

(a) receiving a single-graph edge index tensor `edge_idx` $\in \mathbb{Z}^{2 \times E}$ representing $E$ directed edges between source and target node sets of size $N_{\text{src}}$ and $N_{\text{tgt}}$ respectively;

(b) computing a source offset tensor by multiplying a batch sequence arange(B) by N_src, and repeating the result E times via a hardware-optimized repeat_interleave operation;

(c) computing a target offset tensor by multiplying a batch sequence arange(B) by N_tgt, and repeating the result E times via a hardware-optimized repeat_interleave operation;

(d) computing batched source indices by adding the source offset tensor to a repeated view of the source indices edge_idx[0];

(e) computing batched target indices by adding the target offset tensor to a repeated view of the target indices edge_idx[1]; and

(f) returning a batched edge index tensor of shape $[2, B \times E]$ for parallel graph processing;

wherein operations (b) through (f) map the batch expansion to a single continuous hardware kernel execution without iterative control flow, thereby reducing processing latency on the parallel computing device.

**Claim 9** (Dependent on Claim 8)

The method of Claim 8, applied independently and in parallel to three distinct edge index tensors corresponding to Grid-to-Mesh, Mesh-to-Mesh, and Mesh-to-Grid bipartite graphs within a single forward pass of an Encoder-Processor-Decoder graph neural network.

**Claim 10** (Independent, Method — Full Forward Pass)

A computer-implemented method for neural weather field prediction, comprising:

(a) receiving, for each sample in a batch of $B$ samples: (i) a grid node feature matrix $\mathbf{X}_g \in \mathbb{R}^{N_{\text{grid}} \times F}$, wherein $F = C \times W + 4$ concatenates $C$ atmospheric channels over $W$ input time steps with four static positional features per grid node; and (ii) a conditioning scalar representing the Nino 3.4 sea surface temperature anomaly index at the time of the forecast;

(b) embedding each grid node's features into a d-dimensional latent representation using a linear projection, layer normalization, and GELU activation;

(c) propagating the $d$-dimensional grid node representations to $N_{\text{mesh}}$ spherical Fibonacci lattice mesh nodes via a first bipartite GraphSAGE convolution over Grid-to-Mesh edges, initializing mesh node states to zero;

(d) for each of L processor layers, injecting the conditioning scalar's MLP-projected d-dimensional embedding additively into the GraphSAGE message at each Mesh-to-Mesh message-passing step, maintaining a residual connection over the unconditioned mesh state;

(e) propagating the processed mesh-node representations back to the $N_{\text{grid}}$ observation grid nodes via a second bipartite GraphSAGE convolution over Mesh-to-Grid edges, adding a skip connection from step (b)'s output;

(f) projecting the resulting grid-node representations through a linear layer to produce predicted atmospheric field values; and

(g) computing a training loss as a latitude-cosine-weighted mean squared error between the predicted and true atmospheric field values.

---

---

# PATENT APPLICATION — DOCUMENT 2

## TITLE

UNIFIED MULTI-MODAL CLIMATE INFERENCE SYSTEM WITH INTEGRATED CROSS-DOMAIN TEMPORAL CONDITIONING AND HYBRID PROBABILISTIC-DETERMINISTIC TRAINING OBJECTIVE

---

## FIELD OF THE INVENTION

The present invention relates to integrated machine learning systems for multi-scale climate analysis. More particularly, it relates to a system that jointly operates an ocean-atmospheric forecasting module, a sensor-network pollution dispersion module, a global neural weather module, and a statistical change-point analysis module, sharing a common data ingestion pipeline, unified configuration layer, and metadata store, wherein the Nino 3.4 sea surface temperature anomaly index that the first module is trained to forecast also serves as the conditioning input to the global weather prediction module, establishing a shared-variable cross-domain architecture in which the same geophysical signal drives both ENSO phase classification and atmospheric field prediction.

---

## BACKGROUND

Climate modeling has historically been organized into isolated disciplinary silos. No existing open-source system integrates ENSO prediction, air quality dispersion modeling, global atmospheric field prediction, and Bayesian climate trend analysis in a single reproducible pipeline that shares a common geophysical variable — the Nino 3.4 sea surface temperature anomaly index — as both a prediction target for the ocean-atmospheric module and a conditioning input for the atmospheric field prediction module.

---

## SUMMARY

The present invention provides a unified system comprising:

1. **ENSO Forecasting Module**: A Temporal Fusion Transformer (TFT) configured for multi-horizon prediction of the Nino 3.4 sea surface temperature anomaly index with calibrated quantile regression at tau in {0.1, 0.5, 0.9}.

2. **Pollution Dispersion Module**: An Attention Temporal Graph Convolutional Network (A3TGCN) operating on a sensor proximity graph constructed from real monitoring station coordinates, trained with a mean squared error regression objective and evaluated using a dual-metric framework combining continuous regression metrics with post-hoc threshold-based air quality exceedance classification.

3. **Neural Weather Module**: As described in Patent Application Document 1, conditioned on the observed Nino 3.4 sea surface temperature anomaly index retrieved from the same ERSST data source used to train and evaluate Module 1.

4. **Climate Feedback Module**: A Bayesian piecewise hierarchical regression with CUSUM-based change-point detection as a computational fallback.

5. **Unified Data Pipeline**: A single orchestrator downloading and storing all source data in a common schema with a JSON metadata store and Apache Parquet binary storage.

---

## CLAIMS — PATENT APPLICATION DOCUMENT 2

**Claim 11** (Independent, System)

A unified multi-modal climate analysis system comprising:

(a) a first machine learning module configured to produce probabilistic predictions of an ocean-atmospheric anomaly index at multiple forecast horizons using a sequence-to-sequence attention mechanism with quantile regression;

(b) a second machine learning module configured to predict spatiotemporal air pollutant concentrations across a set of monitoring stations using an Attention Temporal Graph Convolutional Network operating on a spatial proximity graph derived from real geographic station coordinates;

(c) a third machine learning module configured to predict global atmospheric field variables conditioned on the observed anomaly index from the same data source that the first module is trained to predict;

(d) a fourth statistical module configured to perform Bayesian piecewise linear regression on global temperature anomaly time series with NUTS Hamiltonian Monte Carlo posterior sampling, with a CUSUM recursive binary segmentation algorithm as a computational fallback; and

(e) a shared data ingestion pipeline that retrieves, normalizes, and stores data from at least one ocean temperature data source, at least one air quality monitoring network, at least one global temperature anomaly dataset, and at least one atmospheric reanalysis dataset, wherein all persistent data are stored in Apache Parquet binary format indexed by a JSON metadata store tracking ingestion timestamps, source provenance, and model run histories.

**Claim 12** (Dependent on Claim 11)

The system of Claim 11, wherein the first module is a Temporal Fusion Transformer comprising: a Variable Selection Network assigning sparse input weights across time-varying covariates at each encoder step; an LSTM encoder-decoder; a multi-head temporal self-attention layer; and a sum-of-quantile-loss (pinball loss) training objective over quantiles {0.1, 0.5, 0.9}.

**Claim 13** (Dependent on Claim 11)

The system of Claim 11, wherein the spatial proximity graph of the second module is constructed by: (i) converting monitoring station geographic latitude and longitude coordinates into radians; (ii) constructing a spatial index tree configured with a Haversine metric; (iii) querying said spatial index tree to connect each station to its k-nearest neighbors (where k=10) or all neighbors within a maximum kilometer radius; and (iv) assigning inverse-distance edge weights normalized per node.

**Claim 14** (Dependent on Claim 11)

The system of Claim 11, wherein the second module is trained with a mean squared error regression objective and evaluated using a dual-metric framework comprising: (i) continuous regression metrics (RMSE, MAE) computed on raw pollutant concentration predictions; and (ii) binary classification metrics (F1, precision, recall) computed by applying a post-hoc air quality index threshold to the regression predictions and comparing against the same threshold applied to the ground truth, thereby enabling simultaneous assessment of both concentration accuracy and public health exceedance detection capability from a single regression model without requiring an auxiliary classification head or multi-objective training loss.

**Claim 15** (Dependent on Claim 11)

The system of Claim 11, wherein the conditioning input to the third module is the observed Nino 3.4 sea surface temperature anomaly index, retrieved by nearest-month temporal index lookup from a persisted Parquet data store that is shared with the first module's training and evaluation pipeline, such that the same geophysical variable serves as both (i) the prediction target for the first module's quantile regression and (ii) the macro-climatic conditioning signal for the third module's message-passing processor.

---

---

# PATENT APPLICATION — DOCUMENT 3

## TITLE

LATITUDE-COSINE-WEIGHTED NORMALIZATION AND LOSS WEIGHTING FOR TRAINING GRAPH NEURAL NETWORKS ON SPHERICAL EARTH OBSERVATION DATA

---

## FIELD OF THE INVENTION

The present invention relates to training objectives and data normalization methods for machine learning models applied to Earth observation data organized on regular equirectangular latitude-longitude grids.

---

## BACKGROUND

A 5.625 degree equirectangular grid cell at the equator represents approximately 3,400 km^2 of Earth's surface. The same angular cell size at 80 degrees latitude represents approximately 395 km^2. A standard unweighted MSE loss treats every grid cell equally, irrespective of the physical area it represents. Training with unweighted MSE directs optimization effort disproportionately toward polar nodes, which represent smaller physical areas with less climatological impact.

Similarly, computing normalization statistics with uniform node weight produces statistics dominated by the numerically large count of high-latitude nodes, systematically biasing the normalized distribution.

---

## CLAIMS — PATENT APPLICATION DOCUMENT 3

**Claim 16** (Independent, Method)

A computer-implemented method for training a machine learning model on Earth observation data comprising:

(a) providing a dataset of atmospheric field measurements organized as a regular equirectangular latitude-longitude grid with $N_{\text{lat}}$ latitude values and $N_{\text{lon}}$ longitude values, yielding $N_{\text{grid}} = N_{\text{lat}} \times N_{\text{lon}}$ grid nodes;

(b) assigning to each grid node $i$ at latitude $\phi_i$ a normalized latitude weight: $w_i = \cos(\phi_i) / (\frac{1}{N_{\text{grid}}} \sum_j \cos(\phi_j))$;

(c) computing per-channel normalization statistics using the same latitude-cosine-weighted averaging from step (b) across the full spatial-temporal domain of the dataset;

(d) normalizing all data splits using the weighted statistics from step (c); and

(e) computing a training loss as a latitude-cosine-weighted mean squared error: $\mathcal{L} = \frac{1}{B} \sum_{b} \frac{1}{N_{\text{grid}}} \sum_{i} w_i \sum_{c} (\hat{y}_{b,i,c} - y_{b,i,c})^2$.

**Claim 17** (Dependent on Claim 16)

The method of Claim 16, wherein the normalized latitude weight vector $\mathbf{w} \in \mathbb{R}^{N_{\text{grid}}}$ is precomputed once as a fixed floating-point tensor stored in the dataset object and broadcast to shape $[1, N_{\text{grid}}, 1]$ at training time, such that no per-batch latitude weight computation is required.

**Claim 18** (Dependent on Claim 16)

The method of Claim 16, wherein the same latitude-cosine weight vector w from step (b) governs both the normalization statistics in step (c) and the training loss in step (e), such that the input feature distribution and the optimization objective are mathematically consistent under the same physical area weighting, and wherein the weight vector is precomputed once and stored as a fixed tensor to avoid redundant computation across training iterations.

**Claim 19** (Independent, Apparatus)

A non-transitory computer-readable storage medium comprising instructions that, when executed by one or more processors, cause the processors to perform the methods of Claims 8, 10, or 16.

---

## ABSTRACT

A neural weather forecasting system and method are disclosed comprising: (1) a Fibonacci golden-angle spiral lattice generating a near-uniformly distributed spherical latent mesh of configurable resolution; (2) three bipartite GraphSAGE convolution stages (Grid-to-Mesh, Mesh-to-Mesh, Mesh-to-Grid) with KD-Tree nearest-neighbor edge construction in 3D Cartesian unit-sphere space; (3) an ENSO macro-climatic conditioning mechanism that injects the projected Nino 3.4 scalar anomaly index additively into each mesh processor message-passing step, separated from the residual connection to prevent double-counting; (4) a vectorized batch-offset edge index expansion algorithm operating without Python-level loops; and (5) a latitude-cosine-weighted MSE training loss with mathematically consistent latitude-cosine-weighted normalization statistics. When trained on 2020-2022 ERA5 WeatherBench2 reanalysis data at 5.625 degree resolution, the system achieves a best validation MSE of 0.1182 (epoch 78 of 89), representing a 39.5% improvement over initialization. The unified ClimatePulse framework additionally comprises an ENSO forecasting module achieving skill score 0.9051 (RMSE 0.3505 deg C, ACC 0.9449), a pollution ST-GNN achieving F1 0.9750 on PM2.5 exceedance detection across 75 global monitoring stations, and a Bayesian changepoint module detecting zero statistically significant structural breaks in 912 months of global temperature anomaly data.

---

## INVENTOR DECLARATION

I/We declare that the above-identified specification describes my/our invention; that I/we believe myself/ourselves to be the original and sole inventor(s) of the subject matter claimed; and that I/we acknowledge the duty to disclose to the USPTO all information known to me/us to be material to patentability as defined by 37 C.F.R. 1.56.

**Inventor Name(s):** [TO BE COMPLETED]
**Date:** [TO BE COMPLETED]
**Signature:** [TO BE COMPLETED]

---

## REFERENCES FOR PRIOR ART DIFFERENTIATION

The following prior art is known to the inventors and distinguished from the claims above:

1. Lam, R. et al. (2022). "GraphCast." arXiv:2212.12794. [Icosahedral mesh; no ENSO conditioning; no Fibonacci lattice; no latitude-cosine-weighted loss]
2. Keisler, R. (2022). "Forecasting Global Weather with Graph Neural Networks." arXiv:2202.07575. [Icosahedral mesh; no ENSO conditioning]
3. Pathak, J. et al. (2022). "FourCastNet." arXiv:2202.11214. [Regular grid Fourier operators; no graph mesh; no ENSO conditioning]
4. Ham, Y.-G., Kim, J.-H., & Luo, J.-J. (2019). "Deep Learning for Multi-Year ENSO Forecasts." Nature, 573. [Single-task ENSO forecast only; no NWM conditioning downstream]
5. Bai, L. et al. (2020). "A3TGCN for Traffic Forecasting." NeurIPS, 33. [No climate application; no ENSO conditioning; no hybrid loss]
6. Hamilton, W. L. et al. (2017). "GraphSAGE." NeurIPS, 30. [Base SAGEConv method; no spherical mesh, no ENSO conditioning, no climate application]
7. Bi, K. et al. (2023). "Pangu-Weather: Fast and accurate global weather forecasting with 3D neural networks." Nature. [3D pressure levels; no Fibonacci mesh]
8. Chen, L. et al. (2023). "FuXi: a cascade machine learning forecasting system." npj Climate and Atmospheric Science. [Cascade architecture; no Fibonacci mesh]
9. Bonev, B. et al. (2023). "Spherical Fourier Neural Operators: Learning Stable Dynamics on the Sphere." ICML. [SFNO operator; no GNN message-passing over Fibonacci mesh]
10. Kochkov, D. et al. (2024). "Neural General Circulation Models for Weather and Climate." Nature. [Hybrid physics-ML; distinct conditioning approach]
11. Bodnar, C. et al. (2024). "Aurora: A Foundation Model of the Atmosphere." arXiv. [Foundation model; no explicit ENSO causal conditioning pathway]

---

*END OF PATENT APPLICATION DRAFT*
