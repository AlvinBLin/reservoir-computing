<h1 align="center">Reservoir Computing for Chaotic Dynamics</h1>
<div align="center"><i>Reconstructing and forecasting complex dynamical systems via Takens’ Embedding and Echo State Networks.</i></div>
<br>
This project investigates the intersection of Dynamical Systems Theory and Reservoir Computing (RC). We demonstrate how high-dimensional recurrent networks "unfold" chaotic attractors from scalar observations using Generalized Synchronization.

## Contents

- Introduction

- How It Works

- Theoretical Foundation

- Algorithm & Implementation

- References

## Introduction

Real-world systems (e.g., ECG or atmospheric flow) often provide only scalar time series from high-dimensional, unknown nonlinear equations. The challenge is reconstructing the hidden state space for accurate forecasting.

### Why Reservoir Computing?

- **Attractor Tracking:** Acts as a nonlinear observer to reconstruct full phase-space dynamics.

- **Dimensionality:** Maps low-dimensional observations into a high-dimensional feature space.

- **Gradient-Free Training:** Bypasses the vanishing/exploding gradient problems inherent in BPTT.

## How It Works

RC uses a fixed dynamical reservoir and a trainable linear readout.

1. State Update (Fixed)

    The reservoir state $\mathbf{x}_t$ evolves via a non-trainable recurrence relation:

    $$ \mathbf{x}_{t+1} = (1 - \alpha)\mathbf{x}_t + \alpha \tanh(\mathbf{A}\mathbf{x}_t + \mathbf{C}\mathbf{u}_t) $$

    where $\mathbf{A}$ (reservoir) and $\mathbf{C}$ (input) are fixed random matrices, and $\alpha$ is the leaking rate.


3. Linear Readout (Trainable)

   The prediction $\mathbf{Y_t} = \mathbf{W}_{out} \mathbf{x}_t$ is solved via Ridge Regression:

$$\mathbf{W}_{out} = \mathbf{Y}_{target} \mathbf{X}^T (\mathbf{X} \mathbf{X}^T + \beta \mathbf{I})^{-1}$$

   This closed-form solution ensures deterministic, efficient training without backpropagation.

## Theoretical Foundation

### From Linear Theory to Non-linear Embeddings

While Grigoryeva [1] proved embeddings for linear reservoirs, the non-linear form (Platt [2]) is supported by Genericity Theorems (Hart et al. [6]). These establish that for a reservoir with the Echo State Property, the synchronization map is a diffeomorphism if:

- Dimension: $N$ is sufficiently large (e.g., $N=20$ for Lorenz).

- Generic Coupling: $\mathbf{A}$ and $\mathbf{C}$ are chosen from non-singular distributions.

- Stark’s Extension: Injective mapping remains generic even with $\tanh$ activations.

### Embedding vs. Immersion

- Immersion: Smooth mapping that may self-intersect, leading to non-determinism.

- Embedding: One-to-one (injective) mapping preserving topology. We use the Cao Method [4] to ensure the reservoir dimension $d_{min}$ prevents self-intersections.

## Algorithm

1. Phase Space Profiling (Cao_Method.py): Determines $d_{min}$ to size the reservoir correctly.

2. Localized ESNs (Platt_ESN.py): Implements parallel reservoirs for high-dimensional spatiotemporal chaos.

3. Sparse Teacher Forcing (Stabilizer.py): "Nudges" the reservoir toward ground truth to stabilize chaotic orbit learning [5].

## References

[1] Grigoryeva & Ortega. Annals of Applied Probability, 2021.

[2] Platt et al. Chaos, 2022.

[3] Lukoševičius. A practical guide to ESNs, 2012.

[4] Cao, L. Physica D, 1997.

[5] Mikhaeil et al. NeurIPS, 2022.

[6] Hart et al. Neural Networks, 2020.
