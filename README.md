<h1 align="center">Reservoir Computing for Chaotic Dynamics</h1>
<div align="center"><i>Reconstructing and forecasting complex dynamical systems via Takens’ Embedding and Echo State Networks.</i></div>
<br>
This project investigates the intersection of Dynamical Systems Theory and Reservoir Computing (RC). We demonstrate how high-dimensional recurrent networks "unfold" chaotic attractors from scalar observations using Generalized Synchronization.

## Introduction

Real-world systems (e.g., ECG or atmospheric flow) often provide only scalar time series from high-dimensional, unknown nonlinear equations. The challenge is reconstructing the hidden state space for accurate forecasting.

![gif](https://github.com/AlvinBLin/reservoir-computing/blob/main/media/lorenz_takens_reconstruction.gif)

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


2. Linear Readout (Trainable)

    The prediction $\mathbf{Y_t} = \mathbf{W}_{out} \mathbf{x}_t$ is solved via Ridge Regression:

    $$\mathbf{W}_{out} = \mathbf{Y}_{target} \mathbf{X}^T (\mathbf{X} \mathbf{X}^T + \beta \mathbf{I})^{-1}$$

    This closed-form solution ensures deterministic, efficient training without backpropagation.

## Theoretical Foundation

### From Linear Theory to Non-linear Embeddings

While Grigoryeva [1] proved embeddings for linear reservoirs, 

$$\mathbf{x}_t = \mathbf{Ax}_{t-1}+\mathbf{C}\mathbf{u}_t$$

the non-linear form (Platt [2]) is supported by Genericity Theorems (Hart et al. [6]). These establish that for a reservoir with the Echo State Property, the synchronization map is a diffeomorphism if:

- Dimension: $N$ is sufficiently large (e.g., $N=20$ for Lorenz).

- Generic Coupling: $\mathbf{A}$ and $\mathbf{C}$ are chosen from non-singular distributions.

- Stark’s Extension: Injective mapping remains generic even with $\tanh$ activations.

### Embedding vs. Immersion

- Immersion: Smooth mapping that may self-intersect, leading to non-determinism.

- Embedding: One-to-one (injective) mapping preserving topology, preventing self-intersections.

## Algorithm for Prediction

1. Data Preparation (Preprocessing.py)

    Handles normalization of scalar signals and training/testing set generation. This module ensures input data is scaled appropriately for the reservoir's $\tanh$ activation range.

2. Echo State Solver (ESN_Core.py)

    The core implementation of the reservoir and the Ridge Regression solver. It constructs the high-dimensional state space and computes the optimal $\mathbf{W}_{out}$ using the closed-form solution.

3. Multi-step Forecaster (Forecaster.py)

    Implements the autonomous prediction loop. Once trained, the model enters a recursive mode where its own output $\mathbf{\hat{y}}_t$ is fed back as the next input $\mathbf{u}_{t+1}$, enabling long-term forecasting without external data.

## References

[1] Grigoryeva, L., & Ortega, J.-P. (2021). "Learning strange attractors with reservoir systems." The Annals of Applied Probability, 31(1), 106-127.

[2] Platt, J. A., Penny, S. G., Hunt, B. R., & Kalnay, E. (2022). "A systematic exploration of reservoir computing for forecasting complex spatiotemporal dynamics." Chaos: An Interdisciplinary Journal of Nonlinear Science, 32(1).

[3] Lukoševičius, M. (2012). "A practical guide to applying echo state networks." In Neural Networks: Tricks of the Trade (pp. 659-686). Springer, Berlin, Heidelberg.

[4] Cao, L. (1997). "Practical method for determining the minimum embedding dimension of a scalar time series." Physica D: Nonlinear Phenomena, 110(1-2), 43-50.

[5] Mikhaeil, M., Knüsel, R., Monfared, Z., & Durstewitz, D. (2022). "On the difficulty of learning chaotic dynamics with RNNs." Advances in Neural Information Processing Systems (NeurIPS), 35, 11297-11310.

[6] Hart, A., Hook, J., & Dawes, J. (2020). "Embedding and approximation theorems for echo state networks." Neural Networks, 128, 234-247.
