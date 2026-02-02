<h1 align="center">Reservoir Computing for Chaotic Dynamics</h1>
<div align="center"><i>Comparing Stochastic Echo State Networks and Deterministic NVAR Frameworks.</i></div>
<br>

This project investigates the application of Reservoir Computing (RC) in modeling nonlinear dynamical systems. We contrast the "Classic" stochastic approach (ESN) with the "Next-Gen" deterministic approach (NVAR) to demonstrate how each captures chaotic attractors for autonomous forecasting.

## Introduction

Chaotic systems are characterised by their sensitivity to initial conditions. Reservoir Computing provides a computationally efficient alternative to traditional RNNs by mapping inputs into a high-dimensional feature space where the dynamics can be solved linearly.

![gif](https://github.com/AlvinBLin/reservoir-computing/blob/main/media/lorenz_takens_reconstruction.gif)

### Why Reservoir Computing?

- **Efficient Readout Training:** Both paradigms use Ridge Regression, bypassing the gradient vanishing/exploding problems (BPTT) common in chaotic RNN training [5].

- **Temporal Memory:** History is encoded either through internal hidden dynamics (ESN) or explicit time-delay embeddings (NVAR).

## How It Works
The project implements two distinct architectures to "unfold" the system's state space:

### Classic ESN (Stochastic)

Utilises a fixed random reservoir. The state $\mathbf{x}_t$ evolves as a non-trainable recurrence:

$$\mathbf{x}_{t+1} = (1 - \alpha)\mathbf{x}_t + \alpha \tanh(\mathbf{A}\mathbf{x}_t + \mathbf{C}\mathbf{u}_t)$$

### Next-Gen RC (NVAR / Deterministic)

Eliminates the reservoir in favor of a feature vector $\mathbb{O}_t$ constructed from:

- **Linear Part:** Current and time-shifted observations $[\mathbf{u_t,\ u_{t-k\cdot\text{d}t},\ u_{t-2k\cdot\text{d}t}} \dots]$.

- **Nonlinear Part:** Unique polynomial combinations (monomials) of the linear terms.

## Theoretical Foundation

### Echo State Networks: GS & Embedding

The ESN relies on Generalised Synchronisation (GS). For a reservoir to function as an observer, it must satisfy the Echo State Property (ESP). While Grigoryeva et al. [1] established the embedding properties for linear reservoirs, Hart et al. [6] serve as the critical bridge to the non-linear dynamics used in this project. Hart’s work proves that the synchronisation map from the attractor manifold $M$ to the internal reservoir state $\mathbf{x} \in \mathbb{R}^N$ remains a diffeomorphism (an embedding) for non-linear, leaky-integrated recurrences. This theoretical extension ensures that the topology of the attractor is uniquely unfolded in the high-dimensional reservoir space, maintaining the one-to-one mapping necessary for deterministic, autonomous forecasting.

### Next-Gen RC: Takens & Approximation

The NVAR approach is grounded in Takens’ Embedding Theorem.

1. **The Diffeomorphism:** Let $M$ be the $d$-dimensional attractor manifold. Takens' theorem defines a map $\Phi: M \to \mathbb{R}^k$ such that $$\Phi(x) = (h(x), h(\psi^{-\tau}(x)), \dots, h(\psi^{-(k-1)\tau}(x)))$$. If $k > 2d$, $\Phi$ is an embedding (a diffeomorphism onto its image), providing the necessary coordinates to reconstruct the phase space.

2. **Function Approximation:** Once the state space is reconstructed in $\mathbb{R}^k$, NVAR uses the Universal Approximation Theorem (via polynomial expansion) to approximate the unknown nonlinear map $f: \mathbb{R}^k \to \mathbb{R}$ that governs the system's evolution. By expanding the $k$-dimensional delay vector into a $D$-dimensional polynomial feature space, NVAR solves the forecasting problem as a linear regression.

## Implementation
### 1. Classic ESN: 
Path: `script/ESN.ipynb` 

- Implementation of the stochastic reservoir approach [2]. It features a fixed random neural network that "echoes" input history to create a high-dimensional state space.

### 2. Next Generation Reservoir Computing: 
Path: `script/NextGen_RC.ipynb`

- Implementation of the Next-Generation Reservoir Computer (NVAR) based on Gauthier et al. [7]. This deterministic approach offers several key advantages:

- **Efficiency:** It is $33$ to $162$ times less computationally expensive than traditional RCs.

- **Data Savings:** Requires extremely small training sets (as few as $400$ points) and minimal "warm-up" periods (as few as $2$ points).

- **Interpretability:** Since it utilises explicit polynomial basis functions rather than stochastic random networks, the resulting forecasting model is easier to analyze and relate to the underlying physical equations.

## References

[1] Grigoryeva, L., & Ortega, J.-P. (2021). "Learning strange attractors with reservoir systems." The Annals of Applied Probability, 31(1), 106-127.

[2] Platt, J. A., Penny, S. G., Hunt, B. R., & Kalnay, E. (2022). "A systematic exploration of reservoir computing for forecasting complex spatiotemporal dynamics." Chaos: An Interdisciplinary Journal of Nonlinear Science, 32(1).

[3] Lukoševičius, M. (2012). "A practical guide to applying echo state networks." In Neural Networks: Tricks of the Trade (pp. 659-686). Springer, Berlin, Heidelberg.

[4] Cao, L. (1997). "Practical method for determining the minimum embedding dimension of a scalar time series." Physica D: Nonlinear Phenomena, 110(1-2), 43-50.

[5] Mikhaeil, M., Knüsel, R., Monfared, Z., & Durstewitz, D. (2022). "On the difficulty of learning chaotic dynamics with RNNs." Advances in Neural Information Processing Systems (NeurIPS), 35, 11297-11310.

[6] Hart, A., Hook, J., & Dawes, J. (2020). "Embedding and approximation theorems for echo state networks." Neural Networks, 128, 234-247.
