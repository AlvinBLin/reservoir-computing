<h1 align="center">Reservoir Computing for Chaotic Dynamics</h1>


<div align="center"><i>A framework for reconstructing and forecasting complex dynamical systems via Takens’ Embedding and Echo State Networks.</i></div> 
<br>
This research project investigates the intersection of Dynamical Systems Theory and Reservoir Computing (RC). It demonstrates how high-dimensional recurrent networks can "unfold" and forecast chaotic attractors from limited scalar observations by leveraging the principle of Generalized Synchronization.

## Contents

- Introduction

- How It Works

- Theoretical Foundation

- Algorithm & Implementation

- References

## Introduction

In many real-world complex systems—such as cardiac dynamics (ECG) or atmospheric flow—we often only have access to a single observed variable (a scalar time series). However, the underlying physical system is high-dimensional and governed by unknown, coupled nonlinear equations. The challenge lies in reconstructing the "hidden" state space to perform accurate long-term forecasting.

### Why use Reservoir Computing?

**Tracking the Attractor:** RC acts as a nonlinear observer, capable of reconstructing the full phase-space dynamics even when most system variables are unobserved.

**Dimensionality Reduction:** It provides a robust framework for mapping low-dimensional observations into a high-dimensional feature space.

**Efficient Readout Training:** Unlike standard RNNs, RC only requires training a linear output layer (the "readout"), significantly reducing computational costs.

**Acknowledging Training Difficulties:** While the readout training is efficient, it is important to note that the internal reservoir still faces the fundamental "Difficulty of Learning" chaotic dynamics. As indicated by Mikhaeil et al. [5], standard activation functions ($\tanh$, ReLU, or even Identity) do not inherently solve the gradient vanishing or exploding problems when dealing with the positive Lyapunov exponents of chaotic orbits. Our framework addresses this via specialized stabilization techniques.

## How It Works

The Reservoir Computing architecture (specifically the Echo State Network) consists of a fixed, high-dimensional dynamical reservoir and a trainable linear readout.

1. Reservoir State Update (Non-Trainable)

The internal state $\mathbf{x}_t$ of the reservoir evolves according to a nonlinear, leaky-integrated recurrence relation. Unlike traditional RNNs, the matrices $\mathbf{A}$ and $\mathbf{C}$ are fixed and not updated via gradient descent:

$$\mathbf{x}_{t+1} = (1 - \alpha)\mathbf{x}_t + \alpha \tanh(\mathbf{A}\mathbf{x}_t + \mathbf{C}\mathbf{u}_t)$$

2. Linear Readout (Trainable)

The prediction $\mathbf{y}_t$ is computed by a linear combination of the reservoir states. This is the only part of the system that is trained:

$$\mathbf{y}_t = \mathbf{W}_{out} [\mathbf{x}_t; \mathbf{u}_t]$$


## Theoretical Foundation

The success of this approach is grounded in the topological properties of chaotic systems.

### Takens’ Embedding & Generalized Synchronization

Following the work of Grigoryeva et al. [1], we treat the reservoir as a dynamical system that achieves Generalized Synchronization with the input signal. According to Takens’ Embedding Theorem, a sufficient number of delayed observations can reconstruct a space that is topologically equivalent to the original system.

### Embedding vs. Immersion

A critical distinction in this project is the pursuit of a true Embedding rather than a simple Immersion:

- Immersion: A smooth mapping where the image can self-intersect, potentially leading to non-deterministic states where the same "reconstructed" point leads to different futures.

- Embedding: A one-to-one (injective) mapping that preserves the topology. By ensuring our reservoir dimension satisfies the requirements found via the Cao Method [4], we ensure a unique mapping from the reservoir state to the physical attractor, which is a prerequisite for stable forecasting.

### Algorithm

The project is divided into three functional modules:

:pencil2: Phase Space Profiling (Cao_Method.py)

Implementation of Cao’s Method [4] to determine the minimum embedding dimension $d_{min}$. This ensures the reservoir is sized correctly to capture the system's complexity without overfitting.

:pencil2: Localized Echo State Networks (Platt_ESN.py)

Based on Platt et al. [2], this module implements localization for high-dimensional spatiotemporal chaos (e.g., Kuramoto-Sivashinsky equations), utilizing parallel reservoirs to track local dynamics.

:pencil2: Sparse Teacher Forcing (Stabilizer.py)

To address the gradient issues identified by Mikhaeil et al. [5], we implement Sparse Teacher Forcing. This technique "nudges" the reservoir state back to the ground truth at optimal intervals, allowing the network to learn chaotic orbits without numerical divergence.

## References

[1] Grigoryeva, L., & Ortega, J. P. "Learning strange attractors with reservoir systems." Annals of Applied Probability, 2021.

[2] Platt, J. A., et al. "A systematic exploration of reservoir computing for forecasting complex spatiotemporal dynamics." Chaos: An Interdisciplinary Journal of Nonlinear Science, 2022.

[3] Lukoševičius, M. "A practical guide to applying echo state networks." Springer, 2012.

[4] Cao, L. "Practical method for determining the minimum embedding dimension of a scalar time series." Physica D: Nonlinear Phenomena, 1997.

[5] Mikhaeil, M., et al. "On the difficulty of learning chaotic dynamics with RNNs." Advances in Neural Information Processing Systems, 2022.
