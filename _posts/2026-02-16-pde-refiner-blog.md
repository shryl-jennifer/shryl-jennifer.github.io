---
title: "PDE-Refiner: Achieving Accurate Long Rollouts with Neural PDE Solvers (NeurIPS 2023)"
date: 2026-02-13
permalink: /posts/pde-refiner/
layout: single
author_profile: true
toc: true
toc_sticky: true
tags:
  - Machine Learning
  - Neural PDE Solvers
  - Scientific Computing
  - Diffusion Models
  - NeurIPS
---

## Introduction

Partial Differential Equations (PDEs) are fundamental mathematical tools used to describe physical systems such as fluid dynamics, weather forecasting, heat transfer, and engineering simulations.

Traditionally, PDEs are solved using numerical methods like finite difference, finite element, or spectral methods. While accurate, these approaches are computationally expensive, especially for large-scale simulations or long time horizons.

Recently, neural networks have been proposed as surrogate models for PDE solving. These neural PDE solvers can be significantly faster once trained. However, they suffer from a major limitation:

**They become inaccurate when predicting over long time horizons.**

This paper introduces **PDE-Refiner**, a method that improves long-term accuracy and stability using an iterative refinement process inspired by diffusion models.

---

## Background: Neural PDE Solvers

Consider a time-dependent PDE:

$$
\frac{\partial u}{\partial t} = F(t, x, u, u_x, u_{xx}, \dots)
$$

where:

- u(t,x) is the state
- t is time
- x is spatial position

Neural PDE solvers learn an operator:

$$
\hat{u}(t + \Delta t) = NO(u(t))
$$

This operator predicts the next state from the current state.

To predict multiple steps, the model is applied repeatedly:

$$
\hat{u}(t + 2\Delta t) = NO(\hat{u}(t + \Delta t))
$$

$$
\hat{u}(t + 3\Delta t) = NO(\hat{u}(t + 2\Delta t))
$$

This process is called a **rollout**.

---

## Problem: Why Neural PDE Solvers Fail

Neural PDE solvers are typically trained using Mean Squared Error (MSE):

$$
L_{MSE} = \| u(t) - NO(u(t-\Delta t)) \|^2
$$

This optimizes one-step prediction accuracy.

However, errors accumulate during rollout.

Even small prediction errors compound over time, causing instability.

The paper identifies a key root cause:

> Neural networks neglect low-amplitude spatial frequency components.

Even though these components are small, they strongly influence long-term system behavior.

This leads to unstable rollouts.

---

## Frequency Perspective of PDE Solutions

A PDE solution can be represented as a sum of frequencies:

$$
u(x) = \sum_k a_k \sin(kx)
$$

High-amplitude frequencies dominate short-term behavior.

Low-amplitude frequencies influence long-term dynamics.

Standard training focuses on dominant frequencies and ignores weak ones.

This causes rollout failure.

---

## Key Idea: Iterative Refinement

PDE-Refiner improves predictions using multiple refinement steps.

Instead of predicting once, the model refines its prediction iteratively.

Initial prediction:

$$
\hat{u}_0(t) = NO(u(t-\Delta t))
$$

Then refinement improves this prediction.

---

## Refinement Step Using Gaussian Noise

At step k, Gaussian noise is added:

$$
\tilde{u}_k(t) = \hat{u}_k(t) + \sigma_k \epsilon_k
$$

where

$$
\epsilon_k \sim \mathcal{N}(0,1)
$$

The model predicts the noise:

$$
\hat{\epsilon}_k = NO(\tilde{u}_k(t), u(t-\Delta t), k)
$$

Then the prediction is refined:

$$
\hat{u}_{k+1}(t) =
\tilde{u}_k(t) - \sigma_k \hat{\epsilon}_k
$$

This removes noise and improves accuracy.

---

## Why Noise Helps

Gaussian noise affects all frequencies equally.

This forces the model to learn:

- high-amplitude frequencies
- low-amplitude frequencies

Low-amplitude frequencies are critical for long-term stability.

Without noise, the model ignores them.

---

## Training Objective

The model is trained to predict noise:

$$
L =
\mathbb{E}
\left[
\|
\epsilon_k -
NO(u(t) + \sigma_k \epsilon_k, u(t-\Delta t), k)
\|^2
\right]
$$

This ensures learning across all frequency scales.

---

## Connection to Diffusion Models

Diffusion models work by adding and removing noise iteratively.

PDE-Refiner uses a similar idea, but for refinement instead of generation.

Key difference:

Diffusion models generate data.

PDE-Refiner improves prediction accuracy.

---

## Algorithm Summary

Prediction process:

1. Predict initial state
2. Add noise
3. Predict noise
4. Remove noise
5. Repeat refinement

Final refined prediction:

$$
\hat{u}_K(t)
$$

---

## Experimental Setup

The paper evaluates PDE-Refiner on:

### 1D Kuramoto-Sivashinsky Equation

A chaotic PDE:

$$
u_t + u u_x + u_{xx} + \nu u_{xxxx} = 0
$$

This equation has complex dynamics.

Ideal benchmark for stability testing.

---

## Results: Improved Rollout Stability

Standard neural solver:

Accurate rollout ~75 seconds

PDE-Refiner:

Accurate rollout ~100 seconds

Improvement: ~33%

---

## Kolmogorov Flow Experiment

A 2D fluid dynamics problem based on Navier-Stokes equation:

$$
\partial_t u + \nabla \cdot (u \otimes u)
=
\nu \nabla^2 u
-
\nabla p
+
f
$$

PDE-Refiner outperformed:

- classical solvers
- hybrid ML solvers
- standard neural solvers

---

## Frequency Analysis Results

The paper shows:

- Standard neural solvers ignore low frequencies.

- PDE-Refiner learns all frequencies.

This leads to stable rollouts.

---

## Uncertainty Estimation

By sampling different noise realizations:

$$
\epsilon_k^{(1)}, \epsilon_k^{(2)}, \dots
$$

PDE-Refiner estimates uncertainty.

This predicts when the model becomes unreliable.

---

## Advantages of PDE-Refiner

- Improved stability  
- Better frequency modeling  
- Longer rollout accuracy  
- Uncertainty estimation  
- Better data efficiency  

---

## Limitations

- Higher computational cost due to refinement steps.

- More refinement steps improve accuracy but increase runtime.

---

## Applications

- Fluid simulation  
- Weather forecasting  
- Climate modeling  
- Engineering simulation  
- Scientific computing  

---

## Conclusion

PDE-Refiner introduces a new paradigm for neural PDE solving.
Instead of predicting once, it refines predictions iteratively using denoising.
This ensures accurate modeling of all frequency components.

## Result

Stable, accurate long-term PDE prediction.

---

## Reference

Lippe et al.,  
"PDE-Refiner: Achieving Accurate Long Rollouts with Neural PDE Solvers",  
NeurIPS 2023
